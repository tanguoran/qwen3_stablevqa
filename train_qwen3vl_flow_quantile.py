#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a grid-quantile optical-flow encoder on a frozen Qwen3-VL.

Inputs : npz files made by extract_flow_quantiles.py + annotations/{split}_list.csv
Model  : QuantileFlowEncoder -> N soft tokens that replace <|flow_token|> placeholders
         -> frozen Qwen3-VL (text-only) -> last-token hidden state -> regression head
Loss   : MSE on z-scored score + (1 - PLCC) within the batch
All parameters come from configs/train.yaml (no command-line arguments).
"""

import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "train.yaml"

FLOW_TOKEN = "<|flow_token|>"

# Silence the per-matmul "bf16 -> fp16 cast" warning from bitsandbytes 8-bit layers.
logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)

SYSTEM_PROMPT = (
    "You are a video stabilization quality evaluator. "
    "Judge only the perceived stabilization quality of the video."
)
USER_PROMPT = (
    "The sequence above contains injected optical-flow statistics of the video. "
    "Rate its stabilization quality."
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

def read_list(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.reader(f):
            if r:
                rows.append((Path(r[0].strip()).stem, float(r[3])))
    return rows


def compress(q: torch.Tensor, k: float) -> torch.Tensor:
    return torch.sign(q) * torch.log1p(k * q.abs())


class QuantileFlowDataset(Dataset):
    """Loads every npz into memory; features are [T, G*G, C, Q]."""

    def __init__(self, split: str, cfg: Dict[str, Any]):
        ann = ROOT / cfg["paths"]["video_list_dir"] / f"{split}_list.csv"
        feat_dir = ROOT / cfg["paths"]["feature_dir"]
        enc = cfg["encoder"]
        expected = (enc["num_frames"], enc["grid"] ** 2,
                    enc["num_channels"], enc["num_quantiles"])
        lo, hi = cfg["data"]["score_range"]
        k = float(cfg["data"]["log_k"])

        self.names, feats, scores, self.raw_scores = [], [], [], []
        for stem, score in read_list(ann):
            path = feat_dir / f"{stem}.npz"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run extract_flow_quantiles.py")
            with np.load(path) as z:
                q = torch.from_numpy(z["q"].astype(np.float32))
            if tuple(q.shape) != expected:
                raise ValueError(f"{path}: shape {tuple(q.shape)} != {expected}")
            self.names.append(stem)
            feats.append(compress(q, k))
            self.raw_scores.append(score)
            scores.append(min(max(score, lo), hi))

        self.feats = torch.stack(feats)               # [N, T, G2, C, Q]
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.num_clipped = sum(s != r for s, r in zip(scores, self.raw_scores))
        self.feat_mean = None
        self.feat_std = None
        self.score_mean = None
        self.score_std = None

    def fit_stats(self):
        # statistics per (channel, quantile), shared over frames and cells
        self.feat_mean = self.feats.mean(dim=(0, 1, 2), keepdim=True)[0, 0, 0]
        self.feat_std = self.feats.std(dim=(0, 1, 2), keepdim=True)[0, 0, 0].clamp_min(1e-6)
        self.score_mean = float(self.scores.mean())
        self.score_std = float(self.scores.std())

    def set_stats(self, feat_mean, feat_std, score_mean, score_std):
        self.feat_mean, self.feat_std = feat_mean, feat_std
        self.score_mean, self.score_std = score_mean, score_std

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        flow = (self.feats[i] - self.feat_mean) / self.feat_std
        target = (float(self.scores[i]) - self.score_mean) / self.score_std
        return {
            "flow": flow,
            "target": torch.tensor(target, dtype=torch.float32),
            "raw_score": self.raw_scores[i],
            "name": self.names[i],
        }


# ------------------------------------------------------------
# Encoder
# ------------------------------------------------------------

def sincos_2d(grid: int, dim: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding for grid x grid cells -> [grid*grid, dim]."""
    assert dim % 4 == 0
    d = dim // 4
    freqs = torch.exp(-math.log(10000.0) * torch.arange(d) / d)
    pos = torch.arange(grid).float()
    ang = pos[:, None] * freqs[None, :]                          # [grid, d]
    emb1d = torch.cat([ang.sin(), ang.cos()], dim=1)             # [grid, 2d]
    row = emb1d[:, None, :].expand(grid, grid, -1)
    col = emb1d[None, :, :].expand(grid, grid, -1)
    return torch.cat([row, col], dim=-1).reshape(grid * grid, dim)


class QuantileFlowEncoder(nn.Module):
    """
    [B, T, G*G, C, Q] quantile features
      -> per-cell MLP -> + 2D spatial pos-enc + frame embedding
      -> learnable-query cross-attention resampler
      -> [B, N_FLOW_TOKENS, llm_dim]
    """

    def __init__(self, llm_dim: int, enc_cfg: Dict[str, Any]):
        super().__init__()
        G = enc_cfg["grid"]
        in_dim = enc_cfg["num_channels"] * enc_cfg["num_quantiles"]
        raw_dim = enc_cfg["raw_dim"]
        D = enc_cfg["resample_dim"]
        N = enc_cfg["num_flow_tokens"]

        self.num_frames = enc_cfg["num_frames"]
        self.cell_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, raw_dim),
            nn.GELU(),
            nn.Linear(raw_dim, D),
        )
        self.register_buffer("pos_2d", sincos_2d(G, D), persistent=False)
        self.frame_embed = nn.Embedding(self.num_frames, D)
        self.kv_norm = nn.LayerNorm(D)

        self.query = nn.Parameter(torch.randn(1, N, D) * 0.02)
        self.cross_attn = nn.MultiheadAttention(D, enc_cfg["num_heads"], batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D)
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, llm_dim), nn.LayerNorm(llm_dim)
        )
        self.log_scale = nn.Parameter(torch.tensor(float(enc_cfg["init_log_scale"])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, G2, C, Q = x.shape
        if T != self.num_frames:
            raise ValueError(f"T={T} != num_frames={self.num_frames}")
        y = self.cell_mlp(x.reshape(B, T, G2, C * Q))                  # [B,T,G2,D]
        y = y + self.pos_2d[None, None] + self.frame_embed.weight[None, :, None]
        kv = self.kv_norm(y.reshape(B, T * G2, -1))
        z, _ = self.cross_attn(self.query.expand(B, -1, -1), kv, kv, need_weights=False)
        z = z + self.ffn(z)
        return self.projector(z) * self.log_scale.exp()


# ------------------------------------------------------------
# Frozen Qwen + injection + regression head
# ------------------------------------------------------------

class FrozenQwenWithFlow(nn.Module):
    def __init__(self, qwen, flow_encoder, flow_token_id: int, llm_dim: int):
        super().__init__()
        self.qwen = qwen
        self.flow_encoder = flow_encoder
        self.flow_token_id = int(flow_token_id)
        self.head = nn.Sequential(nn.LayerNorm(llm_dim), nn.Linear(llm_dim, 1))
        self._pending = None
        qwen.get_input_embeddings().register_forward_hook(self._embedding_hook)

    def _embedding_hook(self, module, inputs, output):
        if self._pending is None:
            return output
        input_ids = inputs[0]
        if input_ids.ndim != 2:
            return output
        mask = input_ids.eq(self.flow_token_id)
        if not mask.any():
            return output
        tokens = self._pending
        B, N, D = tokens.shape
        counts = mask.sum(dim=1)
        if not torch.all(counts == N):
            raise RuntimeError(f"expected {N} {FLOW_TOKEN} per sample, got {counts.tolist()}")
        out = output.clone()
        out[mask] = tokens.to(device=out.device, dtype=out.dtype).reshape(B * N, D)
        return out

    def forward(self, flow, input_ids, attention_mask):
        self._pending = self.flow_encoder(flow)
        try:
            hidden = self.qwen.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
        finally:
            self._pending = None
        last = attention_mask.sum(dim=1) - 1                      # right padding
        h = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return self.head(h.float()).squeeze(-1).float()


class Collator:
    def __init__(self, processor, num_flow_tokens: int):
        self.processor = processor
        self.flow_block = " ".join([FLOW_TOKEN] * num_flow_tokens)

    def __call__(self, batch: List[Dict[str, Any]]):
        messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "text", "text": self.flow_block + "\n" + USER_PROMPT}]},
            ]
            for _ in batch
        ]
        enc = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "flow": torch.stack([b["flow"] for b in batch]),
            "target": torch.stack([b["target"] for b in batch]),
            "raw_score": [b["raw_score"] for b in batch],
            "name": [b["name"] for b in batch],
        }


# ------------------------------------------------------------
# Metrics / loss
# ------------------------------------------------------------

def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    plcc = float(np.corrcoef(pred, gt)[0, 1])
    srcc = float(np.corrcoef(ranks(pred), ranks(gt))[0, 1])
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return {"plcc": plcc, "srcc": srcc, "rmse": rmse}


@torch.no_grad()
def evaluate(model, ds, collator, device, dtype, batch_size, out_csv=None):
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    names, preds, gts = [], [], []
    for batch in dl:
        with torch.autocast("cuda", dtype=dtype):
            p = model(
                batch["flow"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
        preds += (p.float().cpu().numpy() * ds.score_std + ds.score_mean).tolist()
        gts += batch["raw_score"]
        names += batch["name"]
    preds, gts = np.array(preds), np.array(gts)
    if out_csv is not None:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "gt", "pred"])
            w.writerows(zip(names, gts.tolist(), preds.tolist()))
    return metrics(preds, gts)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def load_qwen(mcfg: Dict[str, Any], processor, dtype):
    kwargs = {}
    if mcfg["quantization"] == "8bit":
        # visual tower / lm_head are removed below anyway; keeping them quantized
        # lets the load fit on small GPUs.
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif mcfg["quantization"] != "none":
        raise ValueError(f"unknown quantization {mcfg['quantization']}")

    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        mcfg["model_id"], torch_dtype=dtype,
        attn_implementation=mcfg["attn_implementation"],
        device_map="cuda", **kwargs,
    )

    # The added <|flow_token|> id is normally already inside the padded embedding
    # table; avoid resizing (it would have to resize a quantized lm_head).
    emb = qwen.get_input_embeddings()
    if len(processor.tokenizer) > emb.weight.shape[0]:
        qwen.resize_token_embeddings(len(processor.tokenizer))

    for p in qwen.parameters():
        p.requires_grad = False
    qwen.eval()
    qwen.config.use_cache = False

    if mcfg["drop_unused"]:
        qwen.model.visual = None
        qwen.lm_head = None
        torch.cuda.empty_cache()
    return qwen


def trainable_state(model):
    return {
        "flow_encoder": model.flow_encoder.state_dict(),
        "head": model.head.state_dict(),
    }


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tc, enc_cfg = cfg["train"], cfg["encoder"]
    seed_everything(tc["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    out_dir = ROOT / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = QuantileFlowDataset("train", cfg)
    train_ds.fit_stats()
    val_ds = QuantileFlowDataset("val", cfg)
    test_ds = QuantileFlowDataset("test", cfg)
    for ds in (val_ds, test_ds):
        ds.set_stats(train_ds.feat_mean, train_ds.feat_std,
                     train_ds.score_mean, train_ds.score_std)
    print(f"train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}, "
          f"train labels clipped: {train_ds.num_clipped}, "
          f"score mean/std = {train_ds.score_mean:.2f}/{train_ds.score_std:.2f}")

    model_id = cfg["model"]["model_id"]
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [FLOW_TOKEN]})

    qwen = load_qwen(cfg["model"], processor, dtype)
    flow_token_id = processor.tokenizer.convert_tokens_to_ids(FLOW_TOKEN)
    llm_dim = qwen.config.text_config.hidden_size
    print(f"llm_dim={llm_dim} flow_token_id={flow_token_id}")

    encoder = QuantileFlowEncoder(llm_dim, enc_cfg).to(device)
    model = FrozenQwenWithFlow(qwen, encoder, flow_token_id, llm_dim)
    model.head.to(device)

    collator = Collator(processor, enc_cfg["num_flow_tokens"])
    dl = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
        num_workers=cfg["data"]["num_workers"], collate_fn=collator, pin_memory=True,
    )
    params = list(encoder.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None

    mse_w, plcc_w = cfg["loss"]["mse_weight"], cfg["loss"]["plcc_weight"]
    best, history = -1e9, []
    stats = {
        "feat_mean": train_ds.feat_mean, "feat_std": train_ds.feat_std,
        "score_mean": train_ds.score_mean, "score_std": train_ds.score_std,
    }

    step = 0
    for epoch in range(1, tc["epochs"] + 1):
        encoder.train()
        model.head.train()
        qwen.eval()
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            with torch.autocast("cuda", dtype=dtype):
                pred = model(
                    batch["flow"].to(device),
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
            target = batch["target"].to(device)
            mse = F.mse_loss(pred, target)
            plcc = pearson(pred, target)
            loss = mse_w * mse + plcc_w * (1.0 - plcc)
            (loss / tc["grad_accum"]).backward() if scaler is None else \
                scaler.scale(loss / tc["grad_accum"]).backward()

            if (i + 1) % tc["grad_accum"] == 0 or i + 1 == len(dl):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"])
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % tc["log_every"] == 0:
                    print(f"epoch={epoch} step={step} loss={loss.item():.4f} "
                          f"mse={mse.item():.4f} plcc={plcc.item():.4f}")

        val = evaluate(model, val_ds, collator, device, dtype, tc["batch_size"])
        score = val["plcc"] + val["srcc"]
        print(f"[val] epoch={epoch} {val}")
        history.append({"epoch": epoch, **val})
        if score > best:
            best = score
            torch.save({**trainable_state(model), "stats": stats, "config": cfg,
                        "epoch": epoch, "val": val}, out_dir / "best.pt")

    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["flow_encoder"])
    model.head.load_state_dict(ckpt["head"])
    test = evaluate(model, test_ds, collator, device, dtype, tc["batch_size"],
                    out_csv=out_dir / "test_predictions.csv")
    print(f"[test] best_epoch={ckpt['epoch']} {test}")
    (out_dir / "metrics.json").write_text(
        json.dumps({"history": history, "best_epoch": ckpt["epoch"], "test": test}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
