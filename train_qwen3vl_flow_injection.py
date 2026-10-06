#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train optical-flow soft-token injection on a frozen Qwen3-VL.

Two modes:
  1) --mode flow      : only optical-flow images + text prompt
  2) --mode rgb_flow  : Qwen3-VL RGB images + optical-flow soft tokens + text prompt

The script assumes optical-flow visualization images are already precomputed.

Manifest JSONL example:
{"flow":["flow/v001_000.png","flow/v001_001.png"],"rgb":["rgb/v001_000.jpg","rgb/v001_001.jpg"],"score":4}
{"flow":["flow/v002_000.png","flow/v002_001.png"],"rgb":["rgb/v002_000.jpg","rgb/v002_001.jpg"],"score":2}

Paths may be absolute or relative to the manifest directory.

Recommended workflow:
  - First debug with Qwen/Qwen3-VL-8B-Instruct.
  - After correctness is verified, switch --model_id to Qwen/Qwen3-VL-32B-Instruct
    and use an appropriate multi-GPU/FSDP/DeepSpeed setup.

Dependencies:
  pip install -U torch torchvision transformers accelerate pillow

This is intentionally a minimal research training skeleton:
- Qwen3-VL is frozen.
- Only FlowTokenEncoder is trained.
- Flow tokens replace embeddings of a custom <|flow_token|> placeholder.
- In rgb_flow mode, Qwen3-VL's native image pipeline remains unchanged.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


FLOW_TOKEN = "<|flow_token|>"

SYSTEM_PROMPT = (
    "You are a video stabilization quality evaluator. "
    "Judge only the perceived stabilization quality of the video."
)

USER_PROMPT = """The sequence contains injected optical-flow information.
Evaluate video stabilization quality on a 1-5 scale:
1 = very unstable
2 = unstable
3 = moderate
4 = stable
5 = very stable

Return exactly:
Stability: <1-5>
"""


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(base: Path, p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = base / path
    return str(path)


def uniform_select(paths: List[str], n: int) -> List[str]:
    """Uniformly sample n paths; repeat endpoints if too short."""
    if len(paths) == 0:
        raise ValueError("Empty frame list.")
    if len(paths) == n:
        return paths
    if len(paths) == 1:
        return paths * n

    idx = torch.linspace(0, len(paths) - 1, steps=n).round().long().tolist()
    return [paths[i] for i in idx]


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class StabilityDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        mode: str,
        num_flow_frames: int,
        num_rgb_frames: int,
        flow_size: int,
    ):
        self.manifest = Path(manifest)
        self.base = self.manifest.parent
        self.mode = mode
        self.num_flow_frames = num_flow_frames
        self.num_rgb_frames = num_rgb_frames

        with open(self.manifest, "r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

        self.flow_tf = transforms.Compose([
            transforms.Resize((flow_size, flow_size)),
            transforms.ToTensor(),
            # Do NOT use hue/color jitter for color-coded flow visualizations.
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        flow_paths = [resolve_path(self.base, p) for p in item["flow"]]
        flow_paths = uniform_select(flow_paths, self.num_flow_frames)

        flow = []
        for p in flow_paths:
            img = Image.open(p).convert("RGB")
            flow.append(self.flow_tf(img))
        flow = torch.stack(flow, dim=0)   # [T, 3, H, W]

        rgb_images = None
        if self.mode == "rgb_flow":
            if "rgb" not in item:
                raise ValueError("rgb_flow mode requires an 'rgb' list in each manifest item.")
            rgb_paths = [resolve_path(self.base, p) for p in item["rgb"]]
            rgb_paths = uniform_select(rgb_paths, self.num_rgb_frames)
            rgb_images = [Image.open(p).convert("RGB") for p in rgb_paths]

        score = int(item["score"])
        if score < 1 or score > 5:
            raise ValueError(f"score must be 1..5, got {score}")

        return {
            "flow": flow,
            "rgb_images": rgb_images,
            "score": score,
        }


# ------------------------------------------------------------
# Flow -> N soft tokens -> Qwen hidden dimension
# ------------------------------------------------------------

class FlowTokenEncoder(nn.Module):
    """
    Flow visualization images:
        [B, T, 3, H, W]
          -> spatial features per frame
          -> raw spatiotemporal tokens
          -> learnable-query resampler
          -> [B, N_FLOW_TOKENS, D_LLM]
    """
    def __init__(
        self,
        llm_dim: int,
        num_flow_tokens: int = 16,
        max_flow_frames: int = 32,
        raw_dim: int = 256,
        resample_dim: int = 512,
        pooled_grid: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_flow_tokens = num_flow_tokens
        self.pooled_grid = pooled_grid

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, raw_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((pooled_grid, pooled_grid)),
        )

        self.frame_embed = nn.Embedding(max_flow_frames, raw_dim)
        self.raw_proj = nn.Linear(raw_dim, resample_dim)

        self.query = nn.Parameter(
            torch.randn(1, num_flow_tokens, resample_dim) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=resample_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(resample_dim),
            nn.Linear(resample_dim, resample_dim * 4),
            nn.GELU(),
            nn.Linear(resample_dim * 4, resample_dim),
        )

        self.projector = nn.Sequential(
            nn.LayerNorm(resample_dim),
            nn.Linear(resample_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

        # Small trainable injection scale helps avoid a very disruptive random start.
        self.log_scale = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        if T > self.frame_embed.num_embeddings:
            raise ValueError(
                f"T={T} > max_flow_frames={self.frame_embed.num_embeddings}"
            )

        y = self.cnn(x.reshape(B * T, C, H, W))
        # [B*T, raw_dim, G, G]
        G = self.pooled_grid
        y = y.flatten(2).transpose(1, 2)   # [B*T, G*G, raw_dim]
        y = y.reshape(B, T, G * G, -1)

        frame_ids = torch.arange(T, device=x.device)
        frame_e = self.frame_embed(frame_ids)[None, :, None, :]  # [1,T,1,D]
        y = y + frame_e

        y = y.reshape(B, T * G * G, -1)
        kv = self.raw_proj(y)

        q = self.query.expand(B, -1, -1)
        z, _ = self.cross_attn(q, kv, kv, need_weights=False)
        z = z + self.ffn(z)
        z = self.projector(z)

        scale = self.log_scale.exp()
        return z * scale


# ------------------------------------------------------------
# Qwen wrapper using an embedding forward hook
# ------------------------------------------------------------

class FrozenQwenWithFlow(nn.Module):
    """
    Keeps Qwen's normal forward intact.

    The custom flow placeholder token is first embedded normally.
    A forward hook on Qwen's input embedding layer replaces those positions
    with FlowTokenEncoder outputs.

    This lets rgb_flow mode keep Qwen3-VL's own image placeholder handling,
    vision encoder, multimodal RoPE, and visual feature scatter logic.
    """
    def __init__(
        self,
        qwen: Qwen3VLForConditionalGeneration,
        flow_encoder: FlowTokenEncoder,
        flow_token_id: int,
    ):
        super().__init__()
        self.qwen = qwen
        self.flow_encoder = flow_encoder
        self.flow_token_id = int(flow_token_id)
        self._pending_flow_tokens = None

        emb = self.qwen.get_input_embeddings()
        self._hook_handle = emb.register_forward_hook(self._embedding_hook)

    def _embedding_hook(self, module, inputs, output):
        if self._pending_flow_tokens is None:
            return output

        input_ids = inputs[0]
        # Ignore any unrelated embedding call.
        if input_ids.ndim != 2:
            return output

        mask = input_ids.eq(self.flow_token_id)
        if not mask.any():
            return output

        flow_tokens = self._pending_flow_tokens
        B, N, D = flow_tokens.shape

        counts = mask.sum(dim=1)
        if not torch.all(counts == N):
            raise RuntimeError(
                f"Each sample must contain exactly {N} {FLOW_TOKEN} placeholders; "
                f"found counts={counts.tolist()}"
            )

        out = output.clone()
        flow_tokens = flow_tokens.to(device=out.device, dtype=out.dtype)

        # Boolean indexing is row-major here: sample 0 tokens, then sample 1, ...
        out[mask] = flow_tokens.reshape(B * N, D)
        return out

    def forward(self, flow_images, **qwen_inputs):
        flow_tokens = self.flow_encoder(flow_images)
        self._pending_flow_tokens = flow_tokens
        try:
            outputs = self.qwen(
                **qwen_inputs,
                use_cache=False,
            )
        finally:
            self._pending_flow_tokens = None
        return outputs


# ------------------------------------------------------------
# Collator
# ------------------------------------------------------------

class TrainCollator:
    def __init__(
        self,
        processor,
        mode: str,
        num_flow_tokens: int,
    ):
        self.processor = processor
        self.mode = mode
        self.num_flow_tokens = num_flow_tokens
        self.flow_block = " ".join([FLOW_TOKEN] * num_flow_tokens)

    def _messages(self, sample: Dict[str, Any], include_answer: bool):
        content = []

        if self.mode == "rgb_flow":
            for img in sample["rgb_images"]:
                content.append({
                    "type": "image",
                    "image": img,
                })

        content.append({
            "type": "text",
            "text": (
                self.flow_block
                + "\n"
                + USER_PROMPT
            ),
        })

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        if include_answer:
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"Stability: {sample['score']}",
                }],
            })

        return messages

    def __call__(self, batch: List[Dict[str, Any]]):
        full_messages = [
            self._messages(x, include_answer=True)
            for x in batch
        ]

        prompt_messages = [
            self._messages(x, include_answer=False)
            for x in batch
        ]

        # Full training sequence, including assistant answer.
        full = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Same prompt but with assistant-generation prefix.
        # Used only to find where the supervised answer begins.
        prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )

        labels = full["input_ids"].clone()
        prompt_lens = prompt["attention_mask"].sum(dim=1).tolist()

        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        labels[full["attention_mask"] == 0] = -100
        full["labels"] = labels

        flow = torch.stack([x["flow"] for x in batch], dim=0)

        # Some processors may return token_type_ids; Qwen3-VL does not need it here.
        full.pop("token_type_ids", None)

        return {
            "qwen_inputs": full,
            "flow": flow,
        }


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mode", choices=["flow", "rgb_flow"], default="flow")

    # Use 8B to validate the pipeline first. Swap to 32B after it works.
    ap.add_argument(
        "--model_id",
        default="Qwen/Qwen3-VL-8B-Instruct",
    )
    ap.add_argument("--output_dir", default="./flow_injection_ckpt")

    ap.add_argument("--num_flow_frames", type=int, default=8)
    ap.add_argument("--num_rgb_frames", type=int, default=4)
    ap.add_argument("--num_flow_tokens", type=int, default=16)
    ap.add_argument("--flow_size", type=int, default=224)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This training script expects a CUDA GPU.")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading {args.model_id} ...")

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"

    # Add a single special placeholder token.
    added = processor.tokenizer.add_special_tokens({
        "additional_special_tokens": [FLOW_TOKEN]
    })

    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )

    if added > 0:
        qwen.resize_token_embeddings(len(processor.tokenizer))

    # Freeze ALL Qwen / vision parameters.
    for p in qwen.parameters():
        p.requires_grad = False

    qwen.eval()
    qwen.config.use_cache = False
    qwen.to(device)

    flow_token_id = processor.tokenizer.convert_tokens_to_ids(FLOW_TOKEN)
    llm_dim = qwen.config.text_config.hidden_size

    print(f"LLM hidden dim = {llm_dim}")
    print(f"flow_token_id = {flow_token_id}")

    flow_encoder = FlowTokenEncoder(
        llm_dim=llm_dim,
        num_flow_tokens=args.num_flow_tokens,
        max_flow_frames=max(args.num_flow_frames, 32),
    ).to(device)

    model = FrozenQwenWithFlow(
        qwen=qwen,
        flow_encoder=flow_encoder,
        flow_token_id=flow_token_id,
    )

    ds = StabilityDataset(
        manifest=args.manifest,
        mode=args.mode,
        num_flow_frames=args.num_flow_frames,
        num_rgb_frames=args.num_rgb_frames,
        flow_size=args.flow_size,
    )

    collator = TrainCollator(
        processor=processor,
        mode=args.mode,
        num_flow_tokens=args.num_flow_tokens,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        flow_encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = None
    # bfloat16 does not need GradScaler; fp16 does.
    if dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        flow_encoder.train()
        qwen.eval()

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dl):
            flow = batch["flow"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            qwen_inputs = {}
            for k, v in batch["qwen_inputs"].items():
                if torch.is_tensor(v):
                    qwen_inputs[k] = v.to(device, non_blocking=True)
                else:
                    qwen_inputs[k] = v

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                outputs = model(
                    flow_images=flow,
                    **qwen_inputs,
                )
                loss = outputs.loss / args.grad_accum

            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            do_step = ((step + 1) % args.grad_accum == 0) or (step + 1 == len(dl))

            if do_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    flow_encoder.parameters(),
                    args.max_grad_norm,
                )

                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                print(
                    f"epoch={epoch+1}/{args.epochs} "
                    f"step={global_step} "
                    f"loss={loss.item() * args.grad_accum:.4f}"
                )

        # Save only the trainable part + processor with added token.
        ckpt = {
            "flow_encoder": flow_encoder.state_dict(),
            "model_id": args.model_id,
            "flow_token": FLOW_TOKEN,
            "flow_token_id": flow_token_id,
            "num_flow_tokens": args.num_flow_tokens,
            "num_flow_frames": args.num_flow_frames,
            "mode": args.mode,
            "llm_dim": llm_dim,
        }
        torch.save(
            ckpt,
            os.path.join(args.output_dir, f"flow_encoder_epoch{epoch+1}.pt"),
        )
        processor.save_pretrained(args.output_dir)

    print("Training finished.")


if __name__ == "__main__":
    main()
