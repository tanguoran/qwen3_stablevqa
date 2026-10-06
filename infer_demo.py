#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference demo: video -> 65-frame window -> RAFT -> grid quantiles
                -> QuantileFlowEncoder + frozen Qwen3-VL -> regression head -> score.

All parameters come from configs/infer.yaml (no command-line arguments).
RAFT / window / grid settings come from configs/preprocess.yaml and must match training.
"""

import csv
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor

import extract_flow_quantiles as E
import train_qwen3vl_flow_quantile as T

ROOT = Path(__file__).resolve().parent
INFER_CONFIG = ROOT / "configs" / "infer.yaml"


def load_gt(ann_dir: Path):
    gt = {}
    for split in ("train", "val", "test"):
        p = ann_dir / f"{split}_list.csv"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for r in csv.reader(f):
                if r:
                    gt[Path(r[0].strip()).stem] = (split, float(r[3]))
    return gt


@torch.no_grad()
def video_to_features(path: Path, pcfg, raft, device):
    short_side = int(pcfg["raft"]["short_side"])
    frames, start, total, _, _ = E.read_window(
        path, int(pcfg["window"]["num_frames"]), pcfg["window"]["position"],
        lambda h, w: E.target_size(h, w, short_side),
    )
    flow = E.compute_flow(
        raft, frames, int(pcfg["raft"]["batch_pairs"]),
        int(pcfg["raft"]["num_flow_updates"]), bool(pcfg["raft"]["amp"]), device,
    )
    q = E.grid_quantiles(
        flow, int(pcfg["features"]["grid"]),
        [int(x) for x in pcfg["features"]["quantiles"]], min(frames.shape[1:3]),
    ).cpu()
    q = q.half().float()  # same fp16 rounding as the saved npz files
    return q, start, total


def main():
    with open(INFER_CONFIG, "r", encoding="utf-8") as f:
        icfg = yaml.safe_load(f)
    with open(E.CONFIG_PATH, "r", encoding="utf-8") as f:
        pcfg = yaml.safe_load(f)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    ckpt = torch.load(ROOT / icfg["checkpoint"], map_location="cpu", weights_only=False)
    tcfg, stats = ckpt["config"], ckpt["stats"]
    enc_cfg = tcfg["encoder"]
    expected_frames = int(pcfg["window"]["num_frames"]) - 1
    if enc_cfg["num_frames"] != expected_frames or enc_cfg["grid"] != int(pcfg["features"]["grid"]):
        raise ValueError("configs/preprocess.yaml does not match the checkpoint's encoder config")
    log_k = float(tcfg["data"]["log_k"])

    processor = AutoProcessor.from_pretrained(tcfg["model"]["model_id"])
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [T.FLOW_TOKEN]})
    qwen = T.load_qwen(tcfg["model"], processor, dtype)

    llm_dim = qwen.config.text_config.hidden_size
    encoder = T.QuantileFlowEncoder(llm_dim, enc_cfg).to(device)
    encoder.load_state_dict(ckpt["flow_encoder"])
    model = T.FrozenQwenWithFlow(
        qwen, encoder, processor.tokenizer.convert_tokens_to_ids(T.FLOW_TOKEN), llm_dim
    )
    model.head.to(device)
    model.head.load_state_dict(ckpt["head"])
    encoder.eval()
    model.head.eval()
    collator = T.Collator(processor, enc_cfg["num_flow_tokens"])

    raft = E.build_raft(pcfg["raft"]["model"], device)
    gt = load_gt(ROOT / icfg["annotation_dir"])
    feat_mean = stats["feat_mean"].to(device)
    feat_std = stats["feat_std"].to(device)

    print(f"checkpoint epoch={ckpt['epoch']} val={ckpt['val']}")
    for v in icfg["videos"]:
        path = Path(v)
        path = path if path.is_absolute() else ROOT / path
        t0 = time.time()
        q, start, total = video_to_features(path, pcfg, raft, device)
        flow = (T.compress(q, log_k).to(device) - feat_mean) / feat_std

        batch = collator([{"flow": flow, "target": torch.zeros(()), "raw_score": 0, "name": path.stem}])
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            pred = model(
                flow[None].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
        score = float(pred.item()) * stats["score_std"] + stats["score_mean"]

        line = (f"{path.name}: predicted stability = {score:.1f} "
                f"(window frames {start}-{start + int(pcfg['window']['num_frames']) - 1} of {total}, "
                f"{time.time() - t0:.1f}s)")
        if path.stem in gt:
            split, g = gt[path.stem]
            line += f" | ground truth = {g:.1f} [{split}]"
        print(line)


if __name__ == "__main__":
    main()
