#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For every video listed in annotations/{split}_list.csv:
  fixed 64-frame window -> RAFT flow on consecutive frames -> per-grid-cell
  quantiles of (dx, dy, |flow|) -> one npz per video.

All parameters come from configs/preprocess.yaml (no command-line arguments).

npz contents:
  q            float16 [T-1, G*G, 3, Q]   flow / short_side, channels dx, dy, mag
  quantiles    int     [Q]
  start_frame, num_total_frames, fps, orig_hw, flow_hw, cfg_hash
"""

import csv
import hashlib
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    Raft_Small_Weights,
    raft_large,
    raft_small,
)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "preprocess.yaml"
FEATURE_VERSION = 1


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg) -> str:
    key = {
        "v": FEATURE_VERSION,
        "window": cfg["window"],
        "short_side": cfg["raft"]["short_side"],
        "model": cfg["raft"]["model"],
        "iters": cfg["raft"]["num_flow_updates"],
        "grid": cfg["features"]["grid"],
        "quantiles": cfg["features"]["quantiles"],
    }
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def read_list(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [row[0].strip() for row in csv.reader(f) if row]


def target_size(h: int, w: int, short_side: int):
    scale = short_side / min(h, w)
    nh = max(8, int(round(h * scale / 8)) * 8)
    nw = max(8, int(round(w * scale / 8)) * 8)
    return nh, nw


def read_window(video_path: Path, num_frames: int, position: str, size_hw):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if total < num_frames:
        raise RuntimeError(f"only {total} frames, need {num_frames}")

    start = (total - num_frames) // 2 if position == "center" else 0
    nh, nw = size_hw(orig_h, orig_w)

    frames = []
    for i in range(start + num_frames):
        if i < start:
            if not cap.grab():
                break
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA))
    cap.release()

    if len(frames) != num_frames:
        raise RuntimeError(f"decoded {len(frames)} frames, need {num_frames}")
    return np.stack(frames), start, total, fps, (orig_h, orig_w)


@torch.no_grad()
def compute_flow(model, frames_u8, batch_pairs, num_updates, amp, device):
    x = torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float()
    x = x / 255.0 * 2.0 - 1.0
    outs = []
    for i in range(0, x.shape[0] - 1, batch_pairs):
        a = x[i:i + batch_pairs]
        b = x[i + 1:i + 1 + batch_pairs]
        n = min(a.shape[0], b.shape[0])
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            flow = model(a[:n], b[:n], num_flow_updates=num_updates)[-1]
        outs.append(flow.float())
    return torch.cat(outs, dim=0)  # [T-1, 2, H, W]


@torch.no_grad()
def grid_quantiles(flow, grid, quantiles, short_side):
    flow = flow / float(short_side)
    mag = flow.norm(dim=1, keepdim=True)
    feat = torch.cat([flow, mag], dim=1)  # [T-1, 3, H, W]
    T, C, H, W = feat.shape
    qs = torch.tensor(quantiles, dtype=feat.dtype, device=feat.device) / 100.0

    rows = torch.tensor_split(torch.arange(H, device=feat.device), grid)
    cols = torch.tensor_split(torch.arange(W, device=feat.device), grid)
    cells = []
    for r in rows:
        for c in cols:
            patch = feat[:, :, r[0]:r[-1] + 1, c[0]:c[-1] + 1].reshape(T, C, -1)
            cells.append(torch.quantile(patch, qs, dim=-1))  # [Q, T, C]
    out = torch.stack(cells, dim=0)  # [G*G, Q, T, C]
    return out.permute(2, 0, 3, 1).contiguous()  # [T, G*G, C, Q]


def build_raft(name, device):
    if name == "raft_large":
        model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    elif name == "raft_small":
        model = raft_small(weights=Raft_Small_Weights.DEFAULT)
    else:
        raise ValueError(f"unknown raft model {name}")
    return model.to(device).eval()


def main():
    cfg = load_config()
    paths = cfg["paths"]
    video_dir = ROOT / paths["video_dir"]
    ann_dir = ROOT / paths["annotation_dir"]
    out_dir = ROOT / paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["raft"]["device"])
    model = build_raft(cfg["raft"]["model"], device)
    short_side = int(cfg["raft"]["short_side"])
    chash = config_hash(cfg)
    quantiles = [int(q) for q in cfg["features"]["quantiles"]]
    grid = int(cfg["features"]["grid"])
    num_frames = int(cfg["window"]["num_frames"])
    overwrite = bool(cfg["run"]["overwrite"])

    names = []
    for split in cfg["splits"]:
        names += read_list(ann_dir / f"{split}_list.csv")
    names = list(dict.fromkeys(names))
    print(f"{len(names)} videos, cfg_hash={chash}")

    failed = []
    t0 = time.time()
    for n, name in enumerate(names, 1):
        out_path = out_dir / f"{Path(name).stem}.npz"
        if out_path.exists() and not overwrite:
            try:
                with np.load(out_path) as z:
                    if str(z["cfg_hash"]) == chash:
                        continue
            except Exception:
                pass
        try:
            frames, start, total, fps, orig_hw = read_window(
                video_dir / name,
                num_frames,
                cfg["window"]["position"],
                lambda h, w: target_size(h, w, short_side),
            )
            flow = compute_flow(
                model,
                frames,
                int(cfg["raft"]["batch_pairs"]),
                int(cfg["raft"]["num_flow_updates"]),
                bool(cfg["raft"]["amp"]),
                device,
            )
            q = grid_quantiles(flow, grid, quantiles, min(frames.shape[1:3])).cpu().numpy()
            tmp = out_path.with_name(out_path.stem + ".tmp.npz")
            np.savez_compressed(
                tmp,
                q=q.astype(np.float16),
                quantiles=np.array(quantiles, dtype=np.int16),
                start_frame=start,
                num_total_frames=total,
                fps=fps,
                orig_hw=np.array(orig_hw),
                flow_hw=np.array(frames.shape[1:3]),
                cfg_hash=chash,
            )
            tmp.replace(out_path)
        except Exception as e:
            failed.append(name)
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()

        if n % 20 == 0 or n == len(names):
            el = time.time() - t0
            print(f"{n}/{len(names)} elapsed={el:.0f}s")

    print(f"done, failed={len(failed)}")
    if failed:
        (out_dir / "failed.txt").write_text("\n".join(failed), encoding="utf-8")


if __name__ == "__main__":
    main()
