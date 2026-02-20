#!/usr/bin/env python3
"""Train GlyphSuperRes CNN on (32x32, 256x256) image pairs.

Usage:
    uv run scripts/train_superres.py --data datasets/abstract-v2/superres-abstract-v2-10k.pt
    # -> training-runs/superres-abstract-v2/ (auto-derived)

    # Override output dir
    uv run scripts/train_superres.py \
        --data datasets/abstract-v2/superres-abstract-v2-10k.pt \
        --outdir training-runs/custom

    # Shorter run for testing
    uv run scripts/train_superres.py \
        --data datasets/abstract-v2/superres-abstract-v2-10k.pt \
        --kimg 100 --batch 32
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easey_glyph.model.superres import GlyphSuperRes


def derive_outdir(data_path: str) -> str:
    """Derive training output directory from dataset path.

    Convention: datasets/{type}/superres-{type}-{count}.pt -> training-runs/superres-{type}/
    Fallback: use parent directory name with superres- prefix, or 'superres' if at top level.
    """
    p = Path(data_path)
    parent_name = p.parent.name
    stem = p.stem
    # If filename starts with superres-, extract the type
    if stem.startswith("superres-"):
        parts = stem[len("superres-"):].split("-")
        # Remove trailing count segment (e.g. "10k")
        if len(parts) > 1:
            return f"training-runs/superres-{'-'.join(parts[:-1])}"
        return f"training-runs/superres-{parts[0]}"
    # Fallback: use parent directory name
    if parent_name and parent_name != "datasets":
        return f"training-runs/superres-{parent_name}"
    return "training-runs/superres"


@torch.no_grad()
def save_snapshot(model, low_res, high_res, step, outdir, device, num_samples=8):
    """Save a comparison grid: input | CNN output | target."""
    model.eval()
    n = min(num_samples, low_res.shape[0])
    lr = low_res[:n].to(device).float() / 255.0
    hr = high_res[:n].float() / 255.0
    sr = model(lr).cpu()

    # Upscale low-res to 256 for visual comparison
    lr_up = torch.nn.functional.interpolate(lr.cpu(), size=256, mode="bilinear", align_corners=False)

    # Build grid: 3 columns (input bilinear | CNN output | target) x N rows
    h, w = 256, 256
    grid_img = np.zeros((n * h, 3 * w, 3), dtype=np.uint8)
    for i in range(n):
        for j, tensor in enumerate([lr_up[i], sr[i], hr[i]]):
            pixels = (tensor.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
            grid_img[i * h:(i + 1) * h, j * w:(j + 1) * w] = pixels

    Image.fromarray(grid_img).save(outdir / f"snapshot_{step:06d}.png")
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train GlyphSuperRes CNN")
    parser.add_argument("--data", type=str, required=True, help="Path to superres-pairs .pt file")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (auto-derived from data path if omitted)")
    parser.add_argument("--kimg", type=float, default=500, help="Training duration in thousands of images")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--base-channels", type=int, default=64, help="Base channel count")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--snap-interval", type=int, default=50, help="Snapshot every N kimg")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = derive_outdir(args.data)
        print(f"Output: {args.outdir} (auto-derived from data path)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "snapshots").mkdir(exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}...")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    low_res = data["low_res"]    # [N, 3, 32, 32]
    high_res = data["high_res"]  # [N, 3, 256, 256]
    n = low_res.shape[0]
    print(f"  {n} image pairs loaded")

    dataset = TensorDataset(low_res, high_res)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    # Model
    model = GlyphSuperRes(
        in_channels=3,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"GlyphSuperRes: {params:,} params ({params / 1e6:.2f}M)")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = int(args.kimg * 1000 / args.batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training
    print(f"Training for {args.kimg:.0f} kimg ({total_steps} steps, batch={args.batch})")
    print(f"  Device: {device}")
    print(f"  Snapshots: every {args.snap_interval} kimg -> {outdir / 'snapshots'}")

    model.train()
    step = 0
    kimg_shown = 0
    t0 = time.time()
    last_snap_kimg = -1
    running_loss = 0.0
    loss_count = 0

    while kimg_shown < args.kimg:
        for lr_batch, hr_batch in loader:
            if kimg_shown >= args.kimg:
                break

            lr_batch = lr_batch.to(device).float() / 255.0
            hr_batch = hr_batch.to(device).float() / 255.0

            # Random horizontal flip (same flip for both)
            if torch.rand(1).item() > 0.5:
                lr_batch = lr_batch.flip(-1)
                hr_batch = hr_batch.flip(-1)

            sr = model(lr_batch)
            loss = torch.nn.functional.l1_loss(sr, hr_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            kimg_shown = step * args.batch / 1000
            running_loss += loss.item()
            loss_count += 1

            # Log
            if step % 50 == 0:
                avg_loss = running_loss / loss_count
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                kimg_per_sec = kimg_shown / elapsed if elapsed > 0 else 0
                print(f"  step {step:6d} | kimg {kimg_shown:7.1f}/{args.kimg:.0f} | "
                      f"L1 {avg_loss:.4f} | lr {lr_now:.2e} | "
                      f"{kimg_per_sec:.1f} kimg/s", flush=True)
                running_loss = 0.0
                loss_count = 0

            # Snapshot
            snap_kimg = int(kimg_shown / args.snap_interval)
            if snap_kimg > last_snap_kimg and kimg_shown > 0:
                last_snap_kimg = snap_kimg
                save_snapshot(model, low_res, high_res, step, outdir / "snapshots", device)
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "kimg": kimg_shown,
                    "config": {
                        "in_channels": 3,
                        "base_channels": args.base_channels,
                        "num_blocks": args.num_blocks,
                    },
                }
                torch.save(ckpt, outdir / f"checkpoint_{step:06d}.pt")
                print(f"  Saved snapshot + checkpoint at step {step}", flush=True)

    # Final save
    save_snapshot(model, low_res, high_res, step, outdir / "snapshots", device)
    ckpt = {
        "model": model.state_dict(),
        "step": step,
        "kimg": kimg_shown,
        "config": {
            "in_channels": 3,
            "base_channels": args.base_channels,
            "num_blocks": args.num_blocks,
        },
    }
    torch.save(ckpt, outdir / "final.pt")
    elapsed = time.time() - t0
    print(f"Training complete. {kimg_shown:.1f} kimg in {elapsed:.0f}s")
    print(f"Final model saved to {outdir / 'final.pt'}")


if __name__ == "__main__":
    main()
