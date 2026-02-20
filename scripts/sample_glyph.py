#!/usr/bin/env python3
"""Generate glyph art samples and render to terminal/PNG.

Usage:
    # From trained model
    uv run scripts/sample_glyph.py --checkpoint training-runs/glyph-v1/final.pt

    # Visualize dataset (no model needed)
    uv run scripts/sample_glyph.py --visualize-dataset --data datasets/abstract-v2/glyph-abstract-v2-10k.pt

    # More samples, more steps
    uv run scripts/sample_glyph.py --checkpoint final.pt --n 32 --steps 50 --output-dir output/
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easey_glyph.glyph.converter import render_glyph_grid, render_glyph_grid_ansi, render_glyph_grid_braille
from easey_glyph.glyph.embedding import compute_embeddings
from easey_glyph.glyph.rasterizer import build_glyph_atlas
from easey_glyph.model.unet import FlowUNet


@torch.no_grad()
def euler_sample(
    model: torch.nn.Module,
    z: Tensor,
    audio: Tensor | None = None,
    steps: int = 50,
) -> Tensor:
    dt = 1.0 / steps
    x = z.clone()
    for i in range(steps):
        t_val = i / steps
        t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
        v = model(t, x, audio)
        x = x + v * dt
    return x


def save_grid_png(
    samples: list[Tensor],
    glyph_masks: Tensor,
    glyph_embeddings: Tensor,
    path: Path,
    cols: int = 4,
):
    """Render multiple glyph grids into a single PNG grid."""
    n = len(samples)
    rows = math.ceil(n / cols)

    # Get pixel dimensions from first sample
    s = samples[0]
    h = s.shape[1] * 8  # 32 * 8 = 256
    w = s.shape[2] * 8

    grid_img = np.zeros((rows * h, cols * w, 4), dtype=np.uint8)
    for i, sample in enumerate(samples):
        r, c = divmod(i, cols)
        rendered = render_glyph_grid(sample, glyph_masks, glyph_embeddings)
        grid_img[r * h:(r + 1) * h, c * w:(c + 1) * w] = rendered

    Image.fromarray(grid_img).save(path)
    print(f"Saved grid to {path}")


def visualize_dataset(args):
    """Render a few samples from the preprocessed dataset."""
    glyph_masks = build_glyph_atlas()
    glyph_embeddings = compute_embeddings()

    print(f"Loading dataset {args.data}...")
    raw = torch.load(args.data, map_location="cpu", weights_only=True)
    data = raw["data"]
    print(f"Dataset: {data.shape[0]} samples, shape {data.shape[1:]}")

    n = min(args.n, data.shape[0])
    indices = torch.randperm(data.shape[0])[:n]
    samples = [data[i] for i in indices]

    # Render to terminal (first 4)
    for i, sample in enumerate(samples[:4]):
        print(f"\n--- Sample {indices[i].item()} ---")
        ansi = render_glyph_grid_ansi(sample, glyph_masks, glyph_embeddings)
        print(ansi)

    # Save grid PNG
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_grid_png(samples, glyph_masks, glyph_embeddings, outdir / "dataset_samples.png")

    # Save Braille character renders
    for i, sample in enumerate(samples[:4]):
        braille_img = render_glyph_grid_braille(sample, glyph_masks, glyph_embeddings)
        Image.fromarray(braille_img).save(outdir / f"braille_{indices[i].item():04d}.png")
    print(f"Saved {min(4, len(samples))} Braille renders to {outdir}")


def generate_samples(args):
    """Generate samples from a trained model."""
    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Get config from checkpoint
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = FlowUNet(
        image_size=model_cfg.get("image_size", 32),
        in_channels=model_cfg.get("in_channels", 16),
        out_channels=model_cfg.get("out_channels", 16),
        base_channels=model_cfg.get("base_channels", 64),
        channel_mult=model_cfg.get("channel_mult", [1, 2, 4, 4]),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=model_cfg.get("attention_resolutions", [4]),
        use_depthwise_separable=model_cfg.get("use_depthwise_separable", False),
        audio_dim=model_cfg.get("audio_dim", 12),
        time_emb_dim=model_cfg.get("time_emb_dim", 128),
    ).to(device)

    # Load weights (prefer EMA)
    if "model_ema" in ckpt:
        model.load_state_dict(ckpt["model_ema"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params ({params/1e6:.2f}M)")

    glyph_masks = build_glyph_atlas()
    glyph_embeddings = compute_embeddings()

    in_ch = model_cfg.get("in_channels", 16)
    img_size = model_cfg.get("image_size", 32)

    print(f"Generating {args.n} samples with {args.steps} Euler steps...")
    z = torch.randn(args.n, in_ch, img_size, img_size, device=device)
    samples = euler_sample(model, z, audio=None, steps=args.steps)
    samples = [s.cpu() for s in samples]

    # Render to terminal (first few)
    n_terminal = min(4, len(samples))
    for i in range(n_terminal):
        print(f"\n--- Sample {i} ---")
        ansi = render_glyph_grid_ansi(samples[i], glyph_masks, glyph_embeddings)
        print(ansi)

    # Save grid PNG
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_grid_png(samples, glyph_masks, glyph_embeddings, outdir / "generated_samples.png")

    # Also save individual PNGs + Braille renders
    for i, sample in enumerate(samples):
        rendered = render_glyph_grid(sample, glyph_masks, glyph_embeddings)
        Image.fromarray(rendered).save(outdir / f"sample_{i:03d}.png")
        braille_img = render_glyph_grid_braille(sample, glyph_masks, glyph_embeddings)
        Image.fromarray(braille_img).save(outdir / f"braille_{i:03d}.png")
    print(f"Saved {len(samples)} individual PNGs + Braille renders to {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Sample EASEy-GLYPH")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--visualize-dataset", action="store_true", help="Visualize preprocessed dataset")
    parser.add_argument("--data", type=str, default="datasets/abstract-v2/glyph-abstract-v2-10k.pt", help="Dataset path")
    parser.add_argument("--n", type=int, default=16, help="Number of samples")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    if args.visualize_dataset:
        visualize_dataset(args)
    elif args.checkpoint:
        generate_samples(args)
    else:
        parser.error("Provide either --checkpoint or --visualize-dataset")


if __name__ == "__main__":
    main()
