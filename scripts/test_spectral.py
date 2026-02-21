#!/usr/bin/env python3
"""Validate spectral demo semantic conditioning.

Generates grids from the SAME noise seed with controlled audio vectors to
verify that the model has learned meaningful visual-audio associations.

Tests:
  - Neutral baseline (all 0.5)
  - Each feature isolated at 1.0 vs 0.0
  - RMS brightness sweep (0 -> 1)
  - Warmth sweep (beat_phase 0 -> 1)

Outputs a labeled comparison grid PNG.

Usage:
    uv run scripts/test_spectral.py --checkpoint training-runs/spectral/final.pt
    uv run scripts/test_spectral.py --checkpoint training-runs/spectral/final.pt --output output/spectral_test.png
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easey_glyph.glyph.converter import render_glyph_grid
from easey_glyph.glyph.embedding import compute_embeddings
from easey_glyph.glyph.rasterizer import build_glyph_atlas
from easey_glyph.model.unet import FlowUNet
from easey_glyph.render.pipeline import ode_sample

FEATURE_NAMES = [
    "bass", "mid", "treble", "rms", "beat_phase", "onset_strength",
    "spectral_centroid", "spectral_flux", "spectral_flatness",
    "spectral_rolloff", "spectral_bandwidth", "zero_crossing_rate",
]

SEMANTIC_LABELS = [
    "Heavy shapes", "Medium detail", "Fine detail", "Brightness",
    "Warmth", "Sharpness", "Saturation", "Patchiness",
    "Noise/Order", "Light dist.", "Dyn. range", "Crossings",
]


def make_audio_vector(feature_idx: int | None = None, value: float = 1.0,
                      base: float = 0.5, device: str = "cpu") -> torch.Tensor:
    """Create a [1, 12] audio vector with one feature set, rest at base."""
    audio = torch.full((1, 12), base, device=device)
    if feature_idx is not None:
        audio[0, feature_idx] = value
    return audio


def main():
    parser = argparse.ArgumentParser(description="Test spectral semantic conditioning")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/spectral_test.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

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

    if "model_ema" in ckpt:
        model.load_state_dict(ckpt["model_ema"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    in_ch = model_cfg.get("in_channels", 16)
    img_size = model_cfg.get("image_size", 32)

    glyph_masks = build_glyph_atlas()
    glyph_embeddings = compute_embeddings()

    # Fixed noise
    torch.manual_seed(args.seed)
    z = torch.randn(1, in_ch, img_size, img_size, device=device)

    print(f"Generating with cfg_scale={args.cfg_scale}, {args.steps} steps, seed={args.seed}")
    print(f"Device: {device}\n")

    # --- Section 1: Feature isolation (each feature at 0 vs 1) ---
    print("── Feature Isolation (0 vs 1) ──")
    iso_renders = []  # (label, render_0, render_1)
    for i, (name, semantic) in enumerate(zip(FEATURE_NAMES, SEMANTIC_LABELS)):
        audio_lo = make_audio_vector(i, 0.0, base=0.5, device=device)
        audio_hi = make_audio_vector(i, 1.0, base=0.5, device=device)

        grid_lo = ode_sample(model, z.clone(), steps=args.steps, audio=audio_lo, cfg_scale=args.cfg_scale)
        grid_hi = ode_sample(model, z.clone(), steps=args.steps, audio=audio_hi, cfg_scale=args.cfg_scale)

        # MSE between lo and hi
        mse = (grid_lo - grid_hi).float().pow(2).mean().item()
        print(f"  {name:<22s} ({semantic:<14s}): MSE = {mse:.6f}")

        r_lo = render_glyph_grid(grid_lo[0].cpu(), glyph_masks, glyph_embeddings)
        r_hi = render_glyph_grid(grid_hi[0].cpu(), glyph_masks, glyph_embeddings)
        iso_renders.append((name, semantic, r_lo, r_hi))

    # --- Section 2: Neutral baseline ---
    print("\n── Baseline ──")
    audio_neutral = make_audio_vector(None, base=0.5, device=device)
    grid_neutral = ode_sample(model, z.clone(), steps=args.steps, audio=audio_neutral, cfg_scale=args.cfg_scale)
    r_neutral = render_glyph_grid(grid_neutral[0].cpu(), glyph_masks, glyph_embeddings)

    # --- Section 3: Sweeps ---
    print("\n── Sweeps ──")
    sweep_steps = 5
    sweep_values = [i / (sweep_steps - 1) for i in range(sweep_steps)]

    # RMS sweep (brightness)
    rms_renders = []
    for val in sweep_values:
        audio = make_audio_vector(3, val, base=0.5, device=device)  # rms = idx 3
        grid = ode_sample(model, z.clone(), steps=args.steps, audio=audio, cfg_scale=args.cfg_scale)
        rms_renders.append(render_glyph_grid(grid[0].cpu(), glyph_masks, glyph_embeddings))
    print(f"  RMS sweep: {len(rms_renders)} steps")

    # Warmth sweep (beat_phase)
    warmth_renders = []
    for val in sweep_values:
        audio = make_audio_vector(4, val, base=0.5, device=device)  # beat_phase = idx 4
        grid = ode_sample(model, z.clone(), steps=args.steps, audio=audio, cfg_scale=args.cfg_scale)
        warmth_renders.append(render_glyph_grid(grid[0].cpu(), glyph_masks, glyph_embeddings))
    print(f"  Warmth sweep: {len(warmth_renders)} steps")

    # --- Compose output image ---
    cell_h, cell_w = iso_renders[0][2].shape[:2]  # 256x256
    label_h = 32  # height for text labels
    section_gap = 16

    # Layout:
    # Row 0: "Neutral" label + neutral render
    # Row 1-12: feature name, lo render, hi render (one row per feature)
    # Row 13: "RMS Sweep 0→1" + 5 renders
    # Row 14: "Warmth Sweep 0→1" + 5 renders

    n_iso_rows = len(iso_renders)
    n_cols_iso = 3  # label column + lo + hi
    n_cols_sweep = max(sweep_steps + 1, n_cols_iso)  # +1 for label

    total_w = n_cols_sweep * cell_w
    total_h = (
        label_h + cell_h +  # neutral row
        section_gap +
        label_h +  # "Feature Lo Hi" header
        n_iso_rows * (cell_h + label_h) +  # isolation rows
        section_gap +
        2 * (label_h + cell_h)  # two sweep rows
    )

    canvas = np.zeros((total_h, total_w, 4), dtype=np.uint8)
    # Dark background
    canvas[:, :, 0] = 24
    canvas[:, :, 1] = 24
    canvas[:, :, 2] = 24
    canvas[:, :, 3] = 255

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    y = 0

    # --- Neutral baseline ---
    draw.text((8, y + 8), "Neutral (all 0.5)", fill=(200, 200, 200), font=font)
    y += label_h
    img.paste(Image.fromarray(r_neutral), (0, y))
    y += cell_h + section_gap

    # --- Feature isolation header ---
    draw.text((8, y + 8), "Feature", fill=(160, 160, 160), font=font_small)
    draw.text((cell_w + 8, y + 8), "Value = 0", fill=(100, 180, 255), font=font_small)
    draw.text((2 * cell_w + 8, y + 8), "Value = 1", fill=(255, 180, 100), font=font_small)
    y += label_h

    # --- Feature isolation rows ---
    for name, semantic, r_lo, r_hi in iso_renders:
        draw.text((8, y + 8), f"{name}", fill=(200, 200, 200), font=font_small)
        draw.text((8, y + 20), f"({semantic})", fill=(120, 120, 120), font=font_small)
        y += label_h
        # Place lo and hi renders side by side starting at column 1
        img.paste(Image.fromarray(r_lo), (cell_w, y))
        img.paste(Image.fromarray(r_hi), (2 * cell_w, y))
        y += cell_h

    y += section_gap

    # --- RMS sweep ---
    draw.text((8, y + 8), "RMS Sweep (brightness) 0 -> 1", fill=(200, 200, 200), font=font)
    y += label_h
    for i, r in enumerate(rms_renders):
        img.paste(Image.fromarray(r), (i * cell_w, y))
    y += cell_h

    # --- Warmth sweep ---
    draw.text((8, y + 8), "Warmth Sweep (beat_phase) 0 -> 1", fill=(200, 200, 200), font=font)
    y += label_h
    for i, r in enumerate(warmth_renders):
        img.paste(Image.fromarray(r), (i * cell_w, y))

    # Save
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    img.save(outpath)
    print(f"\nSaved spectral test to {outpath}")
    print(f"  Image size: {img.size[0]}x{img.size[1]}")


if __name__ == "__main__":
    main()
