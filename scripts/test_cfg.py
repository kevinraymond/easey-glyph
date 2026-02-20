#!/usr/bin/env python3
"""Quick diagnostic: verify CFG audio conditioning is working.

Generates grids from the SAME noise with different audio vectors and
reports pixel-level MSE. If CFG works, MSE > 0. Saves a comparison PNG.

Usage:
    uv run scripts/test_cfg.py --checkpoint training-runs/abstract-v2-rand/final.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easey_glyph.glyph.converter import render_glyph_grid
from easey_glyph.glyph.embedding import compute_embeddings
from easey_glyph.glyph.rasterizer import build_glyph_atlas
from easey_glyph.model.unet import FlowUNet
from easey_glyph.render.pipeline import ode_sample


def main():
    parser = argparse.ArgumentParser(description="Test CFG audio conditioning")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/cfg_test.png")
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

    # Fixed noise
    torch.manual_seed(args.seed)
    z = torch.randn(1, in_ch, img_size, img_size, device=device)

    # Generate 3 outputs from same noise:
    # A) unconditional (cfg_scale=0)
    # B) audio = all zeros  (cfg_scale=5)
    # C) audio = all ones   (cfg_scale=5)
    # D) audio = random     (cfg_scale=5)
    audio_zero = torch.zeros(1, 12, device=device)
    audio_ones = torch.ones(1, 12, device=device)
    audio_rand = torch.rand(1, 12, device=device)

    print(f"Generating with cfg_scale={args.cfg_scale}, {args.steps} steps, seed={args.seed}")
    print()

    grid_uncond = ode_sample(model, z.clone(), steps=args.steps, audio=None, cfg_scale=0.0)
    grid_zero = ode_sample(model, z.clone(), steps=args.steps, audio=audio_zero, cfg_scale=args.cfg_scale)
    grid_ones = ode_sample(model, z.clone(), steps=args.steps, audio=audio_ones, cfg_scale=args.cfg_scale)
    grid_rand = ode_sample(model, z.clone(), steps=args.steps, audio=audio_rand, cfg_scale=args.cfg_scale)

    # Compute MSE between all pairs
    pairs = [
        ("uncond vs audio=0", grid_uncond, grid_zero),
        ("uncond vs audio=1", grid_uncond, grid_ones),
        ("audio=0 vs audio=1", grid_zero, grid_ones),
        ("audio=0 vs audio=rand", grid_zero, grid_rand),
        ("audio=1 vs audio=rand", grid_ones, grid_rand),
    ]

    print("Pair                       MSE        Max diff")
    print("-" * 52)
    all_zero = True
    for label, a, b in pairs:
        diff = (a - b).float()
        mse = diff.pow(2).mean().item()
        maxd = diff.abs().max().item()
        status = "OK" if mse > 1e-6 else "SAME!"
        if mse > 1e-6:
            all_zero = False
        print(f"{label:<26} {mse:.6f}   {maxd:.4f}   {status}")

    print()
    if all_zero:
        print("CFG has NO effect — all outputs identical. Something is wrong.")
    else:
        print("CFG is working — different audio vectors produce different outputs.")

    # Render comparison image
    glyph_masks = build_glyph_atlas()
    glyph_embeddings = compute_embeddings()

    labels = ["uncond (cfg=0)", "audio=0", "audio=1", "audio=rand"]
    grids = [grid_uncond[0].cpu(), grid_zero[0].cpu(), grid_ones[0].cpu(), grid_rand[0].cpu()]

    renders = [render_glyph_grid(g, glyph_masks, glyph_embeddings) for g in grids]
    h, w = renders[0].shape[:2]

    # Side by side
    out = np.zeros((h, w * 4, 4), dtype=np.uint8)
    for i, r in enumerate(renders):
        out[:, i * w:(i + 1) * w] = r

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(outpath)
    print(f"\nSaved comparison to {outpath}")
    print(f"  Left to right: {', '.join(labels)}")


if __name__ == "__main__":
    main()
