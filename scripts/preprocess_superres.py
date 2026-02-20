#!/usr/bin/env python3
"""Create super-resolution training pairs from image zip.

For each image: encode to glyph grid, render to 32x32 RGB (low-res input),
keep original 256x256 RGB (high-res target). Saves paired tensors.

Usage:
    uv run scripts/preprocess_superres.py --input datasets/nature-256-10k.zip
    # -> datasets/nature/superres-nature-10k.pt (auto-derived)

    uv run scripts/preprocess_superres.py --input datasets/abstract-v2-256-10k.zip --output custom/path.pt
"""

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easey_glyph.audio.frame import AudioFrame
from easey_glyph.audio.reactive import RenderParams, render_pixel_frame
from easey_glyph.glyph.converter import image_to_glyph_grid
from easey_glyph.glyph.embedding import compute_embeddings
from easey_glyph.glyph.rasterizer import build_glyph_atlas


def derive_output_path(input_path: str) -> str:
    """Derive output path from input ZIP filename.

    Convention: {type}-{resolution}-{count}.zip -> datasets/{type}/superres-{type}-{count}.pt
    Finds the first purely numeric segment (resolution), everything before is type,
    everything after is count.
    """
    stem = Path(input_path).stem  # e.g. "nature-256-10k"
    parts = stem.split("-")
    # Find first purely numeric segment (the resolution)
    res_idx = None
    for i, part in enumerate(parts):
        if part.isdigit():
            res_idx = i
            break
    if res_idx is None or res_idx == 0:
        # Fallback: can't parse, put in datasets/ with superres- prefix
        return str(Path(input_path).parent / f"superres-{stem}.pt")
    dataset_type = "-".join(parts[:res_idx])
    count = "-".join(parts[res_idx + 1:]) if res_idx + 1 < len(parts) else "data"
    return str(Path(input_path).parent / dataset_type / f"superres-{dataset_type}-{count}.pt")


def main():
    parser = argparse.ArgumentParser(description="Create super-resolution training pairs")
    parser.add_argument("--input", type=str, default="datasets/abstract-v2-256-10k.zip")
    parser.add_argument("--output", type=str, default=None, help="Output .pt path (auto-derived from input if omitted)")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images")
    args = parser.parse_args()

    if args.output is None:
        args.output = derive_output_path(args.input)
        print(f"Output: {args.output} (auto-derived from input)")

    print("Building glyph atlas (289 masks)...")
    glyph_masks = build_glyph_atlas()

    print("Computing PCA embeddings (8D)...")
    glyph_embeddings = compute_embeddings()

    # Neutral render params (no brightness modulation)
    neutral_params = RenderParams(
        dot_radius=2.0,
        visibility_threshold=0.0,
        bg_alpha_scale=1.0,
        fg_brightness=1.0,
    )

    print(f"Opening {args.input}...")
    zf = zipfile.ZipFile(args.input, "r")
    image_names = sorted([
        n for n in zf.namelist()
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        and not n.startswith("__MACOSX")
    ])

    if args.max_images:
        image_names = image_names[:args.max_images]

    n = len(image_names)
    print(f"Processing {n} images...")

    low_res_list = []
    high_res_list = []
    t0 = time.time()

    for i, name in enumerate(image_names):
        data = zf.read(name)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.LANCZOS)
        img_np = np.array(img)

        # High-res target: 256x256 RGB uint8
        high_res = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, 256, 256] uint8

        # Low-res input: encode to glyph grid, render to 32x32 pixel RGBA, extract RGB uint8
        grid, _ = image_to_glyph_grid(img_np, glyph_masks, glyph_embeddings)
        frame_arr = render_pixel_frame(grid, neutral_params)  # [32, 32, 4] uint8
        low_res = torch.from_numpy(frame_arr[:, :, :3]).permute(2, 0, 1)  # [3, 32, 32] uint8

        low_res_list.append(low_res)
        high_res_list.append(high_res)

        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  {i + 1:5d}/{n} ({rate:.1f} img/s, ETA {eta:.0f}s)")

    zf.close()

    print("Stacking tensors...")
    low_res_all = torch.stack(low_res_list)    # [N, 3, 32, 32]
    high_res_all = torch.stack(high_res_list)  # [N, 3, 256, 256]

    print(f"Saving to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"low_res": low_res_all, "high_res": high_res_all}, args.output)

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    elapsed = time.time() - t0
    print(f"Done! {n} pairs -> {args.output} ({size_mb:.1f} MB) in {elapsed:.1f}s")
    print(f"\nTo train:\n  uv run scripts/train_superres.py --data {args.output}")


if __name__ == "__main__":
    main()
