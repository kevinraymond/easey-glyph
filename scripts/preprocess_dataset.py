#!/usr/bin/env python3
"""Convert image zip -> glyph .pt dataset.

Usage:
    uv run scripts/preprocess_dataset.py --input datasets/nature-256-10k.zip
    # -> datasets/nature/glyph-nature-10k.pt (auto-derived)

    uv run scripts/preprocess_dataset.py --input datasets/abstract-v2-256-10k.zip --output custom/path.pt
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

from easey_glyph.glyph.converter import image_to_glyph_grid
from easey_glyph.glyph.embedding import compute_embeddings
from easey_glyph.glyph.rasterizer import build_glyph_atlas


def derive_output_path(input_path: str) -> str:
    """Derive output path from input ZIP filename.

    Convention: {type}-{resolution}-{count}.zip -> datasets/{type}/glyph-{type}-{count}.pt
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
        # Fallback: can't parse, put in datasets/ with glyph- prefix
        return str(Path(input_path).parent / f"glyph-{stem}.pt")
    dataset_type = "-".join(parts[:res_idx])
    count = "-".join(parts[res_idx + 1:]) if res_idx + 1 < len(parts) else "data"
    return str(Path(input_path).parent / dataset_type / f"glyph-{dataset_type}-{count}.pt")


def main():
    parser = argparse.ArgumentParser(description="Convert image zip to glyph dataset")
    parser.add_argument("--input", type=str, default="datasets/abstract-v2-256-10k.zip")
    parser.add_argument("--output", type=str, default=None, help="Output .pt path (auto-derived from input if omitted)")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images")
    args = parser.parse_args()

    if args.output is None:
        args.output = derive_output_path(args.input)
        print(f"Output: {args.output} (auto-derived from input)")

    print("Building glyph atlas (289 masks)...")
    glyph_masks = build_glyph_atlas()  # [289, 8, 8]

    print("Computing PCA embeddings (8D)...")
    glyph_embeddings = compute_embeddings()  # [289, 8]

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
    print(f"Converting {n} images to glyph grids...")

    all_grids = []
    all_ids = []
    t0 = time.time()

    for i, name in enumerate(image_names):
        data = zf.read(name)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.LANCZOS)
        img_np = np.array(img)

        grid, glyph_ids = image_to_glyph_grid(img_np, glyph_masks, glyph_embeddings)
        all_grids.append(grid)
        all_ids.append(glyph_ids)

        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  {i + 1:5d}/{n} ({rate:.1f} img/s, ETA {eta:.0f}s)")

    zf.close()

    print("Stacking tensors...")
    data = torch.stack(all_grids)       # [N, 16, 32, 32]
    glyph_ids = torch.stack(all_ids)    # [N, 32, 32]

    print(f"Saving to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "glyph_ids": glyph_ids}, args.output)

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    elapsed = time.time() - t0
    print(f"Done! {n} images -> {args.output} ({size_mb:.1f} MB) in {elapsed:.1f}s")
    print(f"\nTo train:\n  uv run scripts/train_glyph.py --config configs/glyph_base.yaml --data {args.output}")


if __name__ == "__main__":
    main()
