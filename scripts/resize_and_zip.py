#!/usr/bin/env python3
"""Merge image directories and/or zip files, resize to target size, and write to a zip.

Usage:
    uv run scripts/resize_and_zip.py \
        --input-dirs dir1 dir2 \
        --output datasets/nature-256-10k.zip \
        --size 256

    uv run scripts/resize_and_zip.py \
        --input-zips datasets/old.zip \
        --input-dirs dir1 dir2 \
        --output datasets/merged-256.zip \
        --size 256
"""

import argparse
import random
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


def collect_from_dirs(input_dirs: list[Path]) -> list[Path]:
    """Collect all PNG/JPG image paths from directories."""
    images = []
    for d in input_dirs:
        if not d.exists():
            print(f"  WARNING: {d} does not exist, skipping")
            continue
        found = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        print(f"  {d}: {len(found)} images")
        images.extend(found)
    return images


def process_image(img: Image.Image, size: int, invert: bool = False) -> bytes:
    """Resize if needed, optionally invert, and return PNG bytes."""
    img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    if invert:
        img = ImageOps.invert(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Resize images and create zip")
    parser.add_argument("--input-dirs", type=Path, nargs="*", default=[],
                        help="Input directories containing images")
    parser.add_argument("--input-zips", type=Path, nargs="*", default=[],
                        help="Input zip files containing images")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output zip file path")
    parser.add_argument("--size", type=int, default=256,
                        help="Target size (square, default 256)")
    parser.add_argument("--invert-fraction", type=float, default=0.0,
                        help="Fraction of images to randomly invert (0.0-1.0)")
    args = parser.parse_args()

    if not args.input_dirs and not args.input_zips:
        parser.error("At least one of --input-dirs or --input-zips is required")

    # Count totals for progress bar
    print("Collecting images...")
    dir_images = collect_from_dirs(args.input_dirs) if args.input_dirs else []

    zip_entries = []  # list of (zip_path, entry_name)
    for zp in args.input_zips:
        if not zp.exists():
            print(f"  WARNING: {zp} does not exist, skipping")
            continue
        with zipfile.ZipFile(zp, "r") as zf:
            entries = sorted(n for n in zf.namelist()
                             if n.lower().endswith((".png", ".jpg", ".jpeg")))
            print(f"  {zp}: {len(entries)} images")
            zip_entries.extend((zp, e) for e in entries)

    total = len(zip_entries) + len(dir_images)
    print(f"Total: {total} images")

    if total == 0:
        print("No images found, exiting.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    invert_fraction = args.invert_fraction
    print(f"Resizing to {args.size}x{args.size} and writing to {args.output}...")
    if invert_fraction > 0:
        print(f"Randomly inverting ~{invert_fraction:.0%} of images")
    written = 0
    skipped = 0
    inverted = 0
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_STORED) as out_zf:
        pbar = tqdm(total=total, desc="Processing")

        # First: images from input zips (originals)
        for zp, entry_name in zip_entries:
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    img = Image.open(BytesIO(zf.read(entry_name)))
                do_invert = random.random() < invert_fraction
                written += 1
                if do_invert:
                    inverted += 1
                out_zf.writestr(f"{written:05d}.png", process_image(img, args.size, invert=do_invert))
            except Exception as e:
                skipped += 1
                tqdm.write(f"  SKIP (corrupt): {zp}:{entry_name} — {e}")
            pbar.update(1)

        # Then: images from directories (new generations)
        for img_path in dir_images:
            try:
                img = Image.open(img_path)
                do_invert = random.random() < invert_fraction
                written += 1
                if do_invert:
                    inverted += 1
                out_zf.writestr(f"{written:05d}.png", process_image(img, args.size, invert=do_invert))
            except Exception as e:
                skipped += 1
                tqdm.write(f"  SKIP (corrupt): {img_path} — {e}")
            pbar.update(1)

        pbar.close()

    print(f"Done! {written} images written to {args.output} ({skipped} skipped, {inverted} inverted)")


if __name__ == "__main__":
    main()
