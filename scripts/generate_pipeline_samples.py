#!/usr/bin/env python3
"""Generate visual samples at each stage of the art → glyph pipeline.

For each sampled image, produces:
  original.png      — source art (256x256)
  glyph_pixel.png   — glyph-encoded, flat pixel render (upscaled to 256x256 nearest)
  glyph_render.png  — glyph-encoded, actual glyph characters (256x256)
  superres.png       — super-resolution CNN output (256x256, optional)
  comparison.png     — all stages side-by-side strip

Also produces:
  grid_overview.png  — grid of all samples: original → glyph → superres
"""

import argparse
import io
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from easey_glyph.audio.reactive import RenderParams, render_pixel_frame
from easey_glyph.glyph.converter import image_to_glyph_grid, render_glyph_grid
from easey_glyph.glyph.embedding import compute_embeddings
from easey_glyph.glyph.rasterizer import build_glyph_atlas
from easey_glyph.model.superres import load_superres


def neutral_params() -> RenderParams:
    """Neutral render params (no audio modulation)."""
    return RenderParams(
        dot_radius=2.0,
        visibility_threshold=0.0,
        bg_alpha_scale=1.0,
        fg_brightness=1.0,
    )


def load_images_from_zip(zip_path: str, num_samples: int) -> list[np.ndarray]:
    """Load and randomly sample images from a ZIP file."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        image_names = [
            n for n in zf.namelist()
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            and not n.startswith("__MACOSX")
        ]
        if not image_names:
            raise ValueError(f"No images found in {zip_path}")

        chosen = random.sample(image_names, min(num_samples, len(image_names)))
        images = []
        for name in chosen:
            data = zf.read(name)
            img = Image.open(io.BytesIO(data)).convert("RGB").resize((256, 256), Image.LANCZOS)
            images.append(np.array(img))
    return images


@torch.no_grad()
def apply_superres(frame_arr: np.ndarray, model, device) -> np.ndarray:
    """Run super-resolution CNN on a 32x32 RGBA frame -> 256x256 RGBA."""
    rgb = frame_arr[:, :, :3]
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    sr = model(x)
    sr = (sr[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()

    alpha = frame_arr[:, :, 3]
    alpha_t = torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0).float()
    alpha_up = torch.nn.functional.interpolate(
        alpha_t, size=(256, 256), mode="bilinear", align_corners=False
    )
    out = np.empty((256, 256, 4), dtype=np.uint8)
    out[:, :, :3] = sr
    out[:, :, 3] = alpha_up[0, 0].clamp(0, 255).byte().numpy()
    return out


def composite_on_white(rgba: np.ndarray) -> np.ndarray:
    """Composite RGBA onto white background, return RGB uint8."""
    f = rgba.astype(np.float32) / 255.0
    alpha = f[:, :, 3:4]
    rgb = f[:, :, :3] * alpha + 1.0 * (1 - alpha)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline walkthrough samples")
    parser.add_argument("--input", required=True, help="Path to art ZIP file")
    parser.add_argument("--output", required=True, help="Output directory for samples")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of samples")
    parser.add_argument("--superres", default=None, help="Path to superres checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load glyph resources
    print("Loading glyph atlas and embeddings...")
    glyph_masks = build_glyph_atlas()
    glyph_embeddings = compute_embeddings()

    # Load superres model if provided
    sr_model = None
    if args.superres:
        print(f"Loading superres from {args.superres}...")
        sr_model = load_superres(args.superres, device)

    # Load source images
    print(f"Loading {args.num_samples} images from {args.input}...")
    images = load_images_from_zip(args.input, args.num_samples)
    print(f"  Got {len(images)} images")

    params = neutral_params()
    has_sr = sr_model is not None

    # Per-sample: number of stages for the comparison strip
    stages = 3 if not has_sr else 4  # original, pixel, render [, superres]

    all_originals = []
    all_pixels = []
    all_superres = []

    for i, img in enumerate(images):
        sample_dir = out_dir / f"sample_{i:02d}"
        sample_dir.mkdir(exist_ok=True)
        print(f"Processing sample {i + 1}/{len(images)}...")

        # 1. Original
        original = Image.fromarray(img)
        original.save(sample_dir / "01_original.png")
        all_originals.append(img)

        # 2. Encode to glyph grid
        grid, glyph_ids = image_to_glyph_grid(img, glyph_masks, glyph_embeddings)

        # 3. Pixel render (32x32 -> 256x256 nearest)
        pixel_32 = render_pixel_frame(grid, params)  # [32, 32, 4] uint8
        pixel_pil = Image.fromarray(pixel_32, "RGBA")
        pixel_256 = pixel_pil.resize((256, 256), Image.NEAREST)
        pixel_rgb = composite_on_white(np.array(pixel_256))
        Image.fromarray(pixel_rgb).save(sample_dir / "02_glyph_pixel.png")
        all_pixels.append(pixel_rgb)

        # 4. Glyph character render (256x256 RGBA)
        glyph_rgba = render_glyph_grid(grid, glyph_masks, glyph_embeddings)
        glyph_rgb = composite_on_white(glyph_rgba)
        Image.fromarray(glyph_rgb).save(sample_dir / "03_glyph_render.png")

        # 5. Superres (optional)
        sr_rgb = None
        if has_sr:
            sr_rgba = apply_superres(pixel_32, sr_model, device)
            sr_rgb = composite_on_white(sr_rgba)
            Image.fromarray(sr_rgb).save(sample_dir / "04_superres.png")
            all_superres.append(sr_rgb)

        # 6. Comparison strip
        strip_parts = [img, pixel_rgb, glyph_rgb]
        if sr_rgb is not None:
            strip_parts.append(sr_rgb)
        gap = 4
        strip_w = 256 * len(strip_parts) + gap * (len(strip_parts) - 1)
        strip = np.full((256, strip_w, 3), 255, dtype=np.uint8)
        for j, part in enumerate(strip_parts):
            x_off = j * (256 + gap)
            strip[:, x_off:x_off + 256] = part
        Image.fromarray(strip).save(sample_dir / "05_comparison.png")

    # Grid overview: rows=samples, cols=stages
    print("Generating grid overview...")
    n = len(images)
    cols = 2 if not has_sr else 3  # original, pixel [, superres]
    gap = 4
    cell = 256
    grid_w = cols * cell + (cols - 1) * gap
    grid_h = n * cell + (n - 1) * gap
    overview = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for row in range(n):
        y = row * (cell + gap)
        # Original
        overview[y:y + cell, 0:cell] = all_originals[row]
        # Pixel
        x = cell + gap
        overview[y:y + cell, x:x + cell] = all_pixels[row]
        # Superres
        if has_sr and row < len(all_superres):
            x = 2 * (cell + gap)
            overview[y:y + cell, x:x + cell] = all_superres[row]

    Image.fromarray(overview).save(out_dir / "grid_overview.png")
    print(f"Done! Samples written to {out_dir}/")


if __name__ == "__main__":
    main()
