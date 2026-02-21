#!/usr/bin/env python3
"""Compute visual pseudo-audio labels for spectral demo training.

Reads images from a ZIP, computes 12 visual features per image (analogous to
the 12 audio features), normalizes to [0, 1], and injects them into an
existing glyph .pt file or saves standalone.

Feature mapping:
  0  bass              Low spatial frequency energy (2D FFT center 1/8)
  1  mid               Mid spatial frequency energy (2D FFT 1/8-1/2)
  2  treble            High spatial frequency energy (2D FFT beyond 1/2)
  3  rms               Overall brightness (mean luminance)
  4  beat_phase         Color warmth (warm vs cool ratio)
  5  onset_strength     Sharpness/contrast (Laplacian variance)
  6  spectral_centroid  Color saturation (mean chroma)
  7  spectral_flux      Texture complexity (block-wise luminance variance)
  8  spectral_flatness  Noise vs structure (luminance histogram entropy / 8.0)
  9  spectral_rolloff   Brightness distribution (luminance 85th percentile)
  10 spectral_bandwidth Dynamic range (luminance std dev)
  11 zero_crossing_rate Zero-crossing rate (sign changes in centered luminance)

Usage:
    uv run scripts/compute_spectral_labels.py \\
        --input datasets/spectral-256-10k.zip \\
        --inject-into datasets/spectral/glyph-spectral-10k.pt

    # Or standalone output:
    uv run scripts/compute_spectral_labels.py \\
        --input datasets/spectral-256-10k.zip \\
        --output datasets/spectral/labels.pt
"""

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch


FEATURE_NAMES = [
    "bass", "mid", "treble", "rms", "beat_phase", "onset_strength",
    "spectral_centroid", "spectral_flux", "spectral_flatness",
    "spectral_rolloff", "spectral_bandwidth", "zero_crossing_rate",
]


def compute_image_features(img_rgb: np.ndarray) -> np.ndarray:
    """Compute 12 visual features for a single RGB image.

    Args:
        img_rgb: [H, W, 3] uint8 RGB image

    Returns:
        [12] float64 raw feature values (pre-normalization)
    """
    img_f = img_rgb.astype(np.float32) / 255.0
    # Luminance (BT.601)
    lum = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    h, w = lum.shape

    # --- Spatial frequency features (bass, mid, treble) ---
    fft = np.fft.fft2(lum)
    fft_mag = np.abs(np.fft.fftshift(fft))
    # Zero out DC component
    cy, cx = h // 2, w // 2
    fft_mag[cy, cx] = 0.0

    # Create radial distance map
    y_coords = np.arange(h) - cy
    x_coords = np.arange(w) - cx
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    radius = np.sqrt(yy ** 2 + xx ** 2)
    max_radius = min(cy, cx)

    # Frequency bands
    low_mask = radius < (max_radius / 8)
    mid_mask = (radius >= max_radius / 8) & (radius < max_radius / 2)
    high_mask = radius >= (max_radius / 2)

    total_energy = fft_mag.sum() + 1e-10
    bass = fft_mag[low_mask].sum() / total_energy      # 0: low freq
    mid = fft_mag[mid_mask].sum() / total_energy        # 1: mid freq
    treble = fft_mag[high_mask].sum() / total_energy    # 2: high freq

    # --- Brightness (rms) ---
    rms = lum.mean()  # 3: overall brightness

    # --- Color warmth (beat_phase) ---
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    warm = (r + 0.5 * g).mean()
    cool = (b + 0.5 * g).mean()
    warmth = warm / (warm + cool + 1e-10)  # 4: warm vs cool ratio

    # --- Sharpness/contrast (onset_strength) ---
    # Laplacian variance
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Simple convolution via scipy-free approach
    from numpy.lib.stride_tricks import as_strided
    # Pad luminance
    lum_pad = np.pad(lum, 1, mode="reflect")
    # Extract 3x3 patches
    shape = (h, w, 3, 3)
    strides = lum_pad.strides * 2
    patches = as_strided(lum_pad, shape=shape, strides=strides)
    laplacian = (patches * lap_kernel).sum(axis=(2, 3))
    onset_strength = laplacian.var()  # 5: sharpness

    # --- Color saturation (spectral_centroid) ---
    # Mean chroma (max - min across RGB channels per pixel)
    # Independent of luminance — purely about color intensity
    chroma = img_f.max(axis=2) - img_f.min(axis=2)
    centroid = chroma.mean()  # 6: mean color saturation

    # --- Texture complexity (spectral_flux) ---
    # Variance of block-wise mean luminance — measures patchiness/texture scale
    # High = varied patches (textured), Low = uniform (flat gradients)
    block_size = 16  # 256/16 = 16x16 grid of blocks
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    block_means = np.zeros(n_blocks_h * n_blocks_w)
    for bi in range(n_blocks_h):
        for bj in range(n_blocks_w):
            patch = lum[bi * block_size:(bi + 1) * block_size,
                        bj * block_size:(bj + 1) * block_size]
            block_means[bi * n_blocks_w + bj] = patch.mean()
    flux = block_means.std()  # 7: block luminance variation

    # --- Noise vs structure (spectral_flatness) ---
    # Luminance histogram entropy
    hist, bin_edges = np.histogram(lum, bins=256, range=(0.0, 1.0))
    hist_sum = hist.sum() + 1e-10
    hist_norm = hist / hist_sum
    hist_norm = hist_norm[hist_norm > 0]
    entropy = -(hist_norm * np.log2(hist_norm)).sum()
    flatness = entropy / 8.0  # 8: entropy normalized (max ~8 for 256 bins)

    # --- Brightness distribution (spectral_rolloff) ---
    rolloff = np.percentile(lum, 85)  # 9: 85th percentile luminance

    # --- Dynamic range (spectral_bandwidth) ---
    bandwidth = lum.std()  # 10: luminance std dev

    # --- Edge density (zero_crossing_rate) ---
    # Actual zero-crossing rate: sign changes in luminance centered at mean, along scanlines
    # Directly analogous to audio ZCR
    lum_centered = lum - lum.mean()
    sign_changes_h = np.diff(np.sign(lum_centered), axis=1) != 0  # horizontal
    sign_changes_v = np.diff(np.sign(lum_centered), axis=0) != 0  # vertical
    total_transitions = sign_changes_h.sum() + sign_changes_v.sum()
    total_possible = sign_changes_h.size + sign_changes_v.size
    zcr = total_transitions / total_possible  # 11: zero-crossing rate

    return np.array([
        bass, mid, treble, rms, warmth, onset_strength,
        centroid, flux, flatness, rolloff, bandwidth, zcr,
    ], dtype=np.float64)


def normalize_features(features: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    """Per-feature percentile clipping and scaling to [0, 1].

    Args:
        features: [N, 12] raw feature values
        lo_pct: lower percentile for clipping
        hi_pct: upper percentile for clipping

    Returns:
        [N, 12] normalized features in [0, 1]
    """
    result = features.copy()
    for i in range(features.shape[1]):
        col = features[:, i]
        lo = np.percentile(col, lo_pct)
        hi = np.percentile(col, hi_pct)
        if hi - lo < 1e-10:
            result[:, i] = 0.5
        else:
            result[:, i] = np.clip((col - lo) / (hi - lo), 0.0, 1.0)
    return result


def print_stats(features: np.ndarray, normalized: np.ndarray):
    """Print per-feature statistics and correlation matrix."""
    print("\n── Raw Feature Statistics ──")
    print(f"{'Feature':<22s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 64)
    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]
        print(f"{name:<22s} {col.mean():10.4f} {col.std():10.4f} {col.min():10.4f} {col.max():10.4f}")

    print("\n── Normalized Feature Statistics ──")
    print(f"{'Feature':<22s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 64)
    for i, name in enumerate(FEATURE_NAMES):
        col = normalized[:, i]
        print(f"{name:<22s} {col.mean():10.4f} {col.std():10.4f} {col.min():10.4f} {col.max():10.4f}")

    # Correlation matrix
    print("\n── Correlation Matrix (top-10 pairs) ──")
    corr = np.corrcoef(normalized.T)
    pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            pairs.append((abs(corr[i, j]), corr[i, j], FEATURE_NAMES[i], FEATURE_NAMES[j]))
    pairs.sort(reverse=True)
    for abs_r, r, a, b in pairs[:10]:
        flag = " *** HIGH" if abs_r > 0.9 else ""
        print(f"  {a:<20s} x {b:<20s}: {r:+.3f}{flag}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute visual pseudo-audio labels for spectral demo training"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input image ZIP file")
    parser.add_argument("--inject-into", type=str, default=None,
                        help="Inject labels into existing glyph .pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Save standalone labels .pt file")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process")
    args = parser.parse_args()

    if args.inject_into is None and args.output is None:
        parser.error("Specify --inject-into or --output")

    from PIL import Image

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
    print(f"Computing 12 visual features for {n} images...")

    all_features = np.zeros((n, 12), dtype=np.float64)
    t0 = time.time()

    for i, name in enumerate(image_names):
        data = zf.read(name)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.LANCZOS)
        img_np = np.array(img)
        all_features[i] = compute_image_features(img_np)

        if (i + 1) % 500 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  {i + 1:5d}/{n} ({rate:.1f} img/s, ETA {eta:.0f}s)")

    zf.close()

    # Normalize
    print("\nNormalizing features (2nd-98th percentile)...")
    normalized = normalize_features(all_features)
    print_stats(all_features, normalized)

    labels = torch.from_numpy(normalized).float()  # [N, 12]

    if args.inject_into:
        print(f"\nInjecting labels into {args.inject_into}...")
        pt_data = torch.load(args.inject_into, map_location="cpu", weights_only=True)
        n_data = pt_data["data"].shape[0]
        if n_data != n:
            print(f"WARNING: .pt has {n_data} images but ZIP has {n} images")
            if n > n_data:
                labels = labels[:n_data]
                print(f"  Trimmed labels to {n_data}")
            else:
                print(f"  ERROR: Not enough images in ZIP to match .pt file")
                sys.exit(1)
        pt_data["labels"] = labels
        torch.save(pt_data, args.inject_into)
        size_mb = Path(args.inject_into).stat().st_size / 1024 / 1024
        print(f"Saved {args.inject_into} ({size_mb:.1f} MB) with labels key")

    if args.output:
        print(f"\nSaving labels to {args.output}...")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"labels": labels, "feature_names": FEATURE_NAMES}, args.output)
        print(f"Saved {args.output}")

    elapsed = time.time() - t0
    print(f"\nDone! {n} images processed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
