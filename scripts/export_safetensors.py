#!/usr/bin/env python3
"""Export trained EASEy-GLYPH checkpoints to safetensors format for HuggingFace sharing.

Extracts inference-only state_dicts (no optimizer state) and writes companion
JSON sidecar files with model config and training metadata.

Usage:
    uv run scripts/export_safetensors.py [--outdir exports/hf-models]
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# (checkpoint_path, output_name, state_dict_key, architecture)
CHECKPOINTS = [
    # Abstract
    ("training-runs/abstract-v2/final.pt", "easey-glyph-flow-abstract-v2", "model_ema", "FlowUNet"),
    ("training-runs/abstract-v2-realtime/final.pt", "easey-glyph-flow-abstract-v2-realtime", "model_ema", "FlowUNet"),
    ("training-runs/superres-abstract-v2/final.pt", "easey-glyph-superres-abstract-v2", "model", "GlyphSuperRes"),
    # Nature
    ("training-runs/nature/final.pt", "easey-glyph-flow-nature", "model_ema", "FlowUNet"),
    ("training-runs/nature-realtime/final.pt", "easey-glyph-flow-nature-realtime", "model_ema", "FlowUNet"),
    ("training-runs/superres-nature/final.pt", "easey-glyph-superres-nature", "model", "GlyphSuperRes"),
    # Ukiyo-e
    ("training-runs/ukiyoe/final.pt", "easey-glyph-flow-ukiyoe", "model_ema", "FlowUNet"),
    ("training-runs/ukiyoe-realtime/final.pt", "easey-glyph-flow-ukiyoe-realtime", "model_ema", "FlowUNet"),
    ("training-runs/superres-ukiyoe/final.pt", "easey-glyph-superres-ukiyoe", "model", "GlyphSuperRes"),
    # Albums
    ("training-runs/albums/final.pt", "easey-glyph-flow-albums", "model_ema", "FlowUNet"),
    ("training-runs/albums-realtime/final.pt", "easey-glyph-flow-albums-realtime", "model_ema", "FlowUNet"),
    ("training-runs/superres-albums/final.pt", "easey-glyph-superres-albums", "model", "GlyphSuperRes"),
    # Pixel
    ("training-runs/pixel/final.pt", "easey-glyph-flow-pixel", "model_ema", "FlowUNet"),
    ("training-runs/pixel-realtime/final.pt", "easey-glyph-flow-pixel-realtime", "model_ema", "FlowUNet"),
    ("training-runs/superres-pixel/final.pt", "easey-glyph-superres-pixel", "model", "GlyphSuperRes"),
    # Botanical
    ("training-runs/botanical/final.pt", "easey-glyph-flow-botanical", "model_ema", "FlowUNet"),
    ("training-runs/botanical-realtime/final.pt", "easey-glyph-flow-botanical-realtime", "model_ema", "FlowUNet"),
    ("training-runs/superres-botanical/final.pt", "easey-glyph-superres-botanical", "model", "GlyphSuperRes"),
    # Darkpsy
    ("training-runs/darkpsy/final.pt", "easey-glyph-flow-darkpsy", "model_ema", "FlowUNet"),
    ("training-runs/darkpsy-realtime/final.pt", "easey-glyph-flow-darkpsy-realtime", "model_ema", "FlowUNet"),
    ("training-runs/superres-darkpsy/final.pt", "easey-glyph-superres-darkpsy", "model", "GlyphSuperRes"),
]


def export_checkpoint(ckpt_path: Path, output_name: str, key: str, arch: str, outdir: Path):
    """Export a single checkpoint to safetensors + JSON sidecar."""
    if not ckpt_path.exists():
        print(f"  SKIP {ckpt_path} (not found)")
        return False

    print(f"  Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract inference state_dict
    if key not in ckpt:
        # Fall back: try model_ema -> model
        fallback = "model_ema" if key == "model" else "model"
        if fallback in ckpt:
            print(f"    Key '{key}' not found, using '{fallback}'")
            key = fallback
        else:
            print(f"    ERROR: neither '{key}' nor '{fallback}' found in checkpoint")
            return False

    state_dict = ckpt[key]

    # safetensors requires contiguous tensors
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    # Write safetensors
    st_path = outdir / f"{output_name}.safetensors"
    save_file(state_dict, str(st_path))
    size_mb = st_path.stat().st_size / 1e6
    print(f"    -> {st_path} ({size_mb:.1f} MB)")

    # Build sidecar metadata
    cfg = ckpt.get("config", {})
    if arch == "FlowUNet":
        model_cfg = cfg.get("model", {})
    else:
        # SuperRes stores config flat (not nested under "model")
        model_cfg = {k: v for k, v in cfg.items() if k != "training" and k != "data"}

    sidecar = {
        "architecture": arch,
        "model": model_cfg,
        "training": {
            "kimg": ckpt.get("kimg"),
            "step": ckpt.get("step"),
        },
        "state_dict_key": key,
        "num_params": sum(v.numel() for v in state_dict.values()),
    }

    json_path = outdir / f"{output_name}.json"
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"    -> {json_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Export checkpoints to safetensors")
    parser.add_argument("--outdir", type=str, default="models",
                        help="Output directory (default: models)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {outdir}/\n")
    exported = 0
    for ckpt_rel, name, key, arch in CHECKPOINTS:
        ckpt_path = Path(ckpt_rel)
        if export_checkpoint(ckpt_path, name, key, arch, outdir):
            exported += 1
        print()

    print(f"Done: {exported}/{len(CHECKPOINTS)} checkpoints exported.")


if __name__ == "__main__":
    main()
