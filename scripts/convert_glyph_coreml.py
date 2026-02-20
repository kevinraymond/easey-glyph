#!/usr/bin/env python3
"""Convert EASEy-GLYPH models to CoreML for Mac deployment.

Usage:
    # FlowUNet only
    uv run scripts/convert_glyph_coreml.py \
        training-runs/abstract-v2/final.pt \
        --out models/glyph.mlpackage

    # FlowUNet + SuperRes
    uv run scripts/convert_glyph_coreml.py \
        training-runs/abstract-v2/final.pt \
        --out models/glyph.mlpackage \
        --superres training-runs/superres-abstract-v2/final.pt \
        --superres-out models/superres.mlpackage

    # With 6-bit palettization (smaller file)
    uv run scripts/convert_glyph_coreml.py \
        training-runs/abstract-v2/final.pt \
        --out models/glyph.mlpackage \
        --palettize 6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easey_glyph.model.unet import FlowUNet
from easey_glyph.model.superres import GlyphSuperRes


def _compute_schedule(steps: int, schedule: str) -> list[float]:
    """Compute time schedule for CoreML wrappers (same as pipeline._time_schedule)."""
    import math
    if schedule == "cosine":
        return [0.5 * (1.0 - math.cos(math.pi * i / steps)) for i in range(steps + 1)]
    elif schedule == "poly":
        return [3.0 * (i / steps) ** 2 - 2.0 * (i / steps) ** 3 for i in range(steps + 1)]
    return [i / steps for i in range(steps + 1)]


class MultiStepWrapper(nn.Module):
    """Wraps FlowUNet for fixed multi-step Euler generation.

    Unrolls the ODE solve so CoreML runs the full generation in one call.

    forward(grid, audio) -> output
        grid:  [1, 16, 32, 32] noise
        audio: [1, 12] audio features
    """

    def __init__(self, model: FlowUNet, steps: int = 8, schedule: str = "uniform"):
        super().__init__()
        self.model = model
        self.steps = steps
        times = _compute_schedule(steps, schedule)
        self.register_buffer("times", torch.tensor(times, dtype=torch.float32))

    def forward(self, grid: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        x = grid
        for i in range(self.steps):
            dt = self.times[i + 1] - self.times[i]
            t = self.times[i:i+1]
            v = self.model(t, x, audio)
            x = x + v * dt
        return x


class HeunMultiStepWrapper(nn.Module):
    """Wraps FlowUNet for fixed multi-step Heun (trapezoidal) generation.

    2nd order solver — 2 NFE per step, better quality than Euler at same NFE budget.

    forward(grid, audio) -> output
        grid:  [1, 16, 32, 32] noise
        audio: [1, 12] audio features
    """

    def __init__(self, model: FlowUNet, steps: int = 4, schedule: str = "uniform"):
        super().__init__()
        self.model = model
        self.steps = steps
        times = _compute_schedule(steps, schedule)
        self.register_buffer("times", torch.tensor(times, dtype=torch.float32))

    def forward(self, grid: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        x = grid
        for i in range(self.steps):
            dt = self.times[i + 1] - self.times[i]
            t = self.times[i:i+1]
            t_next = self.times[i+1:i+2]
            v1 = self.model(t, x, audio)
            x_euler = x + v1 * dt
            v2 = self.model(t_next, x_euler, audio)
            x = x + (v1 + v2) * 0.5 * dt
        return x


def convert_unet(
    checkpoint_path: str,
    output_path: str,
    steps: int = 8,
    solver: str = "euler",
    schedule: str = "uniform",
    palettize_bits: int | None = None,
    validate: bool = True,
):
    """Convert FlowUNet checkpoint to CoreML."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    )

    if "model_ema" in ckpt:
        model.load_state_dict(ckpt["model_ema"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"FlowUNet: {params:,} params ({params / 1e6:.2f}M)")

    in_ch = model_cfg.get("in_channels", 16)
    img_size = model_cfg.get("image_size", 32)

    # Monkey-patch attention to use F.scaled_dot_product_attention
    # (traces cleanly for CoreML — avoids int ops from manual reshape/bmm)
    from easey_glyph.model.attention import SelfAttention2d
    _orig_attn_forward = SelfAttention2d.forward

    def _coreml_attn_forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # [B, 3*C, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, -1)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, HW, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)

    SelfAttention2d.forward = _coreml_attn_forward

    if solver == "heun":
        wrapper = HeunMultiStepWrapper(model, steps=steps, schedule=schedule)
        nfe = steps * 2
        solver_label = "Heun"
    else:
        wrapper = MultiStepWrapper(model, steps=steps, schedule=schedule)
        nfe = steps
        solver_label = "Euler"
    wrapper.eval()
    print(f"Using {solver_label}MultiStepWrapper ({steps} steps, {nfe} NFE, {schedule})")

    # Trace
    example_grid = torch.randn(1, in_ch, img_size, img_size)
    example_audio = torch.randn(1, 12)

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_grid, example_audio))
    print("  Trace successful")

    # Restore original attention
    SelfAttention2d.forward = _orig_attn_forward

    # Convert to CoreML
    try:
        import coremltools as ct
    except ImportError:
        print("\ncoremltools not installed. Install with: uv pip install coremltools")
        print("Saving traced TorchScript model instead...")
        ts_path = output_path.replace(".mlpackage", ".pt")
        traced.save(ts_path)
        print(f"Saved TorchScript to {ts_path}")
        return

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType("grid", shape=(1, in_ch, img_size, img_size)),
            ct.TensorType("audio", shape=(1, 12)),
        ],
        outputs=[ct.TensorType("output")],
        minimum_deployment_target=ct.target.macOS13,
    )
    print("  CoreML conversion successful")

    # Palettize
    if palettize_bits is not None:
        print(f"Palettizing to {palettize_bits}-bit...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )
        op_config = OpPalettizerConfig(nbits=palettize_bits)
        opt_config = OptimizationConfig(global_config=op_config)
        mlmodel = palettize_weights(mlmodel, opt_config)
        print("  Palettization complete")

    mlmodel.save(output_path)
    size_mb = sum(
        f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
    ) / 1e6
    print(f"Saved CoreML FlowUNet to {output_path} ({size_mb:.1f} MB)")

    # Validate
    import platform
    if validate and platform.system() != "Darwin":
        print("Skipping validation (CoreML predict requires macOS)")
        validate = False

    if validate:
        print("Validating PyTorch vs CoreML outputs...")
        test_grid = torch.randn(1, in_ch, img_size, img_size)
        test_audio = torch.randn(1, 12)

        with torch.no_grad():
            pt_out = wrapper(test_grid, test_audio).numpy()

        ml_pred = mlmodel.predict({
            "grid": test_grid.numpy(),
            "audio": test_audio.numpy(),
        })
        ml_out = ml_pred["output"]

        max_diff = np.abs(pt_out - ml_out).max()
        mean_diff = np.abs(pt_out - ml_out).mean()
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        if max_diff < 0.01:
            print("  PASS: outputs match within tolerance")
        elif max_diff < 0.05:
            print("  WARN: small difference (likely fp16 precision)")
        else:
            print("  FAIL: large difference detected")


def convert_superres(
    checkpoint_path: str,
    output_path: str,
    palettize_bits: int | None = None,
    validate: bool = True,
):
    """Convert GlyphSuperRes checkpoint to CoreML."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    model = GlyphSuperRes(
        in_channels=cfg.get("in_channels", 3),
        base_channels=cfg.get("base_channels", 64),
        num_blocks=cfg.get("num_blocks", 4),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"\nSuperRes: {params:,} params ({params / 1e6:.2f}M)")

    # Trace
    example_rgb = torch.randn(1, 3, 32, 32)
    print("Tracing SuperRes...")
    with torch.no_grad():
        traced = torch.jit.trace(model, example_rgb)
    print("  Trace successful")

    # Convert
    try:
        import coremltools as ct
    except ImportError:
        print("\ncoremltools not installed.")
        ts_path = output_path.replace(".mlpackage", ".pt")
        traced.save(ts_path)
        print(f"Saved TorchScript to {ts_path}")
        return

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType("rgb", shape=(1, 3, 32, 32))],
        outputs=[ct.TensorType("output")],
        minimum_deployment_target=ct.target.macOS13,
    )
    print("  CoreML conversion successful")

    if palettize_bits is not None:
        print(f"Palettizing to {palettize_bits}-bit...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )
        op_config = OpPalettizerConfig(nbits=palettize_bits)
        opt_config = OptimizationConfig(global_config=op_config)
        mlmodel = palettize_weights(mlmodel, opt_config)

    mlmodel.save(output_path)
    size_mb = sum(
        f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
    ) / 1e6
    print(f"Saved CoreML SuperRes to {output_path} ({size_mb:.1f} MB)")

    # Validate
    import platform
    if validate and platform.system() != "Darwin":
        print("Skipping validation (CoreML predict requires macOS)")
        validate = False

    if validate:
        print("Validating PyTorch vs CoreML outputs...")
        test_rgb = torch.randn(1, 3, 32, 32).clamp(0, 1)

        with torch.no_grad():
            pt_out = model(test_rgb).numpy()

        ml_pred = mlmodel.predict({"rgb": test_rgb.numpy()})
        ml_out = ml_pred["output"]

        max_diff = np.abs(pt_out - ml_out).max()
        mean_diff = np.abs(pt_out - ml_out).mean()
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        if max_diff < 0.01:
            print("  PASS: outputs match within tolerance")
        else:
            print("  WARN: difference detected (may be acceptable)")


def main():
    parser = argparse.ArgumentParser(description="Convert EASEy-GLYPH models to CoreML")
    parser.add_argument("checkpoint", type=str, help="FlowUNet checkpoint path")
    parser.add_argument("--out", type=str, default="models/glyph.mlpackage",
                        help="FlowUNet output path")
    parser.add_argument("--steps", type=int, default=8,
                        help="Solver steps baked into the model (default: 8)")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun"],
                        help="ODE solver to bake in (default: euler). Heun uses 2 NFE per step.")
    parser.add_argument("--schedule", type=str, default="uniform", choices=["uniform", "cosine", "poly"],
                        help="Time step schedule (default: uniform)")
    parser.add_argument("--superres", type=str, default=None,
                        help="SuperRes checkpoint path (optional)")
    parser.add_argument("--superres-out", type=str, default="models/superres.mlpackage",
                        help="SuperRes output path")
    parser.add_argument("--palettize", type=int, default=None, choices=[4, 6, 8],
                        help="Palettize weights to N bits (reduces model size)")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    convert_unet(
        args.checkpoint,
        args.out,
        steps=args.steps,
        solver=args.solver,
        schedule=args.schedule,
        palettize_bits=args.palettize,
        validate=not args.no_validate,
    )

    if args.superres:
        Path(args.superres_out).parent.mkdir(parents=True, exist_ok=True)
        convert_superres(
            args.superres,
            args.superres_out,
            palettize_bits=args.palettize,
            validate=not args.no_validate,
        )


if __name__ == "__main__":
    main()
