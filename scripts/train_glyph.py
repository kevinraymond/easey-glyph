#!/usr/bin/env python3
"""Train EASEy-GLYPH on preprocessed glyph grids using vanilla flow matching.

Usage:
    # Single GPU (outdir auto-derived from dataset path)
    uv run scripts/train_glyph.py --config configs/glyph_base.yaml
    # -> training-runs/abstract-v2/ (derived from datasets/abstract-v2/glyph-abstract-v2-10k.pt)

    # Multi-GPU (2x 4090)
    torchrun --nproc_per_node=2 scripts/train_glyph.py --config configs/glyph_base.yaml

    # Override output dir
    uv run scripts/train_glyph.py --config configs/glyph_base.yaml --outdir training-runs/custom
"""

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Flow matching: pure-GPU, no external dependencies
# ---------------------------------------------------------------------------

class FlowMatcher:
    """Vanilla conditional flow matching: uniform time, random pairing, no weighting."""

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def sample(self, x0, x1):
        B = x0.shape[0]

        # Uniform time sampling
        t = torch.rand(B, device=x0.device)
        t = t.clamp(1e-5, 1 - 1e-5)

        t_expand = t.reshape(-1, *([1] * (x0.dim() - 1)))
        eps = torch.randn_like(x0) * self.sigma if self.sigma > 0 else 0
        xt = (1 - t_expand) * x0 + t_expand * x1 + eps
        ut = x1 - x0  # Target velocity

        # Uniform weighting
        weight = torch.ones(B, device=x0.device)

        return t, xt, ut, weight


# ---------------------------------------------------------------------------
# Audio conditioning generator (simplified, for training with random audio)
# ---------------------------------------------------------------------------

class AudioConditioningGenerator:
    """Fixed per-image audio labels with unconditional dropout."""

    def __init__(self, num_images: int, device: torch.device,
                 uncond_ratio: float = 0.2, labels: Tensor | None = None):
        self.uncond_ratio = uncond_ratio
        if labels is not None:
            self.labels = labels.to(device)
        else:
            self.labels = torch.rand(num_images, 12, device=device)

    def sample(self, indices: Tensor) -> Tensor | None:
        if torch.rand(1).item() < self.uncond_ratio:
            return None
        return self.labels[indices]


from easey_glyph.model.unet import FlowUNet
from easey_glyph.training.dataset import GlyphDataset
from easey_glyph.training.ema import ema_update
from easey_glyph.training.smooth_reg import smooth_regularization


def derive_outdir(data_path: str) -> str:
    """Derive training output directory from dataset path.

    Convention: datasets/{type}/glyph-{type}-{count}.pt -> training-runs/{type}/
    Fallback: use the parent directory name, or 'glyph' if at top level.
    """
    p = Path(data_path)
    parent_name = p.parent.name
    if parent_name and parent_name != "datasets":
        return f"training-runs/{parent_name}"
    # Fallback: extract type from filename like glyph-{type}-{count}.pt
    stem = p.stem
    if stem.startswith("glyph-"):
        parts = stem[len("glyph-"):].split("-")
        # Remove trailing count segment (e.g. "10k")
        if len(parts) > 1:
            return f"training-runs/{'-'.join(parts[:-1])}"
        return f"training-runs/{parts[0]}"
    return "training-runs/glyph"


# ---------------------------------------------------------------------------
# DDP gradient sync
# ---------------------------------------------------------------------------

def sync_gradients(model: torch.nn.Module):
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


# ---------------------------------------------------------------------------
# Euler ODE solver
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Snapshot: render glyph grids to PNG
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_snapshot(
    model: torch.nn.Module,
    step: int,
    outdir: Path,
    device: torch.device,
    glyph_masks: Tensor,
    glyph_embeddings: Tensor,
    in_channels: int = 16,
    image_size: int = 32,
    n: int = 16,
    steps: int = 50,
):
    """Generate samples and render as glyph art PNGs."""
    from easey_glyph.glyph.converter import render_glyph_grid

    raw = model
    if hasattr(raw, '_orig_mod'):
        raw = raw._orig_mod
    if hasattr(raw, 'module'):
        raw = raw.module
    torch.cuda.empty_cache()
    raw.eval()

    z = torch.randn(n, in_channels, image_size, image_size, device=device)
    samples = euler_sample(raw, z, audio=None, steps=steps)

    # Render each sample to a 256x256 image
    cols = 4
    rows = math.ceil(n / cols)
    pixel_h, pixel_w = image_size * 8, image_size * 8  # 256x256 for 32x32 grid
    grid_img = np.zeros((rows * pixel_h, cols * pixel_w, 4), dtype=np.uint8)

    for i, sample in enumerate(samples):
        r, c = divmod(i, cols)
        rendered = render_glyph_grid(sample, glyph_masks, glyph_embeddings)
        grid_img[
            r * pixel_h:(r + 1) * pixel_h,
            c * pixel_w:(c + 1) * pixel_w,
        ] = rendered

    Image.fromarray(grid_img).save(outdir / f"snapshot_{step:06d}.png")
    raw.train()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # CLI overrides
    if args.data:
        data_cfg["path"] = args.data
    if args.batch:
        train_cfg["batch_size"] = args.batch
    if args.lr:
        train_cfg["lr"] = args.lr
    if args.kimg:
        train_cfg["total_kimg"] = args.kimg
    if args.smooth_reg_weight is not None:
        train_cfg["smooth_reg_weight"] = args.smooth_reg_weight
    if args.snap_kimg is not None:
        train_cfg["snap_kimg"] = args.snap_kimg
    if args.ema_decay is not None:
        train_cfg["ema_decay"] = args.ema_decay

    # Auto-derive output directory from data path
    outdir_auto = args.outdir is None
    if outdir_auto:
        args.outdir = derive_outdir(data_cfg["path"])
    if args.suffix:
        args.outdir = f"{args.outdir}-{args.suffix}"

    # Distributed setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_BUFFSIZE", "8388608")
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    is_main = rank == 0
    use_amp = train_cfg.get("fp16", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else (torch.float16 if use_amp else torch.float32)

    # Output directory
    outdir = Path(args.outdir)
    if is_main:
        if outdir_auto:
            print(f"Output: {outdir} (auto-derived from data path)")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "snapshots").mkdir(exist_ok=True)

    # Load glyph atlas + embeddings (for snapshot rendering)
    from easey_glyph.glyph.rasterizer import build_glyph_atlas
    from easey_glyph.glyph.embedding import compute_embeddings

    glyph_masks = build_glyph_atlas()        # [289, 8, 8]
    glyph_embeddings = compute_embeddings()  # [289, 8]

    # Validate data path
    data_path = data_cfg["path"]
    if data_path.endswith(".zip"):
        print(f"Error: --data expects a preprocessed .pt file, not a raw ZIP.")
        print(f"Run:  uv run scripts/preprocess_dataset.py --input {data_path}")
        sys.exit(1)
    if not data_path.endswith(".pt"):
        print(f"Error: --data expects a .pt file, got: {data_path}")
        sys.exit(1)

    # Dataset
    dataset = GlyphDataset(
        data_cfg["path"],
        max_images=args.overfit if args.overfit else None,
    )
    if is_main:
        print(f"Dataset: {len(dataset)} glyph grids from {data_cfg['path']}")

    batch_size = train_cfg["batch_size"]
    batch_per_gpu = batch_size // world_size
    effective_batch = min(batch_per_gpu, len(dataset))

    # GPU-resident dataset
    dataset.to_gpu(device)

    in_channels = model_cfg.get("in_channels", 16)
    out_channels = model_cfg.get("out_channels", 16)
    image_size = model_cfg.get("image_size", 32)

    # Model
    model = FlowUNet(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=model_cfg["base_channels"],
        channel_mult=model_cfg["channel_mult"],
        num_res_blocks=model_cfg["num_res_blocks"],
        attention_resolutions=model_cfg["attention_resolutions"],
        use_depthwise_separable=model_cfg.get("use_depthwise_separable", False),
        audio_dim=model_cfg.get("audio_dim", 12),
        time_emb_dim=model_cfg.get("time_emb_dim", 128),
    ).to(device)

    model_ema = copy.deepcopy(model)
    model_ema.eval()

    if is_main:
        params = sum(p.numel() for p in model.parameters())
        print(f"FlowUNet: {params:,} params ({params/1e6:.2f}M)")

    # Performance
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)
        if is_main:
            print("torch.compile(default) + cudnn.benchmark + tf32")

    # DDP wrapper
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank], gradient_as_bucket_view=True, find_unused_parameters=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], betas=(0.9, 0.999))

    # Mixed precision
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Flow matching
    FM = FlowMatcher(sigma=0.0)

    # Audio conditioning (fixed per-image labels)
    uncond_ratio = train_cfg.get("uncond_ratio", 0.2)
    audio_labels = None
    if args.audio_labels:
        label_data = torch.load(args.audio_labels, map_location="cpu", weights_only=True)
        audio_labels = label_data["labels"] if "labels" in label_data else label_data
        if is_main:
            print(f"Audio labels: loaded {audio_labels.shape} from {args.audio_labels}")
    elif dataset.labels is not None:
        audio_labels = dataset.labels
        if is_main:
            print(f"Audio labels: using dataset-embedded labels {audio_labels.shape}")
    else:
        if is_main:
            print("Audio labels: random (no pre-computed labels found)")
    audio_gen = AudioConditioningGenerator(
        len(dataset), device, uncond_ratio=uncond_ratio, labels=audio_labels,
    )

    # Training config
    total_kimg = train_cfg["total_kimg"]
    total_images = int(total_kimg * 1000)
    images_per_step = effective_batch * world_size
    total_steps = total_images // images_per_step
    snap_interval = max(1, int(train_cfg["snap_kimg"] * 1000 // images_per_step))
    smooth_weight = train_cfg.get("smooth_reg_weight", 0.1)
    smooth_interval = train_cfg.get("smooth_reg_interval", 4)
    ema_decay = train_cfg.get("ema_decay", 0.9999)

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        (model.module if ddp else model).load_state_dict(ckpt["model"])
        model_ema.load_state_dict(ckpt["model_ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        if is_main:
            print(f"Resumed from step {start_step}")

    warmup_steps = args.warmup_steps
    base_lr = train_cfg["lr"]

    if is_main:
        print(f"LR: {base_lr} with {warmup_steps}-step warmup")
        print(f"Training for {total_kimg} kimg = {total_steps} steps")
        print(f"Snapshot every {train_cfg['snap_kimg']} kimg = {snap_interval} steps")
        print(f"Smooth reg: weight={smooth_weight}, interval={smooth_interval}")
        print(f"Device: {device}, AMP: {amp_dtype if use_amp else 'off'}")

    # Training loop
    step = start_step
    kimg_shown = step * images_per_step / 1000
    t0 = time.time()
    log_interval = min(50, max(1, total_steps // 20))

    while step < total_steps:
        real, indices = dataset.sample_batch(effective_batch)
        B = real.shape[0]

        optimizer.zero_grad(set_to_none=True)

        audio_cond = audio_gen.sample(indices)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            x1 = real
            x0 = torch.randn_like(x1)

            t, xt, ut, snr_weight = FM.sample(x0, x1)

            vt = model(t, xt, audio_cond)
            per_sample_loss = (vt - ut).pow(2).mean(dim=(1, 2, 3))
            loss = (snr_weight * per_sample_loss).mean()

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Smooth regularization
        smooth_loss_val = 0.0
        if smooth_weight > 0 and step % smooth_interval == 0:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                smooth_loss = smooth_regularization(model, t, xt.detach(), audio_cond)
                scaled_smooth = smooth_weight * smooth_interval * smooth_loss
            if use_scaler:
                scaler.scale(scaled_smooth).backward()
            else:
                scaled_smooth.backward()
            smooth_loss_val = smooth_loss.item()

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # LR schedule: linear warmup then cosine decay to 0
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # EMA update
        raw_model = model.module if ddp else model
        ema_update(model_ema, raw_model, ema_decay)

        step += 1
        kimg_shown = step * images_per_step / 1000

        # Logging
        if is_main and step % log_interval == 0:
            elapsed = time.time() - t0
            kimg_per_sec = kimg_shown / elapsed if elapsed > 0 else 0
            smooth_str = f" | smooth {smooth_loss_val:.4f}" if smooth_loss_val > 0 else ""
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:6d} | kimg {kimg_shown:7.1f}/{total_kimg} | "
                f"loss {loss.item():.4f}{smooth_str} | "
                f"lr {current_lr:.2e} | {kimg_per_sec:.2f} kimg/s",
                flush=True,
            )

        # Snapshot
        if is_main and step % snap_interval == 0:
            ema_warmup = int(1.0 / (1.0 - ema_decay))
            snap_model = model_ema if step >= ema_warmup else model
            save_snapshot(
                snap_model, step, outdir / "snapshots", device,
                glyph_masks, glyph_embeddings,
                in_channels=in_channels, image_size=image_size,
            )

            ckpt = {
                "model": (model.module if ddp else model).state_dict(),
                "model_ema": model_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "kimg": kimg_shown,
                "config": cfg,
            }
            torch.save(ckpt, outdir / f"checkpoint_{step:06d}.pt")
            print(f"  Saved snapshot + checkpoint at step {step}", flush=True)

    # Final save
    if is_main:
        ckpt = {
            "model": (model.module if ddp else model).state_dict(),
            "model_ema": model_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "kimg": kimg_shown,
            "config": cfg,
        }
        torch.save(ckpt, outdir / "final.pt")
        ema_warmup = int(1.0 / (1.0 - ema_decay))
        snap_model = model_ema if step >= ema_warmup else model
        save_snapshot(
            snap_model, step, outdir / "snapshots", device,
            glyph_masks, glyph_embeddings,
            in_channels=in_channels, image_size=image_size,
        )
        print(f"Training complete. {kimg_shown:.1f} kimg in {time.time() - t0:.0f}s", flush=True)
        print(f"Final model saved to {outdir / 'final.pt'}", flush=True)

    if ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train EASEy-GLYPH")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Override dataset path")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (auto-derived from data path if omitted)")
    parser.add_argument("--suffix", type=str, default=None, help="Append -SUFFIX to auto-derived outdir (e.g. --suffix rand → abstract-v2-rand)")
    parser.add_argument("--kimg", type=float, default=None, help="Override total training kimg")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--overfit", type=int, default=None, help="Overfit on N images")
    parser.add_argument("--smooth-reg-weight", type=float, default=None)
    parser.add_argument("--snap-kimg", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear LR warmup steps")
    parser.add_argument("--audio-labels", type=str, default=None,
                        help="Path to .pt file with pre-computed audio labels (spectral demo)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
