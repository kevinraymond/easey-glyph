"""Shared render pipeline: model loading, sampling, SLERP, enhancements, GridPool.

Used by both offline render (render_audio_reactive.py) and the live server.
"""

import math
import threading
from dataclasses import dataclass, field
from queue import SimpleQueue

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from easey_glyph.model.unet import FlowUNet


# ---------------------------------------------------------------------------
# Enhancement parameters (replaces argparse namespace for the server)
# ---------------------------------------------------------------------------

@dataclass
class EnhancementParams:
    fg_brightness: float = 1.0
    sharpen: float = 0.0
    edge_enhance: str = "off"
    edge_intensity: float = 1.0
    saturation: float = 1.0
    contrast: float = 1.0
    auto_contrast: float | None = None
    posterize: int | None = None
    gamma: float = 1.0
    grain: float = 0.0
    scanlines: int | None = None
    pixel_upscale: bool = False
    render_scale: int = 4
    interp: str = "bilinear"
    opacity: float = 1.0
    alpha_curve: float = 1.0


# ---------------------------------------------------------------------------
# Time schedules & velocity evaluation
# ---------------------------------------------------------------------------

def _time_schedule(steps: int, schedule: str = "uniform") -> list[float]:
    """Return steps+1 time points from 0.0 to 1.0.

    Schedules:
        uniform: evenly spaced (current default)
        cosine:  concentrates at both endpoints (where velocity changes fastest)
        poly:    smoothstep 3t^2 - 2t^3 — mild boundary concentration
    """
    if schedule == "cosine":
        return [0.5 * (1.0 - math.cos(math.pi * i / steps)) for i in range(steps + 1)]
    elif schedule == "poly":
        return [3.0 * (i / steps) ** 2 - 2.0 * (i / steps) ** 3 for i in range(steps + 1)]
    else:  # uniform
        return [i / steps for i in range(steps + 1)]


def _eval_velocity(model, t_val: float, x: torch.Tensor,
                   audio: torch.Tensor | None, cfg_scale: float):
    """Evaluate velocity field, optionally with CFG.

    When cfg_scale <= 0 or audio is None: unconditional (current behavior).
    When cfg_scale > 0: two forward passes + linear combination.
    """
    t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
    if cfg_scale > 0 and audio is not None:
        v_uncond = model(t, x, None)
        v_cond = model(t, x, audio)
        return v_uncond + cfg_scale * (v_cond - v_uncond)
    return model(t, x, audio)


# ---------------------------------------------------------------------------
# ODE solvers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def euler_sample(model, z, steps=50, schedule="uniform", audio=None, cfg_scale=0.0):
    """Euler ODE sampling for flow matching."""
    times = _time_schedule(steps, schedule)
    x = z.clone()
    for i in range(steps):
        dt = times[i + 1] - times[i]
        v = _eval_velocity(model, times[i], x, audio, cfg_scale)
        x = x + v * dt
    return x


@torch.inference_mode()
def euler_sample_from(model, z, t_start, steps=50, schedule="uniform", audio=None, cfg_scale=0.0):
    """Euler ODE sampling starting from time t_start instead of 0."""
    if t_start >= 1.0:
        return z
    # Build time subset from t_start to 1.0
    all_times = _time_schedule(steps, schedule)
    # Find first time >= t_start and build sub-schedule
    sub_times = [t for t in all_times if t >= t_start - 1e-6]
    if not sub_times or sub_times[0] > t_start + 1e-6:
        sub_times.insert(0, t_start)
    if sub_times[-1] < 1.0 - 1e-6:
        sub_times.append(1.0)
    if len(sub_times) < 2:
        return z
    x = z.clone()
    for i in range(len(sub_times) - 1):
        dt = sub_times[i + 1] - sub_times[i]
        v = _eval_velocity(model, sub_times[i], x, audio, cfg_scale)
        x = x + v * dt
    return x


@torch.inference_mode()
def heun_sample(model, z, steps=50, schedule="uniform", audio=None, cfg_scale=0.0):
    """Heun (trapezoidal) ODE sampling — 2nd order, 2 NFE per step."""
    times = _time_schedule(steps, schedule)
    x = z.clone()
    for i in range(steps):
        dt = times[i + 1] - times[i]
        v1 = _eval_velocity(model, times[i], x, audio, cfg_scale)
        x_euler = x + v1 * dt
        v2 = _eval_velocity(model, times[i + 1], x_euler, audio, cfg_scale)
        x = x + (v1 + v2) * 0.5 * dt
    return x


@torch.inference_mode()
def heun_sample_from(model, z, t_start, steps=50, schedule="uniform", audio=None, cfg_scale=0.0):
    """Heun ODE sampling starting from time t_start."""
    if t_start >= 1.0:
        return z
    all_times = _time_schedule(steps, schedule)
    sub_times = [t for t in all_times if t >= t_start - 1e-6]
    if not sub_times or sub_times[0] > t_start + 1e-6:
        sub_times.insert(0, t_start)
    if sub_times[-1] < 1.0 - 1e-6:
        sub_times.append(1.0)
    if len(sub_times) < 2:
        return z
    x = z.clone()
    for i in range(len(sub_times) - 1):
        dt = sub_times[i + 1] - sub_times[i]
        v1 = _eval_velocity(model, sub_times[i], x, audio, cfg_scale)
        x_euler = x + v1 * dt
        v2 = _eval_velocity(model, sub_times[i + 1], x_euler, audio, cfg_scale)
        x = x + (v1 + v2) * 0.5 * dt
    return x


@torch.inference_mode()
def midpoint_sample(model, z, steps=50, schedule="uniform", audio=None, cfg_scale=0.0):
    """Midpoint ODE sampling — 2nd order, 2 NFE per step."""
    times = _time_schedule(steps, schedule)
    x = z.clone()
    for i in range(steps):
        dt = times[i + 1] - times[i]
        t_mid = (times[i] + times[i + 1]) * 0.5
        v1 = _eval_velocity(model, times[i], x, audio, cfg_scale)
        x_mid = x + v1 * dt * 0.5
        v_mid = _eval_velocity(model, t_mid, x_mid, audio, cfg_scale)
        x = x + v_mid * dt
    return x


@torch.inference_mode()
def midpoint_sample_from(model, z, t_start, steps=50, schedule="uniform", audio=None, cfg_scale=0.0):
    """Midpoint ODE sampling starting from time t_start."""
    if t_start >= 1.0:
        return z
    all_times = _time_schedule(steps, schedule)
    sub_times = [t for t in all_times if t >= t_start - 1e-6]
    if not sub_times or sub_times[0] > t_start + 1e-6:
        sub_times.insert(0, t_start)
    if sub_times[-1] < 1.0 - 1e-6:
        sub_times.append(1.0)
    if len(sub_times) < 2:
        return z
    x = z.clone()
    for i in range(len(sub_times) - 1):
        dt = sub_times[i + 1] - sub_times[i]
        t_mid = (sub_times[i] + sub_times[i + 1]) * 0.5
        v1 = _eval_velocity(model, sub_times[i], x, audio, cfg_scale)
        x_mid = x + v1 * dt * 0.5
        v_mid = _eval_velocity(model, t_mid, x_mid, audio, cfg_scale)
        x = x + v_mid * dt
    return x


# ---------------------------------------------------------------------------
# Unified dispatchers
# ---------------------------------------------------------------------------

_SOLVERS = {
    "euler": euler_sample,
    "heun": heun_sample,
    "midpoint": midpoint_sample,
}
_SOLVERS_FROM = {
    "euler": euler_sample_from,
    "heun": heun_sample_from,
    "midpoint": midpoint_sample_from,
}


def ode_sample(model, z, steps=50, solver="euler", schedule="uniform",
               audio=None, cfg_scale=0.0):
    """Unified ODE sampling dispatcher."""
    fn = _SOLVERS.get(solver, euler_sample)
    return fn(model, z, steps=steps, schedule=schedule, audio=audio, cfg_scale=cfg_scale)


def ode_sample_from(model, z, t_start, steps=50, solver="euler", schedule="uniform",
                    audio=None, cfg_scale=0.0):
    """Unified ODE sampling dispatcher (partial solve from t_start)."""
    fn = _SOLVERS_FROM.get(solver, euler_sample_from)
    return fn(model, z, t_start, steps=steps, schedule=schedule, audio=audio, cfg_scale=cfg_scale)


# ---------------------------------------------------------------------------
# High-level generation helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_single_grid(model, model_cfg, device, steps=8,
                         solver="euler", schedule="uniform",
                         audio=None, cfg_scale=0.0):
    """Generate a single glyph grid (for realtime mode)."""
    in_ch = model_cfg.get("in_channels", 16)
    img_size = model_cfg.get("image_size", 32)
    z = torch.randn(1, in_ch, img_size, img_size, device=device)
    return ode_sample(model, z, steps=steps, solver=solver, schedule=schedule,
                      audio=audio, cfg_scale=cfg_scale)[0].cpu()


@torch.inference_mode()
def generate_single_grid_img2img(model, model_cfg, device, source_grid,
                                 strength=0.5, steps=8,
                                 solver="euler", schedule="uniform",
                                 audio=None, cfg_scale=0.0):
    """Generate a grid starting from a partially-noised source grid."""
    in_ch = model_cfg.get("in_channels", 16)
    img_size = model_cfg.get("image_size", 32)
    noise = torch.randn(1, in_ch, img_size, img_size, device=device)
    source = source_grid.unsqueeze(0).to(device)
    t_start = 1.0 - strength
    x_start = (1 - t_start) * noise + t_start * source
    result = ode_sample_from(model, x_start, t_start, steps=steps, solver=solver,
                             schedule=schedule, audio=audio, cfg_scale=cfg_scale)
    return result[0].cpu()


def load_model(checkpoint_path, device):
    """Load trained FlowUNet from checkpoint (.pt or .safetensors)."""
    from pathlib import Path

    path = Path(checkpoint_path)

    if path.suffix == ".json":
        import json
        from safetensors.torch import load_file

        with open(path) as f:
            model_cfg = json.load(f).get("model", {})
        sf_path = path.with_suffix(".safetensors")
        if not sf_path.exists():
            raise FileNotFoundError(
                f"Expected companion weights file: {sf_path}")
        state_dict = load_file(str(sf_path), device="cpu")
    elif path.suffix == ".safetensors":
        from safetensors.torch import load_file
        import json

        state_dict = load_file(str(path), device="cpu")
        cfg_path = path.with_suffix(".json")
        if cfg_path.exists():
            with open(cfg_path) as f:
                model_cfg = json.load(f).get("model", {})
        else:
            model_cfg = {}
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        model_cfg = cfg.get("model", {})
        state_dict = ckpt.get("model_ema", ckpt["model"])
        if "model_ema" in ckpt:
            print("Loaded EMA weights")
        else:
            print("Loaded model weights")

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

    model.load_state_dict(state_dict)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params ({params / 1e6:.2f}M)")
    return model, model_cfg


def generate_grids(model, model_cfg, num_grids, steps, device,
                   solver="euler", schedule="uniform",
                   audio=None, cfg_scale=0.0):
    """Generate glyph grids from the model in batches."""
    in_ch = model_cfg.get("in_channels", 16)
    img_size = model_cfg.get("image_size", 32)

    solver_label = solver.capitalize()
    nfe = steps * (2 if solver in ("heun", "midpoint") else 1)
    print(f"Generating {num_grids} glyph grids with {steps} {solver_label} steps ({nfe} NFE, {schedule})...")
    batch_size = min(num_grids, 8)
    all_grids = []
    for start in range(0, num_grids, batch_size):
        n = min(batch_size, num_grids - start)
        z = torch.randn(n, in_ch, img_size, img_size, device=device)
        grids = ode_sample(model, z, steps=steps, solver=solver, schedule=schedule,
                           audio=audio, cfg_scale=cfg_scale)
        all_grids.extend([g.cpu() for g in grids])
        print(f"  {len(all_grids)}/{num_grids}", flush=True)
    return all_grids


# ---------------------------------------------------------------------------
# SLERP
# ---------------------------------------------------------------------------

def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    v0_flat = v0.flatten() if v0.dtype == torch.float32 else v0.flatten().float()
    v1_flat = v1.flatten() if v1.dtype == torch.float32 else v1.flatten().float()

    n0 = v0_flat.norm()
    n1 = v1_flat.norm()
    if n0 < 1e-8 or n1 < 1e-8:
        return (1 - t) * v0 + t * v1

    u0 = v0_flat / n0
    u1 = v1_flat / n1
    dot = torch.dot(u0, u1).clamp(-1, 1)
    omega = torch.acos(dot)

    if omega.abs() < 1e-4:
        return (1 - t) * v0 + t * v1

    sin_omega = torch.sin(omega)
    s0 = torch.sin((1 - t) * omega) / sin_omega
    s1 = torch.sin(t * omega) / sin_omega
    result = s0 * v0_flat + s1 * v1_flat
    return result.reshape(v0.shape)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

INTERP_METHODS = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}

# Gamma LUT cache — avoids recomputing 256 pow() calls per frame
_gamma_lut_cache: dict[float, list[int]] = {}

# Alpha LUT cache — combined alpha_curve + opacity in a single uint8→uint8 table
_alpha_lut_cache: dict[tuple[float, float], np.ndarray] = {}

EDGE_KERNELS = {
    "all": ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 9, -1, -1, -1, -1), scale=1, offset=0),
    "horizontal": ImageFilter.Kernel((3, 3), (0, -1, 0, 0, 3, 0, 0, -1, 0), scale=1, offset=0),
    "vertical": ImageFilter.Kernel((3, 3), (0, 0, 0, -1, 3, -1, 0, 0, 0), scale=1, offset=0),
    "diagonal_bwd": ImageFilter.Kernel((3, 3), (-1, 0, 0, 0, 3, 0, 0, 0, -1), scale=1, offset=0),
    "diagonal_fwd": ImageFilter.Kernel((3, 3), (0, 0, -1, 0, 3, 0, -1, 0, 0), scale=1, offset=0),
    "emboss": ImageFilter.EMBOSS,
    "outline": ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 9, -1, -1, -1, -1), scale=2, offset=0),
}


def scale2x(img: np.ndarray) -> np.ndarray:
    """Scale2x/EPX pixel-art upscaler. Doubles resolution (HxW -> 2Hx2W).

    Edge-directed rules: no new colors introduced, preserves pixel-art character.
    Input/output: [H, W, C] uint8 array. Fully vectorized with numpy.
    """
    h, w, c = img.shape
    # Pad with edge replication to get neighbors without boundary checks
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='edge')
    p = img                           # center  [H, W, C]
    a = padded[0:h, 1:w+1]           # up
    b = padded[1:h+1, 2:w+2]         # right
    cpx = padded[2:h+2, 1:w+1]       # down
    d = padded[1:h+1, 0:w]           # left

    # Boolean equality masks (all channels must match) -> [H, W]
    da = np.all(d == a, axis=2)
    dc = np.all(d == cpx, axis=2)
    ab = np.all(a == b, axis=2)
    cb = np.all(cpx == b, axis=2)
    ad = np.all(a == d, axis=2)       # same as da
    bc = np.all(b == cpx, axis=2)     # same as cb

    # Scale2x rules:
    # e0 = a if (d==a and d!=c and a!=b) else p
    # e1 = a if (a==b and a!=d and b!=c) else p  [original uses 'a' as source]
    # e2 = d if (d==c and d!=a and c!=b) else p
    # e3 = c if (c==b and d!=c and a!=b) else p
    m0 = (da & ~dc & ~ab)[:, :, np.newaxis]
    m1 = (ab & ~ad & ~bc)[:, :, np.newaxis]
    m2 = (dc & ~da & ~cb)[:, :, np.newaxis]
    m3 = (cb & ~dc & ~ab)[:, :, np.newaxis]

    e0 = np.where(m0, a, p)
    e1 = np.where(m1, a, p)
    e2 = np.where(m2, d, p)
    e3 = np.where(m3, cpx, p)

    # Interleave into 2H x 2W output
    out = np.empty((h * 2, w * 2, c), dtype=img.dtype)
    out[0::2, 0::2] = e0
    out[0::2, 1::2] = e1
    out[1::2, 0::2] = e2
    out[1::2, 1::2] = e3

    return out


def apply_intermediate_enhancements(frame_img: Image.Image, params, enh: EnhancementParams) -> np.ndarray:
    """Apply intermediate-resolution effects and return numpy array.

    Pipeline: alpha curve + opacity -> sharpen -> edge (RGBA) -> saturation
    -> contrast -> auto_contrast -> posterize -> gamma (RGB).

    Alpha curve applied first on smooth alpha, then sharpen/edge on RGBA
    (crisp alpha edges), then color ops on RGB only.
    Returns [H, W, 4] uint8 RGBA numpy array.
    """
    img = frame_img

    # Alpha curve + opacity BEFORE convolutions — curve needs smooth alpha
    # values to be effective; convolutions then sharpen the curved result.
    # Exponent boosted (curve ** 1.5) to compensate for convolution re-sharpening.
    needs_alpha_lut = enh.alpha_curve != 1.0 or enh.opacity < 1.0
    if needs_alpha_lut:
        curve, opac = enh.alpha_curve, enh.opacity
        # Boost exponent: curve**1.5 gives more punch at extremes,
        # neutral (1.0) stays 1.0. Compensates for convolutions partially
        # undoing the curve by re-sharpening alpha toward binary.
        eff_curve = curve ** 1.5
        lut_key = (round(curve, 4), round(opac, 4))
        lut = _alpha_lut_cache.get(lut_key)
        if lut is None:
            lut = np.empty(256, dtype=np.uint8)
            for i in range(256):
                v = ((i / 255.0) ** eff_curve) * opac * 255.0
                lut[i] = max(0, min(255, int(v + 0.5)))
            _alpha_lut_cache[lut_key] = lut
        # PIL point() with per-channel LUT: identity for RGB, curve for A
        img = img.point(list(range(256)) * 3 + lut.tolist())

    # Sharpen + edge on RGBA — preserves crisp alpha edges
    # Sharpen (UnsharpMask)
    if enh.sharpen > 0:
        amount = enh.sharpen * params.sharpen_amount
        if amount > 0:
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(amount), threshold=0))

    # Edge enhance (with intensity blending)
    if enh.edge_enhance != "off":
        kernel = EDGE_KERNELS.get(enh.edge_enhance)
        if kernel is not None:
            if enh.edge_intensity >= 1.0:
                img = img.filter(kernel)
            elif enh.edge_intensity > 0.0:
                edge_img = img.filter(kernel)
                img = Image.blend(img, edge_img, enh.edge_intensity)

    # Split alpha — color ops don't need alpha channel
    alpha = img.getchannel("A")
    img = img.convert("RGB")

    # Saturation
    effective_sat = enh.saturation * params.saturation
    if effective_sat != 1.0:
        img = ImageEnhance.Color(img).enhance(effective_sat)

    # Contrast
    effective_contrast = enh.contrast * params.contrast
    if effective_contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(effective_contrast)

    # Auto-contrast, posterize, gamma — all RGB, no alpha concerns
    if enh.auto_contrast is not None:
        img = ImageOps.autocontrast(img, cutoff=enh.auto_contrast)
    if enh.posterize is not None:
        img = ImageOps.posterize(img, max(1, min(8, enh.posterize)))
    if enh.gamma != 1.0:
        gamma_key = round(enh.gamma, 4)
        lut = _gamma_lut_cache.get(gamma_key)
        if lut is None:
            inv_gamma = 1.0 / enh.gamma
            lut = [int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)]
            _gamma_lut_cache[gamma_key] = lut
        img = img.point(lut * 3)

    # Convert to numpy RGBA (recombine RGB + alpha)
    rgb_arr = np.array(img)                    # [H, W, 3] uint8
    alpha_arr = np.array(alpha)                # [H, W] uint8
    h, w = rgb_arr.shape[:2]
    arr = np.empty((h, w, 4), dtype=np.uint8)
    arr[:, :, :3] = rgb_arr
    arr[:, :, 3] = alpha_arr

    return arr


def apply_grain(arr: np.ndarray, grain: float, output_size: tuple[int, int]) -> np.ndarray:
    """Apply film grain, upscaling to a grain floor if intermediate is too small.

    Grain floor is 1/3 of output dims per axis, giving ~3x3 grain blocks at
    output resolution.  When intermediate is already >= grain floor, applies
    grain directly without upscaling.

    Args:
        arr: [H, W, 4] uint8 RGBA numpy array.
        grain: grain intensity (slider value, 0-20 range).
        output_size: (width, height) of the target output.

    Returns:
        [H, W, 4] uint8 RGBA numpy array (may be larger than input if upscaled).
    """
    if grain <= 0:
        return arr

    h_in, w_in = arr.shape[:2]
    w_out, h_out = output_size

    # Grain floor: 1/3 of output dims — gives ~3x3 blocks at output
    grain_h = max(h_in, h_out // 3)
    grain_w = max(w_in, w_out // 3)

    if grain_h > h_in or grain_w > w_in:
        # Upscale intermediate to grain floor via NEAREST (np.take)
        y_idx = (np.arange(grain_h) * h_in // grain_h).astype(np.intp)
        x_idx = (np.arange(grain_w) * w_in // grain_w).astype(np.intp)
        arr = np.take(np.take(arr, y_idx, axis=0), x_idx, axis=1)
    else:
        arr = arr.copy()

    amp = int(grain * 12.75)
    h, w = arr.shape[:2]
    noise = np.random.randint(-amp, amp + 1, size=(h, w), dtype=np.int16)
    arr[:, :, :3] = np.clip(
        arr[:, :, :3].astype(np.int16) + noise[:, :, np.newaxis], 0, 255,
    ).astype(np.uint8)
    return arr


def apply_output_enhancements(intermediate: np.ndarray, enh: EnhancementParams, output_size: tuple[int, int]) -> np.ndarray:
    """Apply output-resolution effects: NEAREST resize + scanlines.

    Uses np.take for NEAREST resize (~1.6ms vs ~11ms PIL at 768→1080p).
    Grain, alpha_curve, and opacity are applied at intermediate resolution
    for performance (exact same result since NEAREST preserves pixel values).

    Args:
        intermediate: [H, W, 4] uint8 RGBA numpy array (from apply_intermediate_enhancements).
        enh: EnhancementParams (slider/CLI values).
        output_size: (width, height) final output resolution.

    Returns:
        [H, W, 4] uint8 RGBA numpy array at output resolution.
    """
    h_in, w_in = intermediate.shape[:2]
    w_out, h_out = output_size

    # NEAREST resize via np.take 2-pass (contiguous output, ~1.6ms at 768→1080p)
    if h_in == h_out and w_in == w_out:
        arr = intermediate.copy()
    else:
        y_idx = (np.arange(h_out) * h_in // h_out).astype(np.intp)
        x_idx = (np.arange(w_out) * w_in // w_out).astype(np.intp)
        arr = np.take(np.take(intermediate, y_idx, axis=0), x_idx, axis=1)

    # Scanlines (integer right-shift — avoids uint8→float32→uint8 round-trip)
    if enh.scanlines is not None and enh.scanlines > 0:
        arr[::enh.scanlines, :, :3] >>= 1

    return arr


def apply_enhancements(frame_img: Image.Image, params, enh: EnhancementParams, output_size: tuple[int, int]) -> np.ndarray:
    """Apply post-processing enhancements to a frame (backward-compat wrapper).

    Pipeline order:
      intermediate resolution: sharpen -> edge_enhance -> saturation -> contrast
        -> auto_contrast -> posterize -> gamma -> grain
      then NEAREST to output resolution: scanlines -> alpha curve -> opacity

    Args:
        frame_img: PIL Image at intermediate resolution.
        params: RenderParams with audio-reactive values.
        enh: EnhancementParams (slider/CLI values).
        output_size: (width, height) final output resolution.

    Returns:
        [H, W, 4] uint8 RGBA numpy array at output resolution.
    """
    intermediate = apply_intermediate_enhancements(frame_img, params, enh)
    intermediate = apply_grain(intermediate, enh.grain, output_size)
    return apply_output_enhancements(intermediate, enh, output_size)


# ---------------------------------------------------------------------------
# GridPool — background grid generation for live use
# ---------------------------------------------------------------------------

class GridPool:
    """Background thread that continuously generates grids and fills a queue."""

    def __init__(self, model, model_cfg, device, steps=8, pool_size=64, low_water=16):
        self.model = model
        self.model_cfg = model_cfg
        self.device = device
        self.steps = steps
        self.pool_size = pool_size
        self.low_water = low_water

        # Solver/schedule/CFG settings
        self.solver: str = "euler"
        self.schedule: str = "uniform"
        self.cfg_scale: float = 0.0
        self.cfg_audio: str = "random"  # "random" | "live"
        self._live_audio_features: torch.Tensor | None = None  # [1, 12] snapshot

        self._queue: SimpleQueue[torch.Tensor] = SimpleQueue()
        self._stop_event = threading.Event()
        self._paused = threading.Event()  # set = paused, clear = running
        self._thread: threading.Thread | None = None

        self.in_ch = model_cfg.get("in_channels", 16)
        self.img_size = model_cfg.get("image_size", 32)

        # CoreML backend (set externally when available)
        self.coreml_model = None

    def start(self):
        """Start background generation thread (non-blocking)."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="grid-pool")
        self._thread.start()

    def _make_audio_tensor(self, n: int) -> torch.Tensor | None:
        """Build audio conditioning tensor for CFG."""
        if self.cfg_scale <= 0:
            return None
        if self.cfg_audio == "live" and self._live_audio_features is not None:
            return self._live_audio_features.expand(n, -1).to(self.device)
        # Random uniform audio vector (matches training distribution)
        return torch.rand(n, 12, device=self.device)

    def set_live_audio_features(self, features: torch.Tensor | None):
        """Snapshot current audio features for pool generation."""
        self._live_audio_features = features

    def _generate_batch(self, count: int) -> list[torch.Tensor]:
        batch_size = min(count, 8)
        grids = []
        for start in range(0, count, batch_size):
            if self._paused.is_set() or self._stop_event.is_set():
                break
            n = min(batch_size, count - start)
            if self.coreml_model is not None:
                for _ in range(n):
                    noise = np.random.randn(1, self.in_ch, self.img_size, self.img_size).astype(np.float32)
                    grid_np = self.coreml_model.generate_grid(noise)
                    grids.append(torch.from_numpy(grid_np[0]))
            else:
                z = torch.randn(n, self.in_ch, self.img_size, self.img_size, device=self.device)
                audio = self._make_audio_tensor(n)
                with torch.inference_mode():
                    batch = ode_sample(self.model, z, steps=self.steps,
                                       solver=self.solver, schedule=self.schedule,
                                       audio=audio, cfg_scale=self.cfg_scale)
                grids.extend([g.cpu() for g in batch])
        return grids

    def _run(self):
        while not self._stop_event.is_set():
            if self._paused.is_set():
                self._stop_event.wait(timeout=0.5)
                continue
            if self._queue.qsize() < self.low_water:
                # Urgent: one batch per iteration, no extra sleep
                batch = min(8, self.pool_size - self._queue.qsize())
                for g in self._generate_batch(batch):
                    if self._stop_event.is_set():
                        return
                    self._queue.put(g)
            elif self._queue.qsize() < self.pool_size:
                # Trickle: fill toward capacity with yields between batches
                for g in self._generate_batch(min(8, self.pool_size - self._queue.qsize())):
                    if self._stop_event.is_set():
                        return
                    self._queue.put(g)
                self._stop_event.wait(timeout=0.05)  # 50ms yield
            else:
                self._stop_event.wait(timeout=0.1)  # Full — idle

    def next_grid(self) -> torch.Tensor | None:
        """Pop a grid from the pool (non-blocking). Returns None if empty."""
        try:
            return self._queue.get_nowait()
        except Exception:
            return None

    @property
    def size(self) -> int:
        return self._queue.qsize()

    def pause(self):
        """Pause the background generation thread."""
        self._paused.set()

    def resume(self):
        """Resume the background generation thread."""
        self._paused.clear()

    def update_steps(self, steps: int):
        self.steps = steps

    def update_solver(self, solver: str):
        if solver in _SOLVERS:
            self.solver = solver

    def update_schedule(self, schedule: str):
        if schedule in ("uniform", "cosine", "poly"):
            self.schedule = schedule

    def update_cfg(self, cfg_scale: float | None = None, cfg_audio: str | None = None):
        if cfg_scale is not None:
            self.cfg_scale = max(0.0, cfg_scale)
        if cfg_audio is not None and cfg_audio in ("random", "live"):
            self.cfg_audio = cfg_audio

    def flush_and_refill(self):
        """Drain the queue — background thread will refill."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
