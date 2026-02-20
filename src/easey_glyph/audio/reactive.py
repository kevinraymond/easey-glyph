"""Audio-reactive parameter mapping for Braille rendering.

Maps AudioFrame features to visual parameters per frame.
"""

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from .frame import AudioFrame


@dataclass
class RenderParams:
    """Per-frame rendering parameters."""
    dot_radius: float
    visibility_threshold: float
    bg_alpha_scale: float
    fg_brightness: float
    # Enhancement params (audio-reactive)
    sharpen_amount: float = 0.0      # 0.0 = off, 1.0 = full strength
    saturation: float = 1.0          # 1.0 = unchanged
    contrast: float = 1.0            # 1.0 = unchanged
    grain_strength: float = 0.0      # 0.0 = off


def frame_to_render_params(
    frame: AudioFrame,
    dot_radius_range: tuple[float, float] = (1.0, 4.0),
    visibility_range: tuple[float, float] = (0.0, 0.6),
    bg_alpha_range: tuple[float, float] = (0.3, 1.0),
    fg_brightness_range: tuple[float, float] = (0.6, 1.4),
    sharpen_range: tuple[float, float] = (0.0, 0.0),
    saturation_range: tuple[float, float] = (1.0, 1.0),
    contrast_range: tuple[float, float] = (1.0, 1.0),
    grain_range: tuple[float, float] = (0.0, 0.0),
) -> RenderParams:
    """Map a single AudioFrame to rendering parameters.

    Args:
        frame: Audio features for this frame.
        dot_radius_range: (min, max) dot radius. Bass maps via sqrt curve.
        visibility_range: (min, max) visibility threshold. Treble maps inversely.
        bg_alpha_range: (min, max) background alpha scale.
        fg_brightness_range: (min, max) foreground brightness multiplier.
        sharpen_range: (min, max) sharpen amount. Bass maps via sqrt curve.
        saturation_range: (min, max) saturation. Mid-band maps linearly.
        contrast_range: (min, max) contrast. Bass maps linearly.
        grain_range: (min, max) grain strength. Inverse energy maps linearly.
    """
    r_min, r_max = dot_radius_range
    v_min, v_max = visibility_range
    ba_min, ba_max = bg_alpha_range
    fb_min, fb_max = fg_brightness_range
    sh_min, sh_max = sharpen_range
    sa_min, sa_max = saturation_range
    co_min, co_max = contrast_range
    gr_min, gr_max = grain_range

    bass = frame.bass
    treble = frame.treble
    mid = frame.mid
    energy = frame.rms

    dot_radius = r_min + (r_max - r_min) * np.sqrt(bass)
    visibility_threshold = v_max - (v_max - v_min) * treble
    bg_alpha_scale = ba_min + (ba_max - ba_min) * bass
    fg_brightness = fb_min + (fb_max - fb_min) * np.sqrt(bass)

    sharpen_amount = sh_min + (sh_max - sh_min) * np.sqrt(bass)
    saturation = sa_min + (sa_max - sa_min) * mid
    contrast = co_min + (co_max - co_min) * bass
    grain_strength = gr_min + (gr_max - gr_min) * (1.0 - energy)

    return RenderParams(
        dot_radius=float(dot_radius),
        visibility_threshold=float(visibility_threshold),
        bg_alpha_scale=float(bg_alpha_scale),
        fg_brightness=float(fg_brightness),
        sharpen_amount=float(sharpen_amount),
        saturation=float(saturation),
        contrast=float(contrast),
        grain_strength=float(grain_strength),
    )


def _apply_brightness(grid: Tensor, brightness: float) -> Tensor:
    """Apply fg_brightness modulation to grid fg RGB channels."""
    if brightness == 1.0:
        return grid
    grid = grid.clone()
    fg_rgb = (grid[8:11] + 1) / 2  # to [0, 1]
    fg_rgb = (fg_rgb * brightness).clamp(0, 1)
    grid[8:11] = fg_rgb * 2 - 1  # back to [-1, 1]
    return grid


def render_braille_frame(
    grid: Tensor,
    glyph_masks: Tensor,
    glyph_embeddings: Tensor,
    params: RenderParams,
    cell_size: tuple[int, int] = (14, 24),
) -> np.ndarray:
    """Render a single Braille frame with audio-reactive parameters.

    Applies fg_brightness to grid colors before rendering.

    Returns:
        [H, W, 4] uint8 RGBA image.
    """
    from easey_glyph.glyph.converter import render_glyph_grid_braille

    grid = _apply_brightness(grid, params.fg_brightness)

    return render_glyph_grid_braille(
        grid,
        glyph_masks,
        glyph_embeddings,
        dot_radius=params.dot_radius,
        cell_size=cell_size,
        visibility_threshold=params.visibility_threshold,
        bg_alpha_scale=params.bg_alpha_scale,
    )


def render_pixel_frame(grid: Tensor, params: RenderParams) -> np.ndarray:
    """Render grid as 32x32 RGBA pixel colors via Porter-Duff "over" compositing.

    FG is composited over BG using both alpha channels. Dark areas where both
    FG and BG are transparent produce transparent output — suitable for VJ
    layering via NDI/Syphon/Spout.

    Returns:
        [32, 32, 4] uint8 RGBA image.
    """
    import torch

    grid = _apply_brightness(grid, params.fg_brightness)
    if grid.device.type != 'cpu':
        grid = grid.cpu()
    if grid.dtype != torch.float32:
        grid = grid.float()

    fg = ((grid[8:12] + 1) / 2).clamp(0, 1)   # [4, 32, 32]
    bg = ((grid[12:16] + 1) / 2).clamp(0, 1)   # [4, 32, 32]

    # Porter-Duff "over": FG composited over BG
    fg_a = fg[3:4]  # [1, 32, 32]
    bg_a = bg[3:4]  # [1, 32, 32]
    out_a = fg_a + bg_a * (1 - fg_a)
    out_rgb = (fg[:3] * fg_a + bg[:3] * bg_a * (1 - fg_a)) / out_a.clamp(min=1e-6)

    out = torch.cat([out_rgb, out_a], dim=0)
    out = (out * 255).clamp(0, 255).byte()
    return out.permute(1, 2, 0).numpy()
