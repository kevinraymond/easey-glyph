"""Convert images to/from glyph grid representation.

Image (256x256x3) -> glyph grid [16, 32, 32]:
  channels 0-7:   PCA glyph embedding (8D)
  channels 8-11:  foreground RGBA [-1, 1]
  channels 12-15: background RGBA [-1, 1]

Alpha is derived from luminance: black -> transparent, bright -> opaque.

Also provides rendering: glyph grid -> image (256x256x4 RGBA).
"""

import numpy as np
import torch
from torch import Tensor


def image_to_glyph_grid(
    img: np.ndarray,
    glyph_masks: Tensor,
    glyph_embeddings: Tensor,
) -> tuple[Tensor, Tensor]:
    """Convert a 256x256 RGB image to a glyph grid.

    Fully vectorized — processes all 1024 patches in bulk via numpy.

    Args:
        img: [256, 256, 3] uint8 numpy array.
        glyph_masks: [289, 8, 8] float32 binary masks.
        glyph_embeddings: [289, 8] float32 PCA embeddings.

    Returns:
        grid: [16, 32, 32] float32 tensor (8 PCA + 4 fg RGBA + 4 bg RGBA).
        glyph_ids: [32, 32] int64 tensor.
    """
    assert img.shape == (256, 256, 3), f"Expected 256x256x3, got {img.shape}"

    img_f = img.astype(np.float32) / 255.0  # [256, 256, 3]

    # Reshape into 8x8 patches: [32, 32, 8, 8, 3]
    patches = img_f.reshape(32, 8, 32, 8, 3).transpose(0, 2, 1, 3, 4)

    # Grayscale luminance per pixel: [32, 32, 8, 8]
    gray = patches[..., 0] * 0.299 + patches[..., 1] * 0.587 + patches[..., 2] * 0.114

    # Median per patch for binary threshold: [32, 32]
    gray_flat = gray.reshape(1024, 64)
    medians = np.median(gray_flat, axis=1).reshape(32, 32, 1, 1)

    # Binary threshold: [32, 32, 8, 8]
    binary = (gray >= medians).astype(np.float32)

    # fg/bg masks
    fg_mask = binary > 0.5   # [32, 32, 8, 8]
    bg_mask = ~fg_mask

    # Counts per patch: [32, 32]
    fg_count = fg_mask.sum(axis=(2, 3))
    bg_count = bg_mask.sum(axis=(2, 3))

    # Weighted sums for fg/bg RGB: [32, 32, 3]
    fg_sum = (patches * fg_mask[..., None]).sum(axis=(2, 3))
    bg_sum = (patches * bg_mask[..., None]).sum(axis=(2, 3))

    # Safe divide (avoid /0)
    fg_count_safe = np.maximum(fg_count, 1)[..., None]
    bg_count_safe = np.maximum(bg_count, 1)[..., None]
    fg_rgb = fg_sum / fg_count_safe  # [32, 32, 3]
    bg_rgb = bg_sum / bg_count_safe

    # Fallback: where count is 0, use mean of whole patch
    patch_mean = patches.mean(axis=(2, 3))  # [32, 32, 3]
    fg_zero = (fg_count == 0)[..., None]
    bg_zero = (bg_count == 0)[..., None]
    fg_rgb = np.where(fg_zero, patch_mean, fg_rgb)
    bg_rgb = np.where(bg_zero, patch_mean, bg_rgb)

    # Mask matching: MSE between each patch binary and all glyph masks
    # binary_all: [1024, 64], masks_flat: [289, 64]
    binary_all = binary.reshape(1024, 64)
    masks_flat = glyph_masks.numpy().reshape(glyph_masks.shape[0], -1)

    # MSE = mean((a-b)^2) = mean(a^2) + mean(b^2) - 2*mean(a*b)
    a_sq = (binary_all ** 2).sum(axis=1)       # [1024]
    b_sq = (masks_flat ** 2).sum(axis=1)       # [289]
    ab = binary_all @ masks_flat.T             # [1024, 289]
    mse = (a_sq[:, None] + b_sq[None, :] - 2.0 * ab) / 64.0
    best_ids = np.argmin(mse, axis=1).reshape(32, 32)  # [32, 32]

    # Alpha from luminance: fg/bg mean luminance per patch
    fg_lum_sum = (gray * fg_mask).sum(axis=(2, 3))
    bg_lum_sum = (gray * bg_mask).sum(axis=(2, 3))
    gray_mean = gray.mean(axis=(2, 3))  # [32, 32]

    fg_lum = np.where(fg_count > 0, fg_lum_sum / np.maximum(fg_count, 1), gray_mean)
    bg_lum = np.where(bg_count > 0, bg_lum_sum / np.maximum(bg_count, 1), gray_mean)
    fg_alpha = np.minimum(np.sqrt(fg_lum) * 1.5, 1.0)  # [32, 32]
    bg_alpha = np.minimum(np.sqrt(bg_lum) * 1.5, 1.0)

    # Assemble grid [16, 32, 32]
    grid = np.zeros((16, 32, 32), dtype=np.float32)
    emb_np = glyph_embeddings.numpy()  # [289, 8]
    grid[:8] = emb_np[best_ids].transpose(2, 0, 1)        # [8, 32, 32]
    grid[8:11] = (fg_rgb * 2 - 1).transpose(2, 0, 1)      # [3, 32, 32]
    grid[11] = fg_alpha * 2 - 1
    grid[12:15] = (bg_rgb * 2 - 1).transpose(2, 0, 1)
    grid[15] = bg_alpha * 2 - 1

    glyph_ids = best_ids.astype(np.int64)
    return torch.from_numpy(grid), torch.from_numpy(glyph_ids)


def render_glyph_grid(
    grid: Tensor,
    glyph_masks: Tensor,
    glyph_embeddings: Tensor,
) -> np.ndarray:
    """Render a glyph grid to a 256x256 RGBA image.

    Args:
        grid: [16, 32, 32] float32 tensor.
        glyph_masks: [289, 8, 8] float32 binary masks.
        glyph_embeddings: [289, 8] float32 PCA embeddings.

    Returns:
        [256, 256, 4] uint8 numpy array (RGBA).
    """
    grid = grid.cpu().float()
    glyph_masks = glyph_masks.cpu()
    glyph_embeddings = glyph_embeddings.cpu()

    # Decode glyph IDs from embeddings
    emb = grid[0:8].reshape(8, -1).T  # [1024, 8]
    dists = torch.cdist(emb, glyph_embeddings)  # [1024, 289]
    ids = dists.argmin(dim=1).reshape(32, 32)  # [32, 32]

    # Extract colors (RGBA)
    fg = grid[8:12]    # [4, 32, 32] in [-1, 1]
    bg = grid[12:16]   # [4, 32, 32] in [-1, 1]

    # Denormalize to [0, 1]
    fg = ((fg + 1) / 2).clamp(0, 1)
    bg = ((bg + 1) / 2).clamp(0, 1)

    # Render
    img = np.zeros((256, 256, 4), dtype=np.float32)

    for row in range(32):
        for col in range(32):
            mask = glyph_masks[ids[row, col]].numpy()  # [8, 8]
            f = fg[:, row, col].numpy()  # [4] RGBA
            b = bg[:, row, col].numpy()  # [4] RGBA

            r0, c0 = row * 8, col * 8
            for c in range(4):
                img[r0:r0 + 8, c0:c0 + 8, c] = f[c] * mask + b[c] * (1 - mask)

    return (img * 255).clip(0, 255).astype(np.uint8)


def render_glyph_grid_ansi(
    grid: Tensor,
    glyph_masks: Tensor,
    glyph_embeddings: Tensor,
) -> str:
    """Render a glyph grid to an ANSI 24-bit color terminal string.

    Args:
        grid: [16, 32, 32] float32 tensor.
        glyph_masks: [289, 8, 8] float32 binary masks.
        glyph_embeddings: [289, 8] float32 PCA embeddings.

    Returns:
        ANSI-colored string for terminal display (alpha ignored).
    """
    from .vocabulary import ID_TO_GLYPH

    grid = grid.cpu().float()
    glyph_embeddings = glyph_embeddings.cpu()

    # Decode glyph IDs
    emb = grid[0:8].reshape(8, -1).T  # [1024, 8]
    dists = torch.cdist(emb, glyph_embeddings)  # [1024, 289]
    ids = dists.argmin(dim=1).reshape(32, 32)

    # Extract RGB only (skip alpha, terminal can't use it)
    fg = ((grid[8:11] + 1) / 2).clamp(0, 1)   # [3, 32, 32]
    bg = ((grid[12:15] + 1) / 2).clamp(0, 1)   # [3, 32, 32]

    lines = []
    for row in range(32):
        parts = []
        for col in range(32):
            glyph = ID_TO_GLYPH[ids[row, col].item()]
            fr, fg_, fb = [int(fg[c, row, col].item() * 255) for c in range(3)]
            br, bg_, bb = [int(bg[c, row, col].item() * 255) for c in range(3)]
            parts.append(f"\033[38;2;{fr};{fg_};{fb}m\033[48;2;{br};{bg_};{bb}m{glyph.char}")
        lines.append("".join(parts) + "\033[0m")

    return "\n".join(lines)


def _braille_dots(char: str) -> list[tuple[int, int]]:
    """Decode a Braille character to its active dot positions.

    Braille U+2800-U+28FF encodes 8 dots in a 2x4 grid:
        col 0  col 1
        dot1   dot4    row 0
        dot2   dot5    row 1
        dot3   dot6    row 2
        dot7   dot8    row 3

    Returns list of (col, row) tuples for active dots.
    """
    code = ord(char) - 0x2800
    if code < 0 or code > 0xFF:
        return []
    # Bit positions: 0=dot1, 1=dot2, 2=dot3, 3=dot4, 4=dot5, 5=dot6, 6=dot7, 7=dot8
    dot_positions = [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3),
    ]
    return [pos for bit, pos in enumerate(dot_positions) if code & (1 << bit)]


def render_glyph_grid_braille(
    grid: Tensor,
    glyph_masks: Tensor,
    glyph_embeddings: Tensor,
    dot_radius: float = 2,
    cell_size: tuple[int, int] = (14, 24),
    visibility_threshold: float = 0.0,
    bg_alpha_scale: float = 1.0,
) -> np.ndarray:
    """Render a glyph grid as a Braille dot-pattern image (no font needed).

    Draws actual dots in a 2x4 grid per cell, matching how Braille characters
    look in a terminal but rendered reliably without font dependencies.

    Args:
        grid: [16, 32, 32] float32 tensor.
        glyph_masks: [289, 8, 8] float32 binary masks.
        glyph_embeddings: [289, 8] float32 PCA embeddings.
        dot_radius: Radius of each Braille dot in pixels.
        cell_size: (width, height) of each character cell in pixels.
        visibility_threshold: Minimum fg alpha (0-1) to draw a dot. 0.0 draws all.
        bg_alpha_scale: Multiplier on background alpha. 1.0 = no change.

    Returns:
        [H, W, 4] uint8 numpy array (RGBA).
    """
    from PIL import Image, ImageDraw
    from .vocabulary import ID_TO_GLYPH

    grid = grid.cpu().float()
    glyph_embeddings = glyph_embeddings.cpu()

    # Decode glyph IDs
    emb = grid[0:8].reshape(8, -1).T
    dists = torch.cdist(emb, glyph_embeddings)
    ids = dists.argmin(dim=1).reshape(32, 32)

    # Extract RGBA
    fg = ((grid[8:12] + 1) / 2).clamp(0, 1)   # [4, 32, 32]
    bg = ((grid[12:16] + 1) / 2).clamp(0, 1)   # [4, 32, 32]

    cell_w, cell_h = cell_size
    img_w = 32 * cell_w
    img_h = 32 * cell_h
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dot grid spacing within a cell (2 cols x 4 rows)
    pad_x = dot_radius + 1
    pad_y = dot_radius + 1
    spacing_x = cell_w - 2 * pad_x
    spacing_y = (cell_h - 2 * pad_y) / 3  # 4 rows -> 3 gaps

    for row in range(32):
        for col in range(32):
            glyph = ID_TO_GLYPH[ids[row, col].item()]
            fa_f = fg[3, row, col].item()
            fr, fg_, fb, fa = [int(fg[c, row, col].item() * 255) for c in range(4)]
            ba_f = min(bg[3, row, col].item() * bg_alpha_scale, 1.0)
            br, bg_, bb = [int(bg[c, row, col].item() * 255) for c in range(3)]
            ba = int(ba_f * 255)

            x0 = col * cell_w
            y0 = row * cell_h

            # Draw background rect
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], fill=(br, bg_, bb, ba))

            # Draw Braille dots (skip if fg alpha below threshold)
            if fa_f < visibility_threshold:
                continue
            dots = _braille_dots(glyph.char)
            for dc, dr in dots:
                cx = x0 + pad_x + dc * spacing_x
                cy = y0 + pad_y + dr * spacing_y
                draw.ellipse(
                    [cx - dot_radius, cy - dot_radius,
                     cx + dot_radius, cy + dot_radius],
                    fill=(fr, fg_, fb, fa),
                )

    return np.array(img)
