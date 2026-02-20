"""Render all 289 glyphs to 8x8 binary masks.

Block elements: hardcoded geometric patterns.
Braille: algorithmic from 8-bit dot pattern in a 2x4 grid, upscaled to 8x8.
Output: Tensor[289, 8, 8] float32 (0.0 or 1.0).
"""

import torch
from torch import Tensor

from .vocabulary import VOCAB_SIZE


def _rasterize_block(codepoint: int) -> Tensor:
    """Rasterize a block element (U+2580-U+259F) to 8x8 binary mask."""
    mask = torch.zeros(8, 8)
    cp = codepoint

    if cp == 0x2580:  # Upper half
        mask[:4, :] = 1
    elif 0x2581 <= cp <= 0x2588:  # Lower N/8 blocks
        n = cp - 0x2580  # 1..8
        mask[8 - n:, :] = 1
    elif 0x2589 <= cp <= 0x258F:  # Left N/8 blocks
        n = 0x2590 - cp  # 7..1
        mask[:, :n] = 1
    elif cp == 0x2590:  # Right half
        mask[:, 4:] = 1
    elif cp == 0x2591:  # Light shade (~25%)
        for i in range(8):
            for j in range(8):
                if i % 2 == 0 and j % 2 == 0:
                    mask[i, j] = 1
    elif cp == 0x2592:  # Medium shade (~50%)
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    mask[i, j] = 1
    elif cp == 0x2593:  # Dark shade (~75%)
        for i in range(8):
            for j in range(8):
                if not (i % 2 == 0 and j % 2 == 0):
                    mask[i, j] = 1
    elif cp == 0x2594:  # Upper 1/8
        mask[0, :] = 1
    elif cp == 0x2595:  # Right 1/8
        mask[:, 7] = 1
    elif cp == 0x2596:  # Quadrant lower left
        mask[4:, :4] = 1
    elif cp == 0x2597:  # Quadrant lower right
        mask[4:, 4:] = 1
    elif cp == 0x2598:  # Quadrant upper left
        mask[:4, :4] = 1
    elif cp == 0x2599:  # Quadrant UL + LL + LR (all except UR)
        mask[:4, :4] = 1
        mask[4:, :] = 1
    elif cp == 0x259A:  # Quadrant UL + LR (diagonal)
        mask[:4, :4] = 1
        mask[4:, 4:] = 1
    elif cp == 0x259B:  # Quadrant UL + UR + LL (all except LR)
        mask[:4, :] = 1
        mask[4:, :4] = 1
    elif cp == 0x259C:  # Quadrant UL + UR + LR (all except LL)
        mask[:4, :] = 1
        mask[4:, 4:] = 1
    elif cp == 0x259D:  # Quadrant upper right
        mask[:4, 4:] = 1
    elif cp == 0x259E:  # Quadrant UR + LL (anti-diagonal)
        mask[:4, 4:] = 1
        mask[4:, :4] = 1
    elif cp == 0x259F:  # Quadrant UR + LL + LR (all except UL)
        mask[:4, 4:] = 1
        mask[4:, :] = 1

    return mask


def _rasterize_braille(offset: int) -> Tensor:
    """Rasterize a braille character (offset 0-255) to 8x8 binary mask.

    Braille dot layout (2 cols x 4 rows):
        bit 0: (row=0, col=0)  bit 3: (row=0, col=1)
        bit 1: (row=1, col=0)  bit 4: (row=1, col=1)
        bit 2: (row=2, col=0)  bit 5: (row=2, col=1)
        bit 6: (row=3, col=0)  bit 7: (row=3, col=1)

    Each dot fills a 2x4 pixel block in the 8x8 grid.
    """
    mask = torch.zeros(8, 8)

    # Dot positions: (bit_index, row_in_2x4, col_in_2x4)
    dots = [
        (0, 0, 0), (1, 1, 0), (2, 2, 0),
        (3, 0, 1), (4, 1, 1), (5, 2, 1),
        (6, 3, 0), (7, 3, 1),
    ]

    for bit, row, col in dots:
        if offset & (1 << bit):
            # Each dot: 2 pixel rows x 4 pixel cols
            r0 = row * 2
            c0 = col * 4
            mask[r0:r0 + 2, c0:c0 + 4] = 1

    return mask


def build_glyph_atlas() -> Tensor:
    """Build the complete glyph atlas: [289, 8, 8] binary masks."""
    masks = torch.zeros(VOCAB_SIZE, 8, 8)

    # ID 0: space (all zeros, already done)

    # IDs 1-32: block elements U+2580-U+259F
    for i in range(32):
        cp = 0x2580 + i
        masks[1 + i] = _rasterize_block(cp)

    # IDs 33-288: braille U+2800-U+28FF
    for offset in range(256):
        masks[33 + offset] = _rasterize_braille(offset)

    return masks
