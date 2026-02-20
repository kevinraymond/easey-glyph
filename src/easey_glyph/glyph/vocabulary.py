"""Glyph vocabulary: 289 glyphs (space + 32 block elements + 256 braille).

Each glyph has: id (0-288), codepoint, char, name.
Designed for easy expansion: append to list, bump VOCAB_SIZE.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Glyph:
    id: int
    codepoint: int
    char: str
    name: str


def _build_vocabulary() -> list[Glyph]:
    glyphs = []
    idx = 0

    # 0: Space
    glyphs.append(Glyph(idx, 0x0020, " ", "space"))
    idx += 1

    # 1-32: Block elements U+2580-U+259F
    block_names = [
        "upper half",           # 2580
        "lower 1/8",            # 2581
        "lower 1/4",            # 2582
        "lower 3/8",            # 2583
        "lower half",           # 2584
        "lower 5/8",            # 2585
        "lower 3/4",            # 2586
        "lower 7/8",            # 2587
        "full block",           # 2588
        "left 7/8",             # 2589
        "left 3/4",             # 258A
        "left 5/8",             # 258B
        "left half",            # 258C
        "left 3/8",             # 258D
        "left 1/4",             # 258E
        "left 1/8",             # 258F
        "right half",           # 2590
        "light shade",          # 2591
        "medium shade",         # 2592
        "dark shade",           # 2593
        "upper 1/8",            # 2594
        "right 1/8",            # 2595
        "quad lower left",      # 2596
        "quad lower right",     # 2597
        "quad upper left",      # 2598
        "quad UL+LL+LR",        # 2599
        "quad UL+LR",           # 259A
        "quad UL+UR+LL",        # 259B
        "quad UL+UR+LR",        # 259C
        "quad upper right",     # 259D
        "quad UR+LL",           # 259E
        "quad UR+LL+LR",        # 259F
    ]
    for i, name in enumerate(block_names):
        cp = 0x2580 + i
        glyphs.append(Glyph(idx, cp, chr(cp), name))
        idx += 1

    # 33-288: Braille U+2800-U+28FF (256 patterns)
    for offset in range(256):
        cp = 0x2800 + offset
        glyphs.append(Glyph(idx, cp, chr(cp), f"braille {offset:08b}"))
        idx += 1

    return glyphs


VOCAB: list[Glyph] = _build_vocabulary()
VOCAB_SIZE: int = len(VOCAB)  # 289

# Quick lookups
ID_TO_GLYPH: dict[int, Glyph] = {g.id: g for g in VOCAB}
CODEPOINT_TO_GLYPH: dict[int, Glyph] = {g.codepoint: g for g in VOCAB}
