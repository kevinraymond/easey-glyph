"""GlyphSuperRes: tiny CNN for 8x super-resolution of glyph pixel renders.

Takes 32x32 RGB rendered glyph images and upscales to 256x256 using
PixelShuffle (sub-pixel convolution). Global skip connection ensures
output starts at bilinear quality and only improves from there.

Architecture: ~742K params, fully MPS-compatible, ~2-4ms per frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRResBlock(nn.Module):
    """Residual block for super-resolution. Second conv is zero-init."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.scale = 0.1

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = self.conv2(h)
        return x + h * self.scale


class PixelShuffleUp(nn.Module):
    """Conv + PixelShuffle 2x upscale + LeakyReLU."""

    def __init__(self, in_ch: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return F.leaky_relu(self.shuffle(self.conv(x)), 0.2)


class GlyphSuperRes(nn.Module):
    """8x super-resolution CNN for glyph pixel renders.

    Input:  [B, 3, 32, 32]   RGB in [0, 1]
    Output: [B, 3, 256, 256] RGB in [0, 1]
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_blocks: int = 4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.body = nn.Sequential(*[SRResBlock(base_channels) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        # 3 PixelShuffle stages: 2^3 = 8x upscale
        self.up1 = PixelShuffleUp(base_channels)
        self.up2 = PixelShuffleUp(base_channels)
        self.up3 = PixelShuffleUp(base_channels)

        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x):
        # Global skip: bilinear 8x of input
        skip = F.interpolate(x, scale_factor=8, mode="bilinear", align_corners=False)

        h = self.conv_in(x)
        body_skip = h
        h = self.body_conv(self.body(h)) + body_skip

        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)

        h = self.conv_out(h)
        return (skip + h).clamp(0, 1)


def load_superres(checkpoint_path: str, device: torch.device) -> GlyphSuperRes:
    """Load a trained GlyphSuperRes checkpoint (.pt or .safetensors)."""
    from pathlib import Path

    path = Path(checkpoint_path)

    if path.suffix == ".json":
        import json
        from safetensors.torch import load_file

        with open(path) as f:
            cfg = json.load(f).get("model", {})
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
                cfg = json.load(f).get("model", {})
        else:
            cfg = {}
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        state_dict = ckpt["model"]

    model = GlyphSuperRes(
        in_channels=cfg.get("in_channels", 3),
        base_channels=cfg.get("base_channels", 64),
        num_blocks=cfg.get("num_blocks", 4),
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"SuperRes: {params:,} params ({params / 1e6:.2f}M)")
    return model
