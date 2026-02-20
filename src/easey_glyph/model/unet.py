"""FlowUNet: UNet for flow matching with FiLM audio conditioning.

Adapted from easey-flow with configurable in_channels/out_channels
for glyph grid tensors [B, 16, 32, 32].

Architecture:
  - Encoder: DownBlocks with optional attention at specified resolutions
  - Mid: ResBlock + Attention + ResBlock at smallest resolution
  - Decoder: UpBlocks with skip connections and optional attention
  - FiLM modulation at every ResBlock (identity when audio=None)
  - Zero-init final conv (model starts predicting zero velocity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import SelfAttention2d
from .embeddings import TimestepMLP
from .film import AudioFiLMEncoder, FiLMLayer


class ResBlock(nn.Module):
    """Residual block with time embedding injection and FiLM modulation."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, use_depthwise_separable: bool = False):
        super().__init__()

        def _gn(ch):
            ng = 32
            while ch % ng != 0:
                ng //= 2
            return nn.GroupNorm(ng, ch)

        self.norm1 = _gn(in_ch)
        self.conv1 = self._make_conv(in_ch, out_ch, use_depthwise_separable and in_ch >= 128)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.film = FiLMLayer(out_ch)
        self.conv2 = self._make_conv(out_ch, out_ch, use_depthwise_separable and out_ch >= 128)
        nn.init.zeros_(self.conv2[-1].weight if isinstance(self.conv2, nn.Sequential) else self.conv2.weight)
        nn.init.zeros_(self.conv2[-1].bias if isinstance(self.conv2, nn.Sequential) else self.conv2.bias)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    @staticmethod
    def _make_conv(in_ch: int, out_ch: int, depthwise_separable: bool) -> nn.Module:
        if depthwise_separable:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, 1),
            )
        return nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(
        self, x: Tensor, t_emb: Tensor,
        gamma: Tensor | None = None, beta: Tensor | None = None,
    ) -> Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.film(h, gamma, beta)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    """Encoder block: ResBlock(s) [+ Attention] + stride-2 downsample."""

    def __init__(
        self, in_ch: int, out_ch: int, time_dim: int,
        num_res_blocks: int = 1, use_attention: bool = False,
        use_depthwise_separable: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            ch_in = in_ch if i == 0 else out_ch
            self.res_blocks.append(ResBlock(ch_in, out_ch, time_dim, use_depthwise_separable))
            self.attn_blocks.append(SelfAttention2d(out_ch) if use_attention else nn.Identity())

        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(
        self, x: Tensor, t_emb: Tensor,
        gamma: Tensor | None = None, beta: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        skips = []
        h = x
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            h = res(h, t_emb, gamma, beta)
            h = attn(h)
            skips.append(h)
        h = self.downsample(h)
        return h, skips


class UpBlock(nn.Module):
    """Decoder block: bilinear upsample + concat(skip) + ResBlock(s) [+ Attention]."""

    def __init__(
        self, in_ch: int, out_ch: int, skip_ch: int, time_dim: int,
        num_res_blocks: int = 1, use_attention: bool = False,
        use_depthwise_separable: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            ch_in = (in_ch + skip_ch) if i == 0 else out_ch
            self.res_blocks.append(ResBlock(ch_in, out_ch, time_dim, use_depthwise_separable))
            self.attn_blocks.append(SelfAttention2d(out_ch) if use_attention else nn.Identity())

    def forward(
        self, x: Tensor, skips: list[Tensor], t_emb: Tensor,
        gamma: Tensor | None = None, beta: Tensor | None = None,
    ) -> Tensor:
        h = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        skip = skips.pop()
        h = torch.cat([h, skip], dim=1)

        for i, (res, attn) in enumerate(zip(self.res_blocks, self.attn_blocks)):
            h = res(h, t_emb, gamma, beta)
            h = attn(h)
        return h


class MidBlock(nn.Module):
    """Middle block: ResBlock + Attention + ResBlock at smallest resolution."""

    def __init__(self, channels: int, time_dim: int, use_depthwise_separable: bool = False):
        super().__init__()
        self.res1 = ResBlock(channels, channels, time_dim, use_depthwise_separable)
        self.attn = SelfAttention2d(channels)
        self.res2 = ResBlock(channels, channels, time_dim, use_depthwise_separable)

    def forward(
        self, x: Tensor, t_emb: Tensor,
        gamma: Tensor | None = None, beta: Tensor | None = None,
    ) -> Tensor:
        h = self.res1(x, t_emb, gamma, beta)
        h = self.attn(h)
        h = self.res2(h, t_emb, gamma, beta)
        return h


class FlowUNet(nn.Module):
    """Full encoder-decoder UNet for flow matching.

    Args:
        image_size: Input spatial resolution.
        in_channels: Input tensor channels (3 for RGB, 16 for glyph grids).
        out_channels: Output tensor channels (3 for RGB, 16 for glyph grids).
        base_channels: Base channel count.
        channel_mult: Channel multipliers per level.
        num_res_blocks: Residual blocks per level.
        attention_resolutions: Resolutions at which to add attention.
        use_depthwise_separable: Use depthwise separable convs for ch >= 128.
        audio_dim: Audio conditioning dimension (12).
        time_emb_dim: Timestep embedding dimension.
    """

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 16,
        out_channels: int = 16,
        base_channels: int = 64,
        channel_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attention_resolutions: list[int] | None = None,
        use_depthwise_separable: bool = False,
        audio_dim: int = 12,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        if channel_mult is None:
            channel_mult = [1, 2, 4, 4]
        if attention_resolutions is None:
            attention_resolutions = [4]

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        channels = [base_channels * m for m in channel_mult]

        # Timestep embedding
        self.time_mlp = TimestepMLP(time_emb_dim, time_emb_dim)

        # Compute decoder output channels (reverse order)
        dec_out_channels = []
        for i in range(len(channels) - 1, -1, -1):
            dec_out_channels.append(channels[i - 1] if i > 0 else channels[0])

        # Audio FiLM encoder: encoder levels + mid + decoder levels
        all_level_channels = channels + [channels[-1]] + dec_out_channels
        self.film_encoder = AudioFiLMEncoder(audio_dim, time_emb_dim, all_level_channels)
        self._num_enc_levels = len(channels)
        self._mid_film_idx = len(channels)
        self._dec_film_offset = len(channels) + 1

        # Input conv: in_channels -> base_channels
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        current_res = image_size
        for i in range(len(channels)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            next_res = current_res // 2
            use_attn = next_res in attention_resolutions
            self.down_blocks.append(DownBlock(
                in_ch, out_ch, time_emb_dim,
                num_res_blocks=num_res_blocks,
                use_attention=use_attn,
                use_depthwise_separable=use_depthwise_separable,
            ))
            current_res = next_res

        # Mid block at smallest resolution
        self.mid = MidBlock(channels[-1], time_emb_dim, use_depthwise_separable)

        # Decoder (reverse order)
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            in_ch = channels[i]
            out_ch = channels[i - 1] if i > 0 else channels[0]
            skip_ch = channels[i]
            use_attn = current_res in attention_resolutions
            self.up_blocks.append(UpBlock(
                in_ch, out_ch, skip_ch, time_emb_dim,
                num_res_blocks=num_res_blocks,
                use_attention=use_attn,
                use_depthwise_separable=use_depthwise_separable,
            ))
            current_res *= 2

        # Output conv: base_channels -> out_channels
        out_ng = 32
        while channels[0] % out_ng != 0:
            out_ng //= 2
        self.output_norm = nn.GroupNorm(out_ng, channels[0])
        self.output_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, t: Tensor, x: Tensor, audio: Tensor | None = None) -> Tensor:
        """
        Args:
            t: [B] scalar timesteps in [0, 1]
            x: [B, in_channels, H, W] input tensor
            audio: [B, 12] audio features or None

        Returns:
            [B, out_channels, H, W] predicted velocity field
        """
        t_emb = self.time_mlp(t)

        film_params = self.film_encoder(audio)

        def get_film(level_idx):
            if film_params is None:
                return None, None
            return film_params[level_idx]

        h = self.input_conv(x)

        all_skips = []
        for i, down in enumerate(self.down_blocks):
            gamma, beta = get_film(i)
            h, skips = down(h, t_emb, gamma, beta)
            all_skips.append(skips)

        mid_gamma, mid_beta = get_film(self._mid_film_idx)
        h = self.mid(h, t_emb, mid_gamma, mid_beta)

        for dec_i, up in enumerate(self.up_blocks):
            skips = all_skips.pop()
            gamma, beta = get_film(self._dec_film_offset + dec_i)
            h = up(h, skips, t_emb, gamma, beta)

        h = F.silu(self.output_norm(h))
        return self.output_conv(h)
