"""Self-attention for UNet feature maps.

Standard multi-head QKV attention, only used at specified resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfAttention2d(nn.Module):
    """Multi-head self-attention on 2D feature maps."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        num_groups = 32
        while channels % num_groups != 0:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Reshape to [B, heads, HW, dim] for fused SDPA (FlashAttention on CUDA)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.proj(out)
        return x + out
