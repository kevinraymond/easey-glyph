"""Timestep embeddings for flow matching."""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimestepEmbedding(nn.Module):
    """Scalar t in [0,1] -> [B, emb_dim] via standard DDPM sin/cos encoding."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.emb_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class TimestepMLP(nn.Module):
    """SinEmbed -> Linear -> SiLU -> Linear -> [B, out_dim]."""

    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sinusoidal(t))
