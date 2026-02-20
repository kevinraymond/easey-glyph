"""FiLM (Feature-wise Linear Modulation) for audio conditioning.

AudioFiLMEncoder maps [B, 12] audio features to per-level (gamma, beta) pairs.
FiLMLayer applies: output = (1 + gamma) * GroupNorm(x) + beta
"""

import torch
import torch.nn as nn
from torch import Tensor


class AudioFiLMEncoder(nn.Module):
    """[B, 12] audio -> shared trunk MLP -> per-level (gamma, beta) heads."""

    def __init__(self, audio_dim: int, hidden_dim: int, level_channels: list[int]):
        super().__init__()
        self.level_channels = level_channels

        self.feature_projs = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(audio_dim)
        ])

        self.trunk_mix = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.heads = nn.ModuleList()
        for ch in level_channels:
            head = nn.Linear(hidden_dim, ch * 2)
            self.heads.append(head)

    def forward(self, audio: Tensor | None) -> list[tuple[Tensor, Tensor]]:
        if audio is None:
            return None

        h = torch.zeros(audio.shape[0], self.feature_projs[0].out_features, device=audio.device)
        for i, proj in enumerate(self.feature_projs):
            h = h + proj(audio[:, i:i+1])
        h = self.trunk_mix(h)
        result = []
        for head, ch in zip(self.heads, self.level_channels):
            gb = head(h)
            gamma, beta = gb.chunk(2, dim=-1)
            result.append((gamma, beta))
        return result


class FiLMLayer(nn.Module):
    """Apply FiLM modulation: output = (1 + gamma) * GroupNorm(x) + beta."""

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        while channels % num_groups != 0:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups, channels)

    def forward(self, x: Tensor, gamma: Tensor | None = None, beta: Tensor | None = None) -> Tensor:
        h = self.norm(x)
        if gamma is not None:
            h = h * (1 + gamma[:, :, None, None])
        if beta is not None:
            h = h + beta[:, :, None, None]
        return h
