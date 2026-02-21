"""GlyphDataset: loads preprocessed .pt glyph grids into GPU memory.

The .pt file contains {"data": Tensor[N, 16, 32, 32], "glyph_ids": Tensor[N, 32, 32]}.
Only "data" is used for training.
"""

import torch
from torch import Tensor


class GlyphDataset:
    """GPU-resident glyph grid dataset."""

    def __init__(self, path: str, max_images: int | None = None):
        raw = torch.load(path, map_location="cpu", weights_only=True)
        self.data: Tensor = raw["data"].float()  # [N, 16, 32, 32]
        self.labels: Tensor | None = raw.get("labels", None)  # [N, 12] or None
        if self.labels is not None:
            self.labels = self.labels.float()
        if max_images is not None:
            self.data = self.data[:max_images]
            if self.labels is not None:
                self.labels = self.labels[:max_images]
        self._on_gpu = False

    def __len__(self) -> int:
        return self.data.shape[0]

    def to_gpu(self, device: torch.device):
        self.data = self.data.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        self._on_gpu = True

    def sample_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Random batch of glyph grids. Returns (batch, indices)."""
        idx = torch.randint(0, len(self), (batch_size,), device=self.data.device)
        batch = self.data[idx]
        # Random horizontal flip augmentation
        if torch.rand(1).item() > 0.5:
            batch = batch.flip(-1)  # flip width dimension
        return batch, idx
