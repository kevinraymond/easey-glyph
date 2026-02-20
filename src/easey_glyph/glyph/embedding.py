"""PCA glyph embeddings: 289 binary masks -> 8D continuous vectors.

Flatten [289, 8, 8] -> [289, 64], PCA to 8D, normalize to ~[-1, 1].
Provides encode (glyph_id -> embedding) and decode (embedding -> glyph_id).
"""

import torch
import numpy as np
from torch import Tensor
from sklearn.decomposition import PCA

from .rasterizer import build_glyph_atlas
from .vocabulary import VOCAB_SIZE

EMBED_DIM = 8


def compute_embeddings() -> Tensor:
    """Compute PCA embeddings for all 289 glyphs.

    Returns:
        [289, 8] float32 tensor, values approximately in [-1, 1].
    """
    masks = build_glyph_atlas()  # [289, 8, 8]
    flat = masks.reshape(VOCAB_SIZE, 64).numpy()  # [289, 64]

    pca = PCA(n_components=EMBED_DIM)
    embedded = pca.fit_transform(flat)  # [289, 8]

    # Normalize to approximately [-1, 1]
    max_abs = np.abs(embedded).max(axis=0, keepdims=True)
    max_abs = np.maximum(max_abs, 1e-8)
    embedded = embedded / max_abs

    return torch.from_numpy(embedded).float()


def decode_nearest(embeddings: Tensor, table: Tensor) -> Tensor:
    """Find nearest glyph ID for each embedding vector.

    Args:
        embeddings: [N, 8] query embeddings.
        table: [289, 8] glyph embedding table.

    Returns:
        [N] int64 glyph IDs.
    """
    # cdist + argmin
    dists = torch.cdist(embeddings.float(), table.float())  # [N, 289]
    return dists.argmin(dim=1)
