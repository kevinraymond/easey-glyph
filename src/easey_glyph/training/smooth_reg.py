"""Smooth Diffusion regularization (CVPR 2024).

Penalizes large output changes for small input perturbations.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def smooth_regularization(
    model: torch.nn.Module,
    t: Tensor,
    xt: Tensor,
    audio: Tensor | None,
    eps_scale: float = 0.01,
) -> Tensor:
    with torch.no_grad():
        v_base = model(t, xt, audio).clone()

    eps = torch.randn_like(xt) * eps_scale
    xt_perturbed = xt + eps

    v_perturbed = model(t, xt_perturbed, audio)

    delta_v = v_perturbed - v_base
    delta_v_norm_sq = delta_v.pow(2).sum(dim=(1, 2, 3))
    eps_norm_sq = eps.pow(2).sum(dim=(1, 2, 3))

    loss = (delta_v_norm_sq / (eps_norm_sq + 1e-8)).mean()
    return loss
