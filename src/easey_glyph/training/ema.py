"""Exponential Moving Average for model parameters."""

import torch


@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    """Update EMA parameters: p_ema = decay * p_ema + (1-decay) * p."""
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.lerp_(p.data, 1 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b.data)
