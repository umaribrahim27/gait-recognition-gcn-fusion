"""Temporal max pooling and GeM (Generalised Mean) pooling, Sec. 3.3.2.

Neither is given as an explicit equation in the paper (they're described
only in prose), so the exact formulas below (GeM in particular) follow the
standard published definitions rather than anything spelled out in this
paper's text.
"""

import torch
import torch.nn as nn


class TemporalMaxPool(nn.Module):
    """Collapses the T dimension of a (B, C, T, H, W) tensor via max."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.amax(dim=2)  # (B, C, H, W)


class GeM(nn.Module):
    """Generalised Mean pooling over spatial dims (H, W) -> (B, C)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        init_p = torch.tensor(float(p))
        self.p = nn.Parameter(init_p) if learn_p else init_p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        clamped = x.clamp(min=self.eps)
        pooled = clamped.pow(self.p).mean(dim=(2, 3))
        return pooled.pow(1.0 / self.p)
