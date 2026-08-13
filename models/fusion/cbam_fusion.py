"""CBAM-style channel gating applied to the pooled per-modality embeddings
at the fusion stage (Sec. 3.4: "CBAM is employed... by dynamically
emphasising relevant spatial regions... channel-wise attention selectively
highlights informative channels").

By the time features reach the fusion module in our pipeline they are
already pooled to (B, D) embeddings (no spatial H/W axis remains), so only
a channel-attention gate is meaningful here -- the spatial-attention half
of CBAM (eq. 11) is not reapplied at this stage since it operates on
spatial feature maps, which fusion, as implemented, does not receive. This
is an inferred simplification: the paper's block diagram (Fig. 1) suggests
CBAM sits between the two branches before their final pooling, but does
not give the exact feature-map shapes at that junction.
"""

import torch
import torch.nn as nn


class ChannelGate(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -- "avg" and "max" pooling collapse to identity/no-op
        # over a single-vector-per-sample input, so both paths use x itself.
        gate = torch.sigmoid(self.mlp(x))
        return x * gate
