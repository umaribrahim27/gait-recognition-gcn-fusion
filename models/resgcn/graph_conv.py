"""Graph convolution layer implementing equation 1 of the paper.

    X_t^(l+1) = sigma( D~^{-1/2} A~ D~^{-1/2} X_t^(l) Theta^(l) )

with A~ = A + I (self-loops added) and D~ the degree matrix of A~.
Feature layout used throughout this module: (B, C, T, N).
"""

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_adjacency: torch.Tensor):
        super().__init__()
        # norm_adjacency = D~^{-1/2} A~ D~^{-1/2}, precomputed, shape (N, N)
        self.register_buffer("norm_adjacency", norm_adjacency)
        # Theta^(l): the learnt per-node linear transform, implemented as a 1x1 conv
        # over the channel dimension (applied identically at every (t, n)).
        self.theta = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N)
        # Spatial aggregation from direct neighbours: (D~^{-1/2} A~ D~^{-1/2}) X_t
        x = torch.einsum("vw,bctw->bctv", self.norm_adjacency, x)
        # Learnt transform Theta^(l)
        x = self.theta(x)
        x = self.bn(x)
        return self.activation(x)
