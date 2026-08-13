"""Basic and bottleneck graph-conv blocks used inside each branch and the
main stream (Sec. 3.2.2 references "basic and bottleneck blocks" without
giving their internals, so these follow the conventional ST-GCN-style
residual block shape: this structure is inferred, the graph-conv math
inside each block (equation 1) is not).
"""

import torch
import torch.nn as nn

from models.resgcn.graph_conv import GraphConv


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_adjacency: torch.Tensor):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, norm_adjacency)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gcn(x) + self.residual(x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_adjacency: torch.Tensor, reduction: int = 4):
        super().__init__()
        mid = max(out_channels // reduction, 1)
        self.reduce = nn.Conv2d(in_channels, mid, kernel_size=1)
        self.gcn = GraphConv(mid, mid, norm_adjacency)
        self.expand = nn.Conv2d(mid, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.gcn(out)
        out = self.expand(out)
        out = self.bn(out)
        return self.activation(out + self.residual(x))
