"""CBAM: equations 10-12, applied sequentially (channel attention, then
spatial attention), matching Sec. 3.3.1's "sequentially inferring attention
along two dimensions: channel and spatial."

The paper's equations 10-12 only spell out an average-pooling path:
    Y_c = sigma(W2 . max(0, W1 . X + b1) + b2) . X                (10)
    Y_s = sigma(W2 . max(0, W1 . P(Y_c) + b1) + b2) . P(Y_c)      (11)
    P(Y_c) = average pool over H, W                                (12)
but the surrounding prose explicitly says both modules use "average-pooled
and max-pooled features". We implement the standard dual avg+max CBAM (the
prose version), which is a superset of the literal equations -- flagged
here and in the README as a resolved ambiguity, not a silent guess.

Input convention: (B, C, T, H, W) 5D feature maps from the 3D conv stack.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W); squeeze spatial+temporal dims (eq. 12's P(.),
        # extended here to also pool over T since CBAM in this branch sits
        # on (T, H, W) feature maps rather than plain (H, W) images).
        avg = x.mean(dim=(2, 3, 4))  # (B, C)
        mx = x.amax(dim=(2, 3, 4))  # (B, C)
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # eq. 10
        return x * gate.view(*gate.shape, 1, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W); pool over the channel axis this time.
        avg = x.mean(dim=1, keepdim=True)  # (B, 1, T, H, W)
        mx = x.amax(dim=1, keepdim=True)  # (B, 1, T, H, W)
        pooled = torch.cat([avg, mx], dim=1)  # (B, 2, T, H, W)
        gate = torch.sigmoid(self.conv(pooled))  # eq. 11, (B, 1, T, H, W)
        return x * gate


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)  # channel attention first
        x = self.spatial_attention(x)  # then spatial attention (sequential, not parallel)
        return x
