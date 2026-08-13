"""Full VGGConv3D video branch: raw RGB clip -> embedding.

conv stack (with CBAM per block) -> temporal max pool -> GeM -> FC.
"""

import torch
import torch.nn as nn

from models.vggconv3d.conv_blocks import VGGConv3DBackbone
from models.vggconv3d.pooling import TemporalMaxPool, GeM


class VGGConv3D(nn.Module):
    def __init__(self, in_channels: int = 3, embedding_size: int = 128):
        super().__init__()
        self.backbone = VGGConv3DBackbone(in_channels)
        self.temporal_pool = TemporalMaxPool()
        self.gem = GeM()
        self.fc = nn.Linear(self.backbone.out_channels, embedding_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (B, C, T, H, W)
        feat = self.backbone(clip)  # (B, C', T', H', W')
        feat = self.temporal_pool(feat)  # (B, C', H', W')
        feat = self.gem(feat)  # (B, C')
        return self.fc(feat)  # (B, embedding_size)
