"""Top-level model combining the ResGCN and VGGConv3D branches via the
fusion module (Fig. 1 overview)."""

import torch
import torch.nn as nn

from models.resgcn.resgcn import ResGCN
from models.vggconv3d.vggconv3d import VGGConv3D
from models.fusion.fusion_module import FusionModule


class GaitNet(nn.Module):
    def __init__(self, embedding_size: int = 128, fused_size: int = 128):
        super().__init__()
        self.resgcn = ResGCN(embedding_size=embedding_size)
        self.vggconv3d = VGGConv3D(embedding_size=embedding_size)
        self.fusion = FusionModule(embedding_size, fused_size)

    def forward(self, pose_seq: torch.Tensor, video_clip: torch.Tensor) -> torch.Tensor:
        skeleton_emb = self.resgcn(pose_seq)  # (B, embedding_size)
        video_emb = self.vggconv3d(video_clip)  # (B, embedding_size)
        fused = self.fusion(skeleton_emb, video_emb)  # (B, fused_size)
        return nn.functional.normalize(fused, dim=-1)
