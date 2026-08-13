"""Late fusion of the ResGCN and VGGConv3D embeddings, Sec. 3.4: CBAM
channel gating per modality, then cross-modal scaled dot-product attention,
then projection to the final fused embedding.
"""

import torch
import torch.nn as nn

from models.fusion.cbam_fusion import ChannelGate
from models.fusion.scaled_dot_product_attention import ScaledDotProductAttention


class FusionModule(nn.Module):
    def __init__(self, embedding_size: int = 128, fused_size: int = 128):
        super().__init__()
        self.skeleton_gate = ChannelGate(embedding_size)
        self.video_gate = ChannelGate(embedding_size)
        self.cross_attn_skel_to_video = ScaledDotProductAttention(embedding_size)
        self.cross_attn_video_to_skel = ScaledDotProductAttention(embedding_size)
        self.project = nn.Linear(embedding_size * 2, fused_size)

    def forward(self, skeleton_emb: torch.Tensor, video_emb: torch.Tensor) -> torch.Tensor:
        skel = self.skeleton_gate(skeleton_emb)  # (B, D)
        vid = self.video_gate(video_emb)  # (B, D)

        # skeleton attends to video, and vice versa, then concat + project.
        skel_attended = self.cross_attn_skel_to_video(query_src=skel, kv_src=vid)
        vid_attended = self.cross_attn_video_to_skel(query_src=vid, kv_src=skel)

        fused = torch.cat([skel_attended, vid_attended], dim=-1)  # (B, 2D)
        return self.project(fused)  # (B, fused_size)
