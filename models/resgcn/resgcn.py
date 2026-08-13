"""Full ResGCN skeleton branch: input pose sequence -> embedding.

Assembles: per-branch batch-norm + graph-conv blocks (Sec. 3.2.2) ->
concat (eq. 6) -> main-stream bottleneck blocks -> multi-head attention
(eq. 7-9) -> average pooling -> FC embedding.

The main-stream depth/channel schedule and the exact point at which
attention is inserted are not specified in the paper beyond "subsequent
layers of the main stream ... including multi-head attention and average
pooling" -- that macro ordering is followed; the specific channel widths
below are a reasonable inferred default, not read off the paper.
"""

import torch
import torch.nn as nn

from data.graph import normalized_adjacency
from models.resgcn.attention import MultiHeadAttention
from models.resgcn.blocks import BasicBlock, BottleneckBlock
from models.resgcn.branches import build_branch_inputs, concat_branches


class ResGCN(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        base_channels: int = 64,
        main_channels: int = 256,
        num_heads: int = 8,
        head_dim: int = 32,
        embedding_size: int = 128,
    ):
        super().__init__()
        norm_a = normalized_adjacency(num_joints)

        branch_in_channels = {"bone": 6, "motion": 6, "joint": 3}
        self.branch_bn = nn.ModuleDict(
            {name: nn.BatchNorm2d(c) for name, c in branch_in_channels.items()}
        )
        self.branch_blocks = nn.ModuleDict(
            {
                name: BasicBlock(c, base_channels, norm_a)
                for name, c in branch_in_channels.items()
            }
        )

        concat_channels = base_channels * 3
        self.main_stream = nn.Sequential(
            BottleneckBlock(concat_channels, main_channels, norm_a),
            BottleneckBlock(main_channels, main_channels, norm_a),
        )

        self.attention = MultiHeadAttention(main_channels, num_heads, head_dim)
        self.fc = nn.Linear(main_channels, embedding_size)

    def forward(self, pose_seq: torch.Tensor) -> torch.Tensor:
        # pose_seq: (B, T, N, 3) raw (x, y, confidence) keypoints
        branches = build_branch_inputs(pose_seq)  # each (B, T, N, C_branch)

        branch_outputs = []
        for name in ("bone", "motion", "joint"):
            feat = branches[name].permute(0, 3, 1, 2)  # (B, C, T, N)
            feat = self.branch_bn[name](feat)
            feat = self.branch_blocks[name](feat)
            branch_outputs.append(feat)

        x_concat = concat_branches(*branch_outputs)  # eq. 6: (B, 3*base, T, N)
        o_t = self.main_stream(x_concat)  # (B, main_channels, T, N)

        o_t = o_t.permute(0, 2, 3, 1)  # (B, T, N, C) for attention over joints
        attended = self.attention(o_t)  # eq. 7-9 output, (B, T, N, C)

        pooled = attended.mean(dim=(1, 2))  # average pooling over T and N
        return self.fc(pooled)  # (B, embedding_size)
