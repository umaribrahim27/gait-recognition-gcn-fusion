"""Scaled Dot-Product Attention used in the fusion pipeline (Sec. 3.4).

The paper describes this only in prose ("the extracted features are then
passed to the scaled dot-product attention module... ensures attention
weights are appropriately scaled") without a dedicated equation number, so
this follows the standard formula softmax(QK^T / sqrt(d_k)) V, same
scaling convention as equation 8's multi-head attention.
"""

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.dim = dim

    def forward(self, query_src: torch.Tensor, kv_src: torch.Tensor) -> torch.Tensor:
        # query_src, kv_src: (B, D) single-embedding-per-sample vectors
        # (no sequence axis at this stage), treated as length-1 sequences.
        q = self.w_q(query_src).unsqueeze(1)  # (B, 1, D)
        k = self.w_k(kv_src).unsqueeze(1)  # (B, 1, D)
        v = self.w_v(kv_src).unsqueeze(1)  # (B, 1, D)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)  # (B,1,1)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, 1, D)
        return out.squeeze(1)  # (B, D)
