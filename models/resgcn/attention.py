"""Multi-head attention: equations 7-9.

Equation 7: O_t = concat(Y_t^1, ..., Y_t^j) over channels, the concat of M
bottleneck-block outputs (handled by concat_branches / the bottleneck stack
feeding this module — O_t is this module's input).

Equation 8 (per head p): H_p = softmax(O_t W_Qp (O_t W_Kp)^T / sqrt(D)) . O_t W_Vp
Equation 9: MH(O_t) = concat(H_1, ..., H_P) W_0
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, proj_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.proj_dim = proj_dim
        # One independent Q/K/V projection per head, as required by "each
        # head independently computes attention weights" (Sec. 3.2.3).
        self.w_q = nn.ModuleList([nn.Linear(channels, proj_dim, bias=False) for _ in range(num_heads)])
        self.w_k = nn.ModuleList([nn.Linear(channels, proj_dim, bias=False) for _ in range(num_heads)])
        self.w_v = nn.ModuleList([nn.Linear(channels, proj_dim, bias=False) for _ in range(num_heads)])
        self.w_o = nn.Linear(num_heads * proj_dim, channels, bias=False)

    def forward(self, o_t: torch.Tensor) -> torch.Tensor:
        # o_t: (B, T, N, C) -- attention is computed over the joint (N) axis
        # per the paper's W_Qp in R^{N x D} convention.
        heads = []
        scale = self.proj_dim ** 0.5
        for wq, wk, wv in zip(self.w_q, self.w_k, self.w_v):
            q = wq(o_t)  # (B, T, N, D)
            k = wk(o_t)
            v = wv(o_t)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, T, N, N)
            attn = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(attn, v))  # (B, T, N, D)
        concat = torch.cat(heads, dim=-1)  # (B, T, N, P*D)
        return self.w_o(concat)  # (B, T, N, C)
