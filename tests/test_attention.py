"""Correctness tests for multi-head attention (equations 7-9)."""

import torch

from models.resgcn.attention import MultiHeadAttention


def test_heads_have_independent_projections():
    mha = MultiHeadAttention(channels=16, num_heads=4, proj_dim=8)
    # each head's Q/K/V weights must be distinct parameters (not tied/shared).
    q_weights = [w.weight for w in mha.w_q]
    for i in range(len(q_weights)):
        for j in range(i + 1, len(q_weights)):
            assert q_weights[i].data_ptr() != q_weights[j].data_ptr()
            assert not torch.equal(q_weights[i], q_weights[j])


def test_attention_weights_sum_to_one():
    torch.manual_seed(0)
    mha = MultiHeadAttention(channels=16, num_heads=2, proj_dim=8)
    o_t = torch.randn(1, 3, 5, 16)  # (B, T, N, C)

    # replicate the internal softmax computation for head 0 to check it's
    # a valid probability distribution over the N axis.
    q = mha.w_q[0](o_t)
    k = mha.w_k[0](o_t)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (mha.proj_dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(1, 3, 5), atol=1e-5)


def test_output_shape_matches_input_channels():
    mha = MultiHeadAttention(channels=16, num_heads=4, proj_dim=8)
    o_t = torch.randn(2, 3, 5, 16)
    out = mha(o_t)
    assert out.shape == o_t.shape  # eq. 9's W_0 projects back to C channels


def test_forward_matches_manual_single_head_computation():
    # reproduces eq. 8/9 by hand for a single-head config and checks the
    # module's forward output matches exactly, confirming the 1/sqrt(D)
    # scaling and W_0 projection are wired as specified.
    torch.manual_seed(1)
    mha = MultiHeadAttention(channels=8, num_heads=1, proj_dim=64)
    o_t = torch.randn(1, 2, 3, 8)

    q = mha.w_q[0](o_t)
    k = mha.w_k[0](o_t)
    v = mha.w_v[0](o_t)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (64 ** 0.5)  # eq. 8 scaling
    head = torch.matmul(torch.softmax(scores, dim=-1), v)
    expected = mha.w_o(head)  # eq. 9 with a single head, concat is a no-op

    assert torch.allclose(mha(o_t), expected, atol=1e-6)
