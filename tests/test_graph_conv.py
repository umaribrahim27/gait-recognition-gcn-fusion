"""Correctness tests for the normalised adjacency matrix (equation 1) and
the GraphConv layer, beyond shape checks: verifies actual numerical
properties the equation requires.
"""

import torch

from data.graph import BONE_EDGES, build_adjacency, normalized_adjacency
from models.resgcn.graph_conv import GraphConv


def test_adjacency_is_symmetric_and_tree_edge_count():
    a = build_adjacency()
    assert torch.equal(a, a.T)
    # a nose-rooted tree over 17 joints has exactly 16 edges (32 nonzero
    # entries, symmetric).
    assert len(BONE_EDGES) == 16
    assert a.sum().item() == 32


def test_normalized_adjacency_self_loops_included():
    norm_a = normalized_adjacency()
    # every node has a nonzero self-connection after A~ = A + I.
    assert (norm_a.diagonal() > 0).all()


def test_normalized_adjacency_matches_manual_formula():
    a = build_adjacency()
    a_tilde = a + torch.eye(a.shape[0])
    deg = a_tilde.sum(dim=1)
    d_inv_sqrt = torch.diag(deg.pow(-0.5))
    expected = d_inv_sqrt @ a_tilde @ d_inv_sqrt
    assert torch.allclose(normalized_adjacency(), expected, atol=1e-6)


def test_graph_conv_output_shape_and_finite():
    norm_a = normalized_adjacency()
    layer = GraphConv(in_channels=3, out_channels=8, norm_adjacency=norm_a)
    x = torch.randn(2, 3, 10, 17)  # (B, C, T, N)
    out = layer(x)
    assert out.shape == (2, 8, 10, 17)
    assert torch.isfinite(out).all()


def test_graph_conv_activation_is_nonnegative():
    # the layer ends in ReLU per eq. 1's sigma(.).
    norm_a = normalized_adjacency()
    layer = GraphConv(in_channels=3, out_channels=8, norm_adjacency=norm_a)
    x = torch.randn(2, 3, 10, 17)
    out = layer(x)
    assert (out >= 0).all()
