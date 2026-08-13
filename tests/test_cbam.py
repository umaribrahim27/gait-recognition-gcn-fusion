"""Correctness tests for CBAM (equations 10-12): sequential ordering and
gate value ranges.
"""

import torch

from models.vggconv3d.cbam import CBAM, ChannelAttention, SpatialAttention


def test_channel_attention_preserves_shape_and_gates_in_unit_interval():
    ca = ChannelAttention(channels=8)
    x = torch.randn(2, 8, 4, 5, 5)
    out = ca(x)
    assert out.shape == x.shape

    # recompute the sigmoid gate directly (eq. 10) and check it's a valid
    # gate in [0, 1], and that out == x * gate exactly.
    avg = x.mean(dim=(2, 3, 4))
    mx = x.amax(dim=(2, 3, 4))
    gate = torch.sigmoid(ca.mlp(avg) + ca.mlp(mx))
    assert (gate >= 0).all() and (gate <= 1).all()
    assert torch.allclose(out, x * gate.view(*gate.shape, 1, 1, 1), atol=1e-6)


def test_spatial_attention_preserves_shape():
    sa = SpatialAttention()
    x = torch.randn(2, 8, 4, 5, 5)
    out = sa(x)
    assert out.shape == x.shape


def test_cbam_applies_channel_before_spatial_sequentially():
    torch.manual_seed(0)
    channels = 8
    cbam = CBAM(channels)
    x = torch.randn(2, channels, 4, 5, 5)

    # manually run channel attention then spatial attention and confirm
    # it matches CBAM's forward exactly (i.e. sequential composition, not
    # e.g. gates computed independently and combined in parallel).
    expected = cbam.spatial_attention(cbam.channel_attention(x))
    assert torch.allclose(cbam(x), expected)


def test_cbam_output_shape_matches_input():
    cbam = CBAM(channels=16)
    x = torch.randn(3, 16, 6, 10, 10)
    out = cbam(x)
    assert out.shape == x.shape
