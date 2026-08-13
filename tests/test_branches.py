"""Correctness tests for the multi-branch feature computations
(equations 2-6), checking actual values, not just shapes.
"""

import torch

from data.graph import BONE_PARENTS
from models.resgcn.branches import (
    bone_length,
    build_branch_inputs,
    concat_branches,
    joint_position,
    motion_velocity,
)


def test_bone_length_root_is_zero():
    # BONE_PARENTS[0] == 0 (nose is its own parent) -> zero bone vector.
    x = torch.randn(1, 5, 17, 3)
    lengths = bone_length(x)
    assert torch.allclose(lengths[:, :, 0, :], torch.zeros(1, 5, 3), atol=1e-6)


def test_bone_length_matches_manual_diff():
    x = torch.randn(1, 5, 17, 3)
    lengths = bone_length(x)
    joint_idx = 9  # left wrist, parent = 7 (left elbow)
    parent_idx = BONE_PARENTS[joint_idx]
    expected = x[:, :, joint_idx, :] - x[:, :, parent_idx, :]
    assert torch.allclose(lengths[:, :, joint_idx, :], expected)


def test_motion_velocity_zero_for_static_pose():
    # a pose sequence with no change over time should have zero velocity
    # everywhere it's defined (last 2 frames are zero-padded regardless).
    static_pose = torch.randn(1, 1, 17, 3).repeat(1, 10, 1, 1)
    velocity = motion_velocity(static_pose)
    assert torch.allclose(velocity, torch.zeros_like(velocity), atol=1e-6)


def test_motion_velocity_last_frames_are_padded_zero():
    x = torch.randn(1, 10, 17, 3)
    velocity = motion_velocity(x, offsets=(1, 2))
    # last frame has no i=1 or i=2 successor -> both halves zero.
    assert torch.allclose(velocity[:, -1], torch.zeros(1, 17, 6), atol=1e-6)


def test_joint_position_centroid_is_zero():
    x = torch.randn(1, 5, 17, 3)
    positions = joint_position(x)
    centroid_of_output = positions.mean(dim=2)
    assert torch.allclose(centroid_of_output, torch.zeros(1, 5, 3), atol=1e-6)


def test_branches_are_kept_separate_before_concat():
    x = torch.randn(2, 5, 17, 3)
    branches = build_branch_inputs(x)
    assert set(branches.keys()) == {"bone", "motion", "joint"}
    assert branches["bone"].shape == (2, 5, 17, 6)  # length(3) + angle(3)
    assert branches["motion"].shape == (2, 5, 17, 6)  # 2 offsets x 3
    assert branches["joint"].shape == (2, 5, 17, 3)


def test_concat_branches_matches_equation_6():
    a = torch.randn(2, 4, 5, 17)
    b = torch.randn(2, 4, 5, 17)
    c = torch.randn(2, 4, 5, 17)
    out = concat_branches(a, b, c)
    assert out.shape == (2, 12, 5, 17)
    assert torch.equal(out[:, 0:4], a)
    assert torch.equal(out[:, 4:8], b)
    assert torch.equal(out[:, 8:12], c)
