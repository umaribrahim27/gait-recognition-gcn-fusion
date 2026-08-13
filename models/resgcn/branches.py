"""Multi-branch input features: equations 2-5, and the concatenation of
equation 6.

Input pose sequence convention: X of shape (B, T, N, 3), channels = (x, y,
confidence), matching the paper's v_{t,n} = (x_n, y_n, c_n).

The paper is not fully explicit about two details, both called out here and
in the README:
  - equation 3 (bone angle) divides a 3-vector by a scalar norm and then
    takes arccos, which only type-checks if arccos is applied element-wise
    to the resulting direction-cosine vector. We do that, per-axis.
  - equation 5's "pose centre c" is not defined; we use the per-frame
    centroid of all joints.
"""

import torch

from data.graph import BONE_PARENTS


def bone_length(x: torch.Tensor) -> torch.Tensor:
    """Equation 2: L_bone = v_{t,n} - v_{t,n_adj}."""
    parents = x[:, :, BONE_PARENTS, :]
    return x - parents


def bone_angle(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Equation 3: A_bone = arccos((v_{t,n} - v_{t,n_adj}) / ||v_{t,n}||)."""
    diff = bone_length(x)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    cos_theta = (diff / norm).clamp(-1.0 + eps, 1.0 - eps)
    return torch.arccos(cos_theta)


def motion_velocity(x: torch.Tensor, offsets=(1, 2)) -> torch.Tensor:
    """Equation 4: F = v_{t+i,n} - v_{t,n} for i in {1, 2}.

    Returns a tensor with the same T as the input, channel-concatenated
    over both offsets; the last `max(offsets)` frames are zero-padded since
    v_{t+i} is undefined there (the paper restricts t < T - 2 instead of
    padding, which would shrink T and break the concat in equation 6 — we
    pad instead, documented as an inferred choice).
    """
    b, t, n, c = x.shape
    parts = []
    for i in offsets:
        shifted = torch.zeros_like(x)
        shifted[:, : t - i] = x[:, i:] - x[:, : t - i]
        parts.append(shifted)
    return torch.cat(parts, dim=-1)


def joint_position(x: torch.Tensor) -> torch.Tensor:
    """Equation 5: R = v_{t,n} - v_{t,c}, centre c = per-frame joint centroid."""
    centre = x.mean(dim=2, keepdim=True)
    return x - centre


def build_branch_inputs(x: torch.Tensor) -> dict:
    """Builds the three separate branch inputs described in Sec. 3.2.2,
    each kept distinct (not merged into one feature) prior to equation 6.
    """
    bone = torch.cat([bone_length(x), bone_angle(x)], dim=-1)  # (B,T,N,6)
    motion = motion_velocity(x)  # (B,T,N,6)
    joint = joint_position(x)  # (B,T,N,3)
    return {"bone": bone, "motion": motion, "joint": joint}


def concat_branches(x_bone: torch.Tensor, x_motion: torch.Tensor, x_joint: torch.Tensor) -> torch.Tensor:
    """Equation 6: X_concat = concat(X_bone, X_motion, X_joint) along channels.

    Inputs are (B, C, T, N) branch outputs (post graph-conv blocks).
    """
    return torch.cat([x_bone, x_motion, x_joint], dim=1)
