"""Skeleton graph construction for the 17-keypoint HRNet/COCO layout.

The paper (Sec. 3.2.1) defines the skeleton graph G = (V, E) with an
adjacency matrix A in R^{N x N}, but does not enumerate the exact joint
edges used for CASIA-B. We use the standard COCO-17 kinematic tree
(nose-rooted), which is the conventional choice in skeleton-based gait/
action recognition work building on HRNet-COCO keypoints. This mapping is
an assumption, not something specified in the paper.
"""

import torch

NUM_JOINTS = 17

# joint index -> parent joint index, forming a nose-rooted tree.
# Root (nose) maps to itself, giving it a zero bone vector.
BONE_PARENTS = [
    0,  # 0 nose (root)
    0,  # 1 left_eye -> nose
    0,  # 2 right_eye -> nose
    1,  # 3 left_ear -> left_eye
    2,  # 4 right_ear -> right_eye
    0,  # 5 left_shoulder -> nose
    0,  # 6 right_shoulder -> nose
    5,  # 7 left_elbow -> left_shoulder
    6,  # 8 right_elbow -> right_shoulder
    7,  # 9 left_wrist -> left_elbow
    8,  # 10 right_wrist -> right_elbow
    5,  # 11 left_hip -> left_shoulder
    6,  # 12 right_hip -> right_shoulder
    11,  # 13 left_knee -> left_hip
    12,  # 14 right_knee -> right_hip
    13,  # 15 left_ankle -> left_knee
    14,  # 16 right_ankle -> right_knee
]

# Undirected bone edges (child, parent) with root self-loop removed, used
# to build the spatial adjacency matrix A.
BONE_EDGES = [(n, p) for n, p in enumerate(BONE_PARENTS) if n != p]


def build_adjacency(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """Builds the raw (no self-loop) adjacency matrix A in R^{N x N}."""
    a = torch.zeros(num_joints, num_joints)
    for i, j in BONE_EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    return a


def normalized_adjacency(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """Computes D~^{-1/2} A~ D~^{-1/2} from equation 1, with A~ = A + I."""
    a = build_adjacency(num_joints)
    a_tilde = a + torch.eye(num_joints)
    deg = a_tilde.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    d_tilde = torch.diag(deg_inv_sqrt)
    return d_tilde @ a_tilde @ d_tilde
