"""Rank-1 accuracy evaluation, matching the reporting style of the paper's
Tables 1-3: overall accuracy plus a per-walking-condition (NM/BG/CL)
breakdown, computed by matching probe embeddings against gallery
embeddings of known identity.
"""

from collections import defaultdict

import torch


def rank1_accuracy(
    probe_embeddings: torch.Tensor,
    probe_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> float:
    """Nearest-gallery-neighbour (Lp / cosine-distance) rank-1 accuracy,
    the paper's baseline evaluator before the Bayes classifier (Sec.
    4.2.2) is applied on top.
    """
    # embeddings assumed L2-normalised, so squared-Euclidean distance
    # ranking is equivalent to cosine-similarity ranking.
    dists = torch.cdist(probe_embeddings, gallery_embeddings)  # (P, G)
    nearest = dists.argmin(dim=1)
    predictions = gallery_labels[nearest]
    return (predictions == probe_labels).float().mean().item()


def rank1_by_condition(
    probe_embeddings: torch.Tensor,
    probe_labels: torch.Tensor,
    probe_conditions,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> dict:
    """Breaks rank-1 accuracy down per condition (nm/bg/cl), mirroring
    the paper's NM/BG/CL columns in Table 1.
    """
    by_condition = defaultdict(list)
    for i, condition in enumerate(probe_conditions):
        by_condition[condition].append(i)

    results = {}
    for condition, indices in by_condition.items():
        idx = torch.tensor(indices)
        results[condition] = rank1_accuracy(
            probe_embeddings[idx], probe_labels[idx], gallery_embeddings, gallery_labels
        )
    if results:
        results["average"] = sum(results.values()) / len(results)
    return results
