"""Supervised Contrastive Loss, equation 13:

    L_sup = sum_{i in I} [ -1/|P(i)| * sum_{p in P(i)}
                log( exp(z_i . z_p / tau) / sum_{a in A(i)} exp(z_i . z_a / tau) ) ]

I: batch indices, P(i): indices of other samples sharing i's label,
A(i): all indices other than i, z: L2-normalised embeddings, tau: temperature.
"""

import torch
import torch.nn as nn


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, D), assumed already L2-normalised (z_i)
        device = embeddings.device
        batch_size = embeddings.shape[0]

        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B), z_i . z_a / tau

        # A(i): all samples other than i.
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        logits_mask = ~self_mask

        # numerically stable log-sum-exp over A(i)
        sim_masked = sim.masked_fill(self_mask, float("-inf"))
        log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)  # (B, 1)
        log_prob = sim - log_denom  # log( exp(z_i.z_a/tau) / sum_{A(i)} exp(z_i.z_a/tau) )

        # P(i): same label as i, excluding i itself.
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T) & logits_mask  # (B, B)
        pos_counts = positive_mask.sum(dim=1)  # |P(i)|

        # anchors with at least one positive contribute to the loss.
        valid = pos_counts > 0
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / pos_counts[valid]

        loss = -mean_log_prob_pos.mean()
        return loss
