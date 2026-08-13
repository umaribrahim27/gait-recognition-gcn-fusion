"""Bayesian classifier, equations 14-18.

    P(C)      = (# samples in class C) / (total # samples)             (14)
    P(x|C)    = Gaussian likelihood, mean mu_C, covariance Sigma_C      (15)
    P(C|x)    = P(x|C) P(C) / sum_k P(x|C_k) P(C_k)                     (16)
    y_hat     = argmax_c P(C=c|x)                                      (17)
    Accuracy  = (# correctly predicted) / (total # samples)            (18)

Fit on gallery embeddings (per class mean/covariance + prior), applied to
probe embeddings.
"""

from dataclasses import dataclass

import torch


@dataclass
class GaussianClassStats:
    mean: torch.Tensor  # (D,)
    covariance: torch.Tensor  # (D, D)
    prior: float


class BayesClassifier:
    def __init__(self, diagonal_covariance: bool = True, eps: float = 1e-6):
        # A full D x D covariance is what equation 15 specifies, but with
        # embedding_size=128 and typically few gallery samples per class,
        # a full covariance is singular / not estimable. We default to a
        # diagonal covariance (independent dimensions) for numerical
        # stability -- documented deviation from the literal equation.
        self.diagonal_covariance = diagonal_covariance
        self.eps = eps
        self.classes: dict = {}

    def fit(self, gallery_embeddings: torch.Tensor, gallery_labels: torch.Tensor) -> None:
        total = gallery_labels.shape[0]
        self.classes = {}
        for c in torch.unique(gallery_labels).tolist():
            mask = gallery_labels == c
            samples = gallery_embeddings[mask]  # (n_c, D)
            mean = samples.mean(dim=0)
            if self.diagonal_covariance:
                var = samples.var(dim=0, unbiased=False) + self.eps
                covariance = torch.diag(var)
            else:
                centered = samples - mean
                covariance = (centered.T @ centered) / max(samples.shape[0] - 1, 1)
                covariance += self.eps * torch.eye(covariance.shape[0])
            prior = mask.sum().item() / total  # equation 14
            self.classes[c] = GaussianClassStats(mean=mean, covariance=covariance, prior=prior)

    def _log_gaussian_likelihood(self, x: torch.Tensor, stats: GaussianClassStats) -> torch.Tensor:
        # log P(x|C), equation 15 in log form for numerical stability.
        d = x.shape[-1]
        diff = x - stats.mean
        cov_inv = torch.linalg.inv(stats.covariance)
        sign, logdet = torch.linalg.slogdet(stats.covariance)
        mahalanobis = torch.einsum("bi,ij,bj->b", diff, cov_inv, diff)
        return -0.5 * (d * torch.log(torch.tensor(2 * torch.pi)) + logdet + mahalanobis)

    def predict(self, probe_embeddings: torch.Tensor) -> torch.Tensor:
        class_ids = sorted(self.classes.keys())
        log_joint = []  # log( P(x|C) * P(C) ) per class, softmax gives eq. 16
        for c in class_ids:
            stats = self.classes[c]
            log_likelihood = self._log_gaussian_likelihood(probe_embeddings, stats)
            log_joint.append(log_likelihood + torch.log(torch.tensor(stats.prior)))
        log_joint = torch.stack(log_joint, dim=1)  # (B, num_classes)

        # equation 16: normalising by sum_k P(x|C_k)P(C_k) is exactly a
        # softmax over the log-joint terms; equation 17 argmax is invariant
        # to that normalisation, so we take argmax directly.
        preds_idx = torch.argmax(log_joint, dim=1)
        class_ids_t = torch.tensor(class_ids)
        return class_ids_t[preds_idx]

    @staticmethod
    def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        # equation 18
        return (predictions == labels).float().mean().item()
