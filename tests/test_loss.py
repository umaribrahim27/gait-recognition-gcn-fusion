"""Correctness tests for the supervised contrastive loss (equation 13):
multiple positives per anchor, and actual optimisation behaviour (loss
decreases as same-class embeddings are pulled together).
"""

import torch
import torch.nn.functional as F

from losses.supervised_contrastive import SupervisedContrastiveLoss


def test_loss_is_lower_for_well_separated_classes():
    torch.manual_seed(0)
    loss_fn = SupervisedContrastiveLoss(temperature=0.1)

    labels = torch.tensor([0, 0, 1, 1])

    # well-separated: class 0 embeddings near [1,0,...], class 1 near [0,1,...]
    good = F.normalize(
        torch.stack(
            [
                torch.tensor([1.0, 0.0, 0.0, 0.0]) + 0.01 * torch.randn(4),
                torch.tensor([1.0, 0.0, 0.0, 0.0]) + 0.01 * torch.randn(4),
                torch.tensor([0.0, 1.0, 0.0, 0.0]) + 0.01 * torch.randn(4),
                torch.tensor([0.0, 1.0, 0.0, 0.0]) + 0.01 * torch.randn(4),
            ]
        ),
        dim=-1,
    )
    # poorly separated: all embeddings close together regardless of label.
    bad = F.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(4, 1) + 0.01 * torch.randn(4, 4), dim=-1)

    assert loss_fn(good, labels).item() < loss_fn(bad, labels).item()


def test_multiple_positives_per_anchor_all_contribute():
    # with 3 samples of the same class, each anchor has 2 positives; the
    # loss should equal the mean over both positives per eq. 13's
    # 1/|P(i)| averaging, not just the single nearest one.
    loss_fn = SupervisedContrastiveLoss(temperature=0.5)
    embeddings = F.normalize(torch.randn(3, 4), dim=-1)
    labels = torch.tensor([0, 0, 0])

    sim = torch.matmul(embeddings, embeddings.T) / 0.5
    log_prob = []
    for i in range(3):
        denom = torch.logsumexp(torch.cat([sim[i, :i], sim[i, i + 1 :]]), dim=0)
        positives = [sim[i, p] - denom for p in range(3) if p != i]
        log_prob.append(sum(positives) / len(positives))
    expected = -sum(log_prob) / 3

    assert torch.allclose(loss_fn(embeddings, labels), expected, atol=1e-5)


def test_gradient_descent_reduces_loss():
    torch.manual_seed(0)
    embeddings = torch.nn.Parameter(torch.randn(6, 8))
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss_fn = SupervisedContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.SGD([embeddings], lr=0.5)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        normed = F.normalize(embeddings, dim=-1)
        loss = loss_fn(normed, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
