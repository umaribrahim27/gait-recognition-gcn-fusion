"""Pose-sequence augmentations described in Sec. 3.1: "sequence length
normalisation... flipping the entire sequence, selecting random segments,
and shuffling pose order... horizontally flipping poses and adding noise
to joint positions... random translation."

Exact parameter values (noise scale, translation range, flip probability)
are not given numerically in the paper -- defaults below are reasonable
choices, documented as inferred, and are all overridable.

All transforms operate on a single sequence: a (T, N, 3) tensor of
(x, y, confidence), and are composable via `Compose`.
"""

import random

import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class PadOrTruncate:
    """Normalises sequence length to a fixed T (Sec. 3.1: "ensuring a
    consistent sequence length through padding")."""

    def __init__(self, target_length: int):
        self.target_length = target_length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        if t == self.target_length:
            return x
        if t > self.target_length:
            return x[: self.target_length]
        pad = x[-1:].repeat(self.target_length - t, 1, 1)
        return torch.cat([x, pad], dim=0)


class RandomSequenceFlip:
    """Flips the entire sequence in time (frame order reversed)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return x.flip(dims=(0,))
        return x


class RandomSegmentSelect:
    """Selects a random contiguous segment and pads/truncates back to T,
    per "selecting random segments" (Sec. 3.1)."""

    def __init__(self, target_length: int, min_ratio: float = 0.7):
        self.target_length = target_length
        self.min_ratio = min_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        seg_len = random.randint(int(t * self.min_ratio), t)
        start = random.randint(0, t - seg_len)
        segment = x[start : start + seg_len]
        return PadOrTruncate(self.target_length)(segment)


class RandomPoseShuffle:
    """Shuffles frame order within the sequence ("shuffling pose order")."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            perm = torch.randperm(x.shape[0])
            return x[perm]
        return x


# Left/right joint swap for the COCO-17 layout, used by RandomHorizontalFlip.
_LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class RandomHorizontalFlip:
    """Mirrors x-coordinates and swaps left/right joint indices."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            x = x[:, _LR_SWAP, :]
            x = x.clone()
            x[..., 0] = -x[..., 0]
        return x


class RandomJointNoise:
    """Adds Gaussian noise to joint (x, y) coordinates only, not confidence."""

    def __init__(self, std: float = 0.01, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            noise = torch.zeros_like(x)
            noise[..., :2] = torch.randn_like(x[..., :2]) * self.std
            return x + noise
        return x


class RandomTranslation:
    """Applies a random (dx, dy) shift to all joints in a sequence."""

    def __init__(self, max_shift: float = 0.05, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            shift = (torch.rand(2) * 2 - 1) * self.max_shift
            x = x.clone()
            x[..., :2] = x[..., :2] + shift
        return x


def default_train_transforms(sequence_length: int) -> Compose:
    return Compose(
        [
            PadOrTruncate(sequence_length),
            RandomSequenceFlip(),
            RandomSegmentSelect(sequence_length),
            RandomPoseShuffle(),
            RandomHorizontalFlip(),
            RandomJointNoise(),
            RandomTranslation(),
        ]
    )


def default_eval_transforms(sequence_length: int) -> Compose:
    return Compose([PadOrTruncate(sequence_length)])
