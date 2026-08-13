"""Video-frame transforms described in Sec. 3.1: "resolution
standardisation to 80x80 pixels, intensity normalisation across RGB
channels and statistical normalisation using mean and standard deviation."

Operates on a clip tensor (T, H, W, 3) in [0, 255] uint8/float, channel
layout matching common frame-decoding output; `ClipToCHW` converts to the
(C, T, H, W) layout consumed by VGGConv3D.
"""

import torch
import torch.nn.functional as F

# Mean/std are not given numerically in the paper ("statistical
# normalisation using mean and standard deviation") -- ImageNet statistics
# are used here as a standard default, not a value read off the paper.
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class ResizeFrames:
    """Resizes each frame to a fixed (H, W), default 80x80 (Sec. 3.1)."""

    def __init__(self, size=(80, 80)):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, H, W, C) -> (T, C, H, W) for interpolate -> back to (T, H, W, C)
        t, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1)


class IntensityNormalise:
    """Scales pixel values from [0, 255] to [0, 1] per RGB channel."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0 if x.max() > 1.0 else x


class StatisticalNormalise:
    """Standardises each channel by (x - mean) / std."""

    def __init__(self, mean=DEFAULT_MEAN, std=DEFAULT_STD):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ClipToCHW:
    """Converts (T, H, W, C) -> (C, T, H, W), the layout VGGConv3D expects."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(3, 0, 1, 2).contiguous()


def default_video_transforms(size=(80, 80)) -> Compose:
    return Compose(
        [
            ResizeFrames(size),
            IntensityNormalise(),
            StatisticalNormalise(),
            ClipToCHW(),
        ]
    )
