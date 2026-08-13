"""HRNet pose extraction, wrapping a real pretrained implementation.

The paper (Sec. 3.2, ref. [17]) uses HRNet pretrained on COCO to produce
17 keypoints per frame. This module wraps OpenMMLab's `mmpose`, which
ships the actual HRNet-W32/W48 architecture with official COCO-pretrained
checkpoints -- it is not a from-scratch reimplementation of HRNet.

This wrapper is NOT exercised in this repo's smoke test or verified in
this session: `mmpose` (plus its `mmcv`/`mmengine`/`mmdet` dependencies)
was not installed, and no HRNet checkpoint was downloaded, because this
environment has neither the CASIA-B dataset nor a need to run live
inference for the equation-level verification requested. Everything
downstream of pose extraction (branches.py, resgcn.py) only assumes the
output contract below, and is tested against synthetic tensors of that
shape instead.

Install (not run here):
    pip install openmim
    mim install mmengine mmcv mmpose

Checkpoint (not downloaded here): HRNet-w32, COCO-pretrained, from the
mmpose model zoo (associated config: `td-hm_hrnet-w32_8xb64-210e_coco-256x192`).
"""

from dataclasses import dataclass

import torch


@dataclass
class HRNetConfig:
    config_path: str = "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    checkpoint_path: str = "hrnet_w32_coco_256x192-c78dce93_20200708.pth"
    device: str = "cpu"


class HRNetPoseExtractor:
    """Thin adapter around mmpose's inference API.

    Import of mmpose is deferred to __init__ so the rest of this codebase
    (and its tests) can run without the mmpose dependency installed.
    """

    def __init__(self, config: HRNetConfig = HRNetConfig()):
        try:
            from mmpose.apis import inference_topdown, init_model
        except ImportError as exc:
            raise ImportError(
                "mmpose is required for real HRNet inference. "
                "Install with: pip install openmim && mim install mmengine mmcv mmpose"
            ) from exc
        self._inference_topdown = inference_topdown
        self.model = init_model(config.config_path, config.checkpoint_path, device=config.device)

    def extract(self, frame) -> torch.Tensor:
        """Runs HRNet on a single BGR/RGB frame (H, W, 3) numpy array.

        Returns a (17, 3) tensor of (x, y, confidence), matching the
        v_{t,n} = (x_n, y_n, c_n) convention used throughout Sec. 3.2.
        """
        results = self._inference_topdown(self.model, frame)
        keypoints = results[0].pred_instances.keypoints[0]  # (17, 2)
        scores = results[0].pred_instances.keypoint_scores[0]  # (17,)
        return torch.cat(
            [torch.as_tensor(keypoints), torch.as_tensor(scores).unsqueeze(-1)], dim=-1
        )
