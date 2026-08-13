"""Synthetic gait dataset for demos and tests, in place of CASIA-B (which
is not available in this environment). Each class gets a fixed random
"template" pose sequence and video clip; samples are the template plus
small noise, so classes are trivially separable -- enough to exercise and
sanity-check the full training pipeline (data loading -> model -> loss ->
optimizer step), not to produce a meaningful recognition model.
"""

import torch
from torch.utils.data import Dataset


class SyntheticGaitDataset(Dataset):
    def __init__(
        self,
        num_classes: int = 8,
        samples_per_class: int = 16,
        sequence_length: int = 59,
        num_joints: int = 17,
        frame_size=(80, 80),
        noise_std: float = 0.05,
        seed: int = 0,
    ):
        self.samples_per_class = samples_per_class
        self.sequence_length = sequence_length
        self.noise_std = noise_std

        generator = torch.Generator().manual_seed(seed)
        h, w = frame_size
        self.pose_templates = torch.randn(num_classes, sequence_length, num_joints, 3, generator=generator)
        self.video_templates = torch.randn(num_classes, 3, sequence_length, h, w, generator=generator)

    def __len__(self) -> int:
        return self.pose_templates.shape[0] * self.samples_per_class

    def __getitem__(self, idx: int):
        class_id = idx // self.samples_per_class
        pose = self.pose_templates[class_id] + torch.randn_like(self.pose_templates[class_id]) * self.noise_std
        video = self.video_templates[class_id] + torch.randn_like(self.video_templates[class_id]) * self.noise_std
        return {"pose": pose, "video": video, "label": class_id}
