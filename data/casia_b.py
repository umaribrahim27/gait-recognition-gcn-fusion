"""CASIA-B dataset loading, following the evaluation protocol in Sec. 4.1:

  - 124 subjects total; first 74 (subject ids 1-74) train, remaining 50
    (75-124) test (the "large-sample training" protocol).
  - Per subject: 6 normal-walking (NM), 2 bag (BG), and 2 coat (CL)
    sequences, each captured from 11 views (0-180 degrees, 18-degree
    steps) -> 10 sequences x 11 views = 110 sequences/subject.
  - In the test set: NM 1-4 form the gallery; NM 5-6, BG 1-2, and CL 1-2
    form three separate probe subsets.

This split logic (`split_subjects`, `gallery_probe_sequences`) is pure and
unit-tested without needing the actual dataset on disk. `CasiaBDataset`
additionally defines an expected on-disk layout for pre-extracted
poses/frames -- that layout is an assumption (the paper doesn't specify a
file format), documented below and in the README, since CASIA-B is not
available in this environment.

Expected layout (assumed, not verified against real CASIA-B data):

    root/
      {subject_id:03d}/
        {condition}-{seq:02d}/        e.g. nm-01, bg-01, cl-01
          {view:03d}/                 e.g. 000, 018, ..., 180
            pose.npy                  float array, shape (T_raw, 17, 3)
            frames/
              000000.jpg, 000001.jpg, ...
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

VIEWS = [f"{a:03d}" for a in range(0, 181, 18)]  # 11 views, 0-180 step 18
CONDITION_COUNTS = {"nm": 6, "bg": 2, "cl": 2}
NUM_SUBJECTS = 124
NUM_TRAIN_SUBJECTS = 74


def split_subjects(num_subjects: int = NUM_SUBJECTS, num_train: int = NUM_TRAIN_SUBJECTS):
    """Returns (train_ids, test_ids), subject ids 1-indexed per CASIA-B convention."""
    all_ids = list(range(1, num_subjects + 1))
    return all_ids[:num_train], all_ids[num_train:]


def gallery_probe_sequences():
    """Returns {split_name: [(condition, seq_num), ...]} per Sec. 4.1's protocol."""
    return {
        "gallery": [("nm", i) for i in (1, 2, 3, 4)],
        "probe_nm": [("nm", i) for i in (5, 6)],
        "probe_bg": [("bg", i) for i in (1, 2)],
        "probe_cl": [("cl", i) for i in (1, 2)],
    }


@dataclass
class CasiaBSample:
    subject_id: int
    condition: str
    seq: int
    view: str
    dir_path: Path


def _discover_samples(root: Path, subject_ids, condition_seqs, views=VIEWS):
    samples = []
    for subject_id in subject_ids:
        for condition, seq in condition_seqs:
            for view in views:
                sample_dir = root / f"{subject_id:03d}" / f"{condition}-{seq:02d}" / view
                if sample_dir.is_dir():
                    samples.append(CasiaBSample(subject_id, condition, seq, view, sample_dir))
    return samples


class CasiaBDataset(Dataset):
    """`split` is one of "train", "gallery", "probe_nm", "probe_bg", "probe_cl".

    Missing directories are silently skipped by `_discover_samples` (a
    partial/synthetic mirror of CASIA-B is fine for testing the split
    logic); use `strict=True` to instead raise if nothing is found.
    """

    def __init__(
        self,
        root: str,
        split: str,
        sequence_length: int = 59,
        pose_transform=None,
        video_transform=None,
        load_video: bool = True,
        strict: bool = False,
    ):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.pose_transform = pose_transform
        self.video_transform = video_transform
        self.load_video = load_video

        train_ids, test_ids = split_subjects()
        seqs = gallery_probe_sequences()
        if split == "train":
            subject_ids, condition_seqs = train_ids, [(c, i) for c, n in CONDITION_COUNTS.items() for i in range(1, n + 1)]
        elif split in seqs:
            subject_ids, condition_seqs = test_ids, seqs[split]
        else:
            raise ValueError(f"Unknown split '{split}', expected 'train' or one of {list(seqs)}")

        self.samples = _discover_samples(self.root, subject_ids, condition_seqs)
        if strict and not self.samples:
            raise FileNotFoundError(f"No CASIA-B samples found under {self.root} for split '{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        pose = torch.from_numpy(np.load(sample.dir_path / "pose.npy")).float()
        if self.pose_transform is not None:
            pose = self.pose_transform(pose)

        video = None
        if self.load_video:
            video = self._load_frames(sample.dir_path / "frames")
            if self.video_transform is not None:
                video = self.video_transform(video)

        return {
            "pose": pose,
            "video": video,
            "label": sample.subject_id,
            "condition": sample.condition,
            "view": sample.view,
        }

    @staticmethod
    def _load_frames(frames_dir: Path) -> torch.Tensor:
        from PIL import Image

        paths = sorted(frames_dir.glob("*.jpg"))
        frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
        return torch.from_numpy(np.stack(frames)).float()  # (T, H, W, 3)
