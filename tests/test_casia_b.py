"""Tests for the CASIA-B split protocol and dataset loader, using a small
synthetic on-disk fixture (no real CASIA-B data required) to prove the
loader's directory-walking and shape handling actually work, not just the
pure split-logic functions.
"""

import numpy as np
import pytest

from data.casia_b import CasiaBDataset, gallery_probe_sequences, split_subjects
from data.transforms_pose import default_eval_transforms


def test_split_subjects_sizes_and_no_overlap():
    train_ids, test_ids = split_subjects()
    assert len(train_ids) == 74
    assert len(test_ids) == 50
    assert set(train_ids).isdisjoint(test_ids)
    assert train_ids[0] == 1 and test_ids[-1] == 124


def test_gallery_probe_sequences_protocol():
    seqs = gallery_probe_sequences()
    assert seqs["gallery"] == [("nm", 1), ("nm", 2), ("nm", 3), ("nm", 4)]
    assert seqs["probe_nm"] == [("nm", 5), ("nm", 6)]
    assert seqs["probe_bg"] == [("bg", 1), ("bg", 2)]
    assert seqs["probe_cl"] == [("cl", 1), ("cl", 2)]
    # gallery and every probe subset must be disjoint sequences.
    gallery = set(seqs["gallery"])
    for key in ("probe_nm", "probe_bg", "probe_cl"):
        assert gallery.isdisjoint(seqs[key])


@pytest.fixture
def synthetic_casia_root(tmp_path):
    """Builds a minimal fake CASIA-B tree: subjects 1, 75 (one train, one
    test id), a couple of conditions/views, each with a pose.npy."""
    for subject_id, conditions in ((1, [("nm", 1)]), (75, [("nm", 1), ("nm", 5), ("bg", 1)])):
        for condition, seq in conditions:
            for view in ("000", "018"):
                d = tmp_path / f"{subject_id:03d}" / f"{condition}-{seq:02d}" / view
                d.mkdir(parents=True)
                np.save(d / "pose.npy", np.random.randn(45, 17, 3).astype("float32"))
    return tmp_path


def test_dataset_train_split_only_sees_train_subject(synthetic_casia_root):
    ds = CasiaBDataset(
        root=synthetic_casia_root,
        split="train",
        sequence_length=59,
        pose_transform=default_eval_transforms(59),
        load_video=False,
        strict=True,
    )
    assert len(ds) > 0
    assert all(sample.subject_id == 1 for sample in ds.samples)


def test_dataset_gallery_split_only_sees_test_subject_and_nm1_4(synthetic_casia_root):
    ds = CasiaBDataset(
        root=synthetic_casia_root,
        split="gallery",
        sequence_length=59,
        pose_transform=default_eval_transforms(59),
        load_video=False,
        strict=True,
    )
    assert len(ds) > 0
    assert all(sample.subject_id == 75 for sample in ds.samples)
    assert all(sample.condition == "nm" and sample.seq == 1 for sample in ds.samples)


def test_dataset_getitem_shapes_and_padding(synthetic_casia_root):
    ds = CasiaBDataset(
        root=synthetic_casia_root,
        split="probe_nm",
        sequence_length=59,
        pose_transform=default_eval_transforms(59),
        load_video=False,
        strict=True,
    )
    item = ds[0]
    assert item["pose"].shape == (59, 17, 3)  # padded from the fixture's 45 frames
    assert item["video"] is None
    assert item["label"] == 75


def test_dataset_raises_when_strict_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        CasiaBDataset(root=tmp_path, split="train", load_video=False, strict=True)
