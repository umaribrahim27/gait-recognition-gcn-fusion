"""Evaluation entrypoint: computes embeddings for the gallery and each
probe subset (NM/BG/CL, Sec. 4.1's protocol), then reports rank-1 accuracy
both via nearest-neighbour distance and via the Bayes classifier
(eq. 14-18), matching the two evaluators compared in Sec. 4.2.2.

Usage:
    python -m scripts.evaluate --checkpoint checkpoints/fusion_best.pt --demo
    python -m scripts.evaluate --checkpoint checkpoints/fusion_best.pt --data-root /path/to/casia-b
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from data.casia_b import CasiaBDataset
from data.synthetic import SyntheticGaitDataset
from data.transforms_pose import default_eval_transforms
from data.transforms_video import default_video_transforms
from evaluation.bayes_classifier import BayesClassifier
from evaluation.metrics import rank1_accuracy, rank1_by_condition
from models.gait_net import GaitNet


@torch.no_grad()
def compute_embeddings(model: GaitNet, dataloader: DataLoader):
    embeddings, labels, conditions = [], [], []
    model.eval()
    for batch in dataloader:
        emb = model(batch["pose"], batch["video"])
        embeddings.append(emb)
        labels.append(batch["label"])
        conditions.extend(batch.get("condition", ["nm"] * emb.shape[0]))
    return torch.cat(embeddings), torch.cat(labels), conditions


def build_split(args, split: str, sequence_length: int, frame_size):
    if args.demo:
        # a single synthetic pool stands in for both gallery and probe.
        return SyntheticGaitDataset(
            sequence_length=sequence_length,
            frame_size=tuple(frame_size),
            num_classes=4,
            samples_per_class=4,
            seed=hash(split) % 1000,
        )
    return CasiaBDataset(
        root=args.data_root,
        split=split,
        sequence_length=sequence_length,
        pose_transform=default_eval_transforms(sequence_length),
        video_transform=default_video_transforms(tuple(frame_size)),
        load_video=True,
        strict=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/fusion.yaml", help="used to build the matching GaitNet architecture")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--frame-size", type=int, nargs=2, default=None)
    args = parser.parse_args()

    if not args.demo and args.data_root is None:
        parser.error("either --demo or --data-root is required")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sequence_length = args.sequence_length or (12 if args.demo else 59)
    default_frame_size = (32, 32) if args.demo else (80, 80)
    frame_size = args.frame_size or default_frame_size

    model = GaitNet(**config["model"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    gallery_ds = build_split(args, "gallery", sequence_length, frame_size)
    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size)
    gallery_emb, gallery_labels, _ = compute_embeddings(model, gallery_loader)

    bayes = BayesClassifier()
    bayes.fit(gallery_emb, gallery_labels)

    for split in ("probe_nm", "probe_bg", "probe_cl") if not args.demo else ("probe",):
        probe_ds = build_split(args, split if split != "probe" else "gallery", sequence_length, frame_size)
        probe_loader = DataLoader(probe_ds, batch_size=args.batch_size)
        probe_emb, probe_labels, probe_conditions = compute_embeddings(model, probe_loader)

        lp_acc = rank1_accuracy(probe_emb, probe_labels, gallery_emb, gallery_labels)
        bayes_preds = bayes.predict(probe_emb)
        bayes_acc = BayesClassifier.accuracy(bayes_preds, probe_labels)

        print(f"[{split}] Lp rank-1 accuracy: {lp_acc:.4f}  |  Bayes accuracy: {bayes_acc:.4f}")

        by_condition = rank1_by_condition(probe_emb, probe_labels, probe_conditions, gallery_emb, gallery_labels)
        for condition, acc in by_condition.items():
            print(f"    {condition}: {acc:.4f}")


if __name__ == "__main__":
    main()
