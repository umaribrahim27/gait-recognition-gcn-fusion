"""CLI entrypoint for training the ResGCN skeleton branch.

Usage:
    python -m training.train_resgcn --config configs/resgcn.yaml --demo
    python -m training.train_resgcn --config configs/resgcn.yaml --data-root /path/to/casia-b

`--demo` trains on data/synthetic.py's SyntheticGaitDataset (no CASIA-B
required) -- useful to prove the pipeline runs end to end. `--data-root`
points at a real CASIA-B-layout directory (see data/casia_b.py).
"""

import argparse

import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.casia_b import CasiaBDataset
from data.synthetic import SyntheticGaitDataset
from data.transforms_pose import default_train_transforms
from losses.supervised_contrastive import SupervisedContrastiveLoss
from models.resgcn.resgcn import ResGCN
from training.trainer import OptimConfig, Trainer, build_optimizer_and_scheduler


def forward_fn(model, batch):
    return F.normalize(model(batch["pose"]), dim=-1)


def build_dataset(args, sequence_length: int):
    if args.demo:
        return SyntheticGaitDataset(sequence_length=sequence_length)
    return CasiaBDataset(
        root=args.data_root,
        split="train",
        sequence_length=sequence_length,
        pose_transform=default_train_transforms(sequence_length),
        load_video=False,
        strict=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resgcn.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint", default="checkpoints/resgcn_best.pt")
    args = parser.parse_args()

    if not args.demo and args.data_root is None:
        parser.error("either --demo or --data-root is required")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sequence_length = config["data"]["sequence_length"]
    dataset = build_dataset(args, sequence_length)
    dataloader = DataLoader(dataset, batch_size=config["optim"]["batch_size"], shuffle=True)

    model = ResGCN(**config["model"])
    loss_fn = SupervisedContrastiveLoss(temperature=config["loss"]["temperature"])

    epochs = args.epochs or config["optim"]["epochs"]
    optim_config = OptimConfig(max_lr=config["optim"]["max_lr"], epochs=epochs, batch_size=config["optim"]["batch_size"])
    optimizer, scheduler = build_optimizer_and_scheduler(model, optim_config, steps_per_epoch=len(dataloader))

    trainer = Trainer(model, loss_fn, forward_fn, optimizer, scheduler, device="cpu")
    trainer.fit(dataloader, epochs=epochs, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
