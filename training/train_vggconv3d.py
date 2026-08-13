"""CLI entrypoint for training the VGGConv3D video branch.

Usage:
    python -m training.train_vggconv3d --config configs/vggconv3d.yaml --demo
    python -m training.train_vggconv3d --config configs/vggconv3d.yaml --data-root /path/to/casia-b

3D convs over the paper's full 80x80xT=59 clips are CPU-heavy; --demo
defaults to a much smaller synthetic clip size/batch so the pipeline can
be verified in seconds rather than minutes. Override with --frame-size /
--batch-size, or drop to the full config values for a real (GPU) run.
"""

import argparse

import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.casia_b import CasiaBDataset
from data.synthetic import SyntheticGaitDataset
from data.transforms_video import default_video_transforms
from losses.supervised_contrastive import SupervisedContrastiveLoss
from models.vggconv3d.vggconv3d import VGGConv3D
from training.trainer import OptimConfig, Trainer, build_optimizer_and_scheduler


def forward_fn(model, batch):
    return F.normalize(model(batch["video"]), dim=-1)


def build_dataset(args, sequence_length: int, frame_size):
    if args.demo:
        return SyntheticGaitDataset(
            sequence_length=sequence_length,
            frame_size=tuple(frame_size),
            num_classes=4,
            samples_per_class=8,
        )
    return CasiaBDataset(
        root=args.data_root,
        split="train",
        sequence_length=sequence_length,
        video_transform=default_video_transforms(tuple(frame_size)),
        load_video=True,
        strict=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vggconv3d.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint", default="checkpoints/vggconv3d_best.pt")
    parser.add_argument("--frame-size", type=int, nargs=2, default=None, help="overrides config frame size, e.g. --frame-size 32 32")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    if not args.demo and args.data_root is None:
        parser.error("either --demo or --data-root is required")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sequence_length = config["data"]["sequence_length"]
    frame_size = args.frame_size or (config["data"]["frame_size"] if not args.demo else (32, 32))
    batch_size = args.batch_size or (8 if args.demo else config["optim"]["batch_size"])

    dataset = build_dataset(args, sequence_length, frame_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VGGConv3D(**config["model"])
    loss_fn = SupervisedContrastiveLoss(temperature=config["loss"]["temperature"])

    epochs = args.epochs or config["optim"]["epochs"]
    optim_config = OptimConfig(max_lr=config["optim"]["max_lr"], epochs=epochs, batch_size=batch_size)
    optimizer, scheduler = build_optimizer_and_scheduler(model, optim_config, steps_per_epoch=len(dataloader))

    trainer = Trainer(model, loss_fn, forward_fn, optimizer, scheduler, device="cpu")
    trainer.fit(dataloader, epochs=epochs, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
