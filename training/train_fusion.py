"""CLI entrypoint for training the fusion module on top of the two
branches (Sec. 3.4). Loads pretrained ResGCN/VGGConv3D checkpoints if
present (paths from the config); optionally freezes them
(`freeze_branches: true`, the default) so only the fusion head trains.

Usage:
    python -m training.train_fusion --config configs/fusion.yaml --demo
    python -m training.train_fusion --config configs/fusion.yaml --data-root /path/to/casia-b

--demo uses small synthetic clips/batches (see train_vggconv3d.py) so the
full pipeline can be verified on CPU in a reasonable time.
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.casia_b import CasiaBDataset
from data.synthetic import SyntheticGaitDataset
from data.transforms_pose import default_train_transforms
from data.transforms_video import default_video_transforms
from losses.supervised_contrastive import SupervisedContrastiveLoss
from models.gait_net import GaitNet
from training.trainer import OptimConfig, Trainer, build_optimizer_and_scheduler


def forward_fn(model, batch):
    return model(batch["pose"], batch["video"])  # GaitNet already L2-normalises its output


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
        pose_transform=default_train_transforms(sequence_length),
        video_transform=default_video_transforms(tuple(frame_size)),
        load_video=True,
        strict=True,
    )


def load_branch_checkpoints(model: GaitNet, config: dict):
    ckpts = config.get("branch_checkpoints", {})
    if Path(ckpts.get("resgcn", "")).is_file():
        model.resgcn.load_state_dict(torch.load(ckpts["resgcn"], map_location="cpu"))
    if Path(ckpts.get("vggconv3d", "")).is_file():
        model.vggconv3d.load_state_dict(torch.load(ckpts["vggconv3d"], map_location="cpu"))

    if config.get("freeze_branches", True):
        for param in list(model.resgcn.parameters()) + list(model.vggconv3d.parameters()):
            param.requires_grad = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint", default="checkpoints/fusion_best.pt")
    parser.add_argument("--frame-size", type=int, nargs=2, default=None, help="overrides frame size, e.g. --frame-size 32 32")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    if not args.demo and args.data_root is None:
        parser.error("either --demo or --data-root is required")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sequence_length = 59
    default_frame_size = (32, 32) if args.demo else (80, 80)
    frame_size = args.frame_size or default_frame_size
    batch_size = args.batch_size or (8 if args.demo else config["optim"]["batch_size"])

    dataset = build_dataset(args, sequence_length, frame_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GaitNet(**config["model"])
    load_branch_checkpoints(model, config)
    loss_fn = SupervisedContrastiveLoss(temperature=config["loss"]["temperature"])

    epochs = args.epochs or config["optim"]["epochs"]
    optim_config = OptimConfig(max_lr=config["optim"]["max_lr"], epochs=epochs, batch_size=batch_size)
    optimizer, scheduler = build_optimizer_and_scheduler(model, optim_config, steps_per_epoch=len(dataloader))

    trainer = Trainer(model, loss_fn, forward_fn, optimizer, scheduler, device="cpu")
    trainer.fit(dataloader, epochs=epochs, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
