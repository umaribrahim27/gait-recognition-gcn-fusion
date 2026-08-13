"""Integration test: trains the full GaitNet pipeline (both branches +
fusion) on a small, separable synthetic dataset for a handful of steps and
confirms the supervised contrastive loss actually decreases -- proof the
pipeline is not just shape-correct but trainable end to end.

Uses deliberately tiny model/data dimensions (not the paper's T=59/128-dim
config) purely to keep this fast on CPU; test_smoke.py separately checks
the paper's actual stated shapes.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.synthetic import SyntheticGaitDataset
from losses.supervised_contrastive import SupervisedContrastiveLoss
from models.fusion.fusion_module import FusionModule
from models.resgcn.resgcn import ResGCN
from models.vggconv3d.vggconv3d import VGGConv3D


def test_gait_net_loss_decreases_on_separable_synthetic_data():
    torch.manual_seed(0)

    num_classes, samples_per_class = 4, 6
    sequence_length, num_joints = 12, 17
    frame_size = (16, 16)

    dataset = SyntheticGaitDataset(
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        sequence_length=sequence_length,
        num_joints=num_joints,
        frame_size=frame_size,
        noise_std=0.02,
        seed=1,
    )
    dataloader = DataLoader(dataset, batch_size=num_classes * samples_per_class, shuffle=True)

    resgcn = ResGCN(num_joints=num_joints, base_channels=8, main_channels=16, num_heads=2, head_dim=8, embedding_size=16)
    vggconv3d = VGGConv3D(channels=(8, 16), embedding_size=16)
    fusion = FusionModule(embedding_size=16, fused_size=16)

    params = list(resgcn.parameters()) + list(vggconv3d.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)
    loss_fn = SupervisedContrastiveLoss(temperature=0.1)

    losses = []
    for _ in range(15):
        for batch in dataloader:
            skel_emb = resgcn(batch["pose"])
            vid_emb = vggconv3d(batch["video"])
            fused = F.normalize(fusion(skel_emb, vid_emb), dim=-1)

            loss = loss_fn(fused, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    # allow noisy individual steps but require clear overall improvement.
    early_avg = sum(losses[:3]) / 3
    late_avg = sum(losses[-3:]) / 3
    assert late_avg < early_avg, f"loss did not decrease: early={early_avg:.4f} late={late_avg:.4f}"
