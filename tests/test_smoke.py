"""End-to-end smoke test: forward pass through ResGCN, VGGConv3D, fusion,
loss, and the Bayes classifier with dummy/random tensors at the paper's
stated shapes (T=59, N=17 joints, embedding size 128).
"""

import torch

from evaluation.bayes_classifier import BayesClassifier
from losses.supervised_contrastive import SupervisedContrastiveLoss
from models.fusion.fusion_module import FusionModule
from models.gait_net import GaitNet
from models.resgcn.resgcn import ResGCN
from models.vggconv3d.vggconv3d import VGGConv3D

B, T, N, H, W = 4, 59, 17, 80, 80
EMBED = 128


def test_resgcn_forward():
    model = ResGCN(num_joints=N, embedding_size=EMBED)
    pose_seq = torch.randn(B, T, N, 3)
    out = model(pose_seq)
    assert out.shape == (B, EMBED), out.shape


def test_vggconv3d_forward():
    model = VGGConv3D(embedding_size=EMBED)
    clip = torch.randn(B, 3, T, H, W)
    out = model(clip)
    assert out.shape == (B, EMBED), out.shape


def test_fusion_forward():
    fusion = FusionModule(embedding_size=EMBED, fused_size=EMBED)
    skel = torch.randn(B, EMBED)
    vid = torch.randn(B, EMBED)
    out = fusion(skel, vid)
    assert out.shape == (B, EMBED), out.shape


def test_gait_net_end_to_end():
    model = GaitNet(embedding_size=EMBED, fused_size=EMBED)
    pose_seq = torch.randn(B, T, N, 3)
    clip = torch.randn(B, 3, T, H, W)
    out = model(pose_seq, clip)
    assert out.shape == (B, EMBED), out.shape
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-4), norms


def test_supervised_contrastive_loss():
    embeddings = torch.nn.functional.normalize(torch.randn(B, EMBED), dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    loss_fn = SupervisedContrastiveLoss(temperature=0.01)
    loss = loss_fn(embeddings, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_bayes_classifier():
    torch.manual_seed(0)
    num_classes = 5
    gallery_embeddings = torch.randn(num_classes * 4, EMBED)
    gallery_labels = torch.arange(num_classes).repeat_interleave(4)
    probe_embeddings = torch.randn(num_classes * 2, EMBED)
    probe_labels = torch.arange(num_classes).repeat_interleave(2)

    clf = BayesClassifier(diagonal_covariance=True)
    clf.fit(gallery_embeddings, gallery_labels)
    preds = clf.predict(probe_embeddings)
    assert preds.shape == probe_labels.shape
    acc = BayesClassifier.accuracy(preds, probe_labels)
    assert 0.0 <= acc <= 1.0


def test_gradients_flow_end_to_end():
    model = GaitNet(embedding_size=EMBED, fused_size=EMBED)
    pose_seq = torch.randn(B, T, N, 3)
    clip = torch.randn(B, 3, T, H, W)
    labels = torch.tensor([0, 0, 1, 1])
    loss_fn = SupervisedContrastiveLoss(temperature=0.01)

    out = model(pose_seq, clip)
    loss = loss_fn(out, labels)
    loss.backward()

    grad_found = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grad_found


if __name__ == "__main__":
    test_resgcn_forward()
    print("ResGCN forward: OK")
    test_vggconv3d_forward()
    print("VGGConv3D forward: OK")
    test_fusion_forward()
    print("Fusion forward: OK")
    test_gait_net_end_to_end()
    print("GaitNet end-to-end forward: OK")
    test_supervised_contrastive_loss()
    print("Supervised contrastive loss: OK")
    test_bayes_classifier()
    print("Bayes classifier: OK")
    test_gradients_flow_end_to_end()
    print("Gradients flow end-to-end: OK")
    print("\nAll smoke tests passed.")
