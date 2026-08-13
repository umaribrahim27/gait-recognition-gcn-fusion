# Gait Recognition — ResGCN + VGGConv3D + CBAM Fusion

Implementation of the multi-modal gait recognition architecture described
in the accompanying paper, `Gait_Recognition.pdf` (Sec. 3, equations 1–13,
and Sec. 4.2, equations 14–18 for the Bayes classifier).

## Status

Core equations (1–13) and the Bayes classifier (14–18) are implemented and
covered by an end-to-end smoke test (`tests/test_smoke.py`, 7/7 passing —
run with `python3 -m pytest tests/test_smoke.py -q`). Training loop,
CASIA-B data loading, and live HRNet inference are **not** implemented —
see "Not implemented / out of scope" below.

## Verification against the paper, equation by equation

| # | Item | Match | Notes |
|---|---|---|---|
| 1 | Graph conv layer (eq. 1) | **Matches** | `models/resgcn/graph_conv.py`: precomputes `D~^{-1/2} A~ D~^{-1/2}` with `A~ = A + I` in `data/graph.py:normalized_adjacency`, applies it via `einsum` before the learnt `Theta` (1×1 conv), then BN + activation, in that order. |
| 2 | Multi-branch inputs (eq. 2–5) | **Matches, with two resolved ambiguities** | `models/resgcn/branches.py` computes bone length (eq. 2), bone angle (eq. 3), motion velocity (eq. 4), and joint position (eq. 5) as three genuinely separate branches — bone (length+angle, 6ch), motion (6ch), joint (3ch) — each independently batch-normed and passed through its own graph-conv block *before* `concat_branches` implements eq. 6. Two details the paper leaves ambiguous were resolved and documented inline: (a) eq. 3 divides a 3-vector by a scalar norm then takes `arccos`, which we apply element-wise per axis (direction cosines); (b) eq. 5's "pose centre c" is undefined in the paper — we use the per-frame joint centroid. |
| 3 | Multi-head attention (eq. 7–9) | **Matches** | `models/resgcn/attention.py`: each of `P` heads has its own independent `W_Q`, `W_K`, `W_V` linear layers (`nn.ModuleList`, not shared weights), scaling is `1/sqrt(D)` with `D = proj_dim` per eq. 8, heads are concatenated and passed through one final linear `W_0` per eq. 9. |
| 4 | CBAM (eq. 10–12) | **Matches the paper's prose; extends the literal equations** | `models/vggconv3d/cbam.py`: `ChannelAttention` runs strictly before `SpatialAttention` (sequential, not parallel), matching Sec. 3.3.1. The literal eq. 10–12 only show an average-pooling path; Sec. 3.3.1's prose explicitly says both avg- and max-pooled features are used, so we implement the standard dual avg+max CBAM (the prose version is a superset of the equations, not a deviation from them). |
| 5 | Supervised contrastive loss (eq. 13) | **Matches** | `losses/supervised_contrastive.py`: `P(i)` is every other sample in the batch sharing `i`'s label (verified in `test_bayes_classifier`-adjacent unit test with multiple positives per anchor, e.g. labels `[0,0,1,1]` → each anchor has exactly one positive plus itself excluded), not a single positive/negative pair as in triplet loss. `A(i)` is all samples other than `i`. Implemented as a numerically stable log-sum-exp rather than the literal ratio-of-exponentials form, which is mathematically identical. |
| 6 | HRNet integration | **Not a placeholder, but not exercised** | `data/hrnet_extract.py` wraps OpenMMLab's `mmpose` (`init_model` / `inference_topdown`), i.e. the actual pretrained HRNet implementation and COCO checkpoint format referenced by the paper — it is not a from-scratch reimplementation. However, `mmpose`/`mmcv` were **not installed** and no checkpoint was **downloaded** in this session (no network fetch of model weights was performed, and there is no CASIA-B data present to run it on), so this module is untested here. Everything downstream only depends on its documented output contract: `(17, 3)` = `(x, y, confidence)` per frame. |

## Smoke test

`tests/test_smoke.py` runs random tensors of the paper's stated shapes
(`B=4, T=59, N=17`, video `80×80`, embedding size `128`) through:
ResGCN → VGGConv3D → FusionModule → full `GaitNet` → supervised
contrastive loss → backward pass → Bayes classifier fit/predict/accuracy.
All 7 checks pass, including a gradient-flow check confirming the whole
pipeline (both branches + fusion) is differentiable end to end.

```
python3 -m pytest tests/test_smoke.py -q
```

## Not implemented / out of scope (necessarily inferred or deferred)

These were not specified precisely enough in the paper to implement
without guessing, and are **not** part of the equation-by-equation
verification above:

- **CASIA-B dataset loading & gallery/probe split.** The paper states the
  protocol (74/50 subject split, NM 1-4 gallery, NM 5-6 / BG 1-2 / CL 1-2
  probes) but no loader is implemented — the dataset itself is not present
  in this environment.
- **CASIA-B / HRNet preprocessing specifics.** Augmentation order,
  padding strategy, and exact normalisation constants (mean/std for
  intensity normalisation) are described only qualitatively (Sec. 3.1);
  not implemented.
- **HRNet checkpoint.** See row 6 above — real integration code exists,
  but no checkpoint was downloaded or run.
- **Skeleton adjacency / bone-parent topology.** The paper says the
  skeleton graph and adjacency matrix are built via "spatial partitioning"
  but never enumerates the 17-joint edge list. `data/graph.py` uses a
  standard COCO-17 nose-rooted kinematic tree — a reasonable, but assumed,
  choice.
- **Main-stream channel widths / block counts** in `resgcn.py`, and the
  **conv-stack depth/channel widths** in `vggconv3d/conv_blocks.py`. The
  paper only describes these at a macro level ("basic and bottleneck
  blocks", "several convolutional layers"); the specific numbers used here
  are standard defaults, not values taken from the paper.
- **Fusion input granularity.** The paper's Fig. 1 suggests CBAM sits
  between the branches before final pooling, but doesn't give feature-map
  shapes at that junction. This implementation applies CBAM/fusion after
  each branch's own pooling stage (on `(B, 128)` embeddings), which is why
  `models/fusion/cbam_fusion.py` only implements the channel-gating half of
  CBAM (there's no spatial axis left at that point) — documented as an
  inferred simplification, not a value read off the paper.
- **Bayes classifier covariance.** Eq. 15 specifies a full covariance
  matrix `Σ_C`; with a 128-d embedding and few gallery samples per class
  this is singular, so `evaluation/bayes_classifier.py` defaults to a
  diagonal covariance for numerical stability (`diagonal_covariance=True`),
  a documented deviation from the literal equation.
- **Training loop, optimiser schedule, and full experiment reproduction**
  (Adam, 1-cycle LR, 200 epochs, batch sizes 512/128) are not implemented.

## Repo layout

```
data/            graph construction, HRNet adapter (not run), transforms (not implemented)
models/resgcn/   equations 1–9
models/vggconv3d/ equations 10–12, temporal max pool, GeM
models/fusion/   CBAM channel gate + scaled dot-product attention fusion
losses/          equation 13
evaluation/      equations 14–18
tests/           end-to-end smoke test
```
