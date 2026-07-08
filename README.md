# Cross-Domain Forgetting Analysis in Test-Time Training for Trajectory Prediction

This repository extends [T4P (CVPR 2024)](https://arxiv.org/abs/2403.10052) with an analysis of catastrophic forgetting during test-time training (TTT) across autonomous driving domains, and investigates anti-forgetting strategies based on Learning without Forgetting (LwF).

## Overview

Test-time training adapts a pre-trained trajectory prediction model to a new target domain at inference time. While this improves target-domain performance, it causes **catastrophic forgetting** on the source domain — the model loses its original prediction ability after adaptation.

This work:
- Quantifies cross-domain forgetting under different source→target domain pairs (INTER→nuScenes, INTER→Lyft, etc.)
- Shows that output-level LwF distillation **harms** target adaptation when the teacher is weak on the target domain
- Proposes **feature-level LwF** (distilling encoder representations rather than trajectory outputs), achieving a strict Pareto improvement: source forgetting reduced by ~30% with only ~2.5% target performance cost
- Benchmarks GRPO (Group Relative Policy Optimization)-style reward-guided adaptation as an alternative anti-forgetting mechanism — all variants fail to match feature LwF

## Key Findings

| Method | Source MR (forgetting↓) | Target ADE6↓ |
|--------|------------------------|--------------|
| Baseline TTT (no anti-forgetting) | 0.365 (+21%) | — |
| Output-level LwF | degrades target | — |
| **Feature-level LwF** | **~0.319 (+6%)** | **0.585** |
| GRPO (all variants) | ~0.364 (+21%) | — |

Source domain: INTER. Target domain: nuScenes / Lyft.

## Installation

### Environment
```bash
conda env create --file env.yaml -n forecast_mae
conda activate forecast_mae
```

### Data loader
```bash
git clone https://github.com/daeheepark/trajdata-t4p unified-av-data-loader
```
Download raw datasets and follow the installation steps of the above repo. Dataset directory structure:
```
├── datasets
│   ├── nuScenes
│   ├── waymo
│   ├── interaction_single
│   ├── interaction_multi
│   └── lyft
```

## Usage

### Test-time training with feature-level LwF (recommended)
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config-name=config_test_inter13 \
  datamodule=inter_nus_13 \
  ttt_frequency=12 \
  lwf_feature_agent_weight=0.3 \
  save_adapted=true \
  desc=lwf_feat_inter2nus
```

### Key config options
| Parameter | Description |
|-----------|-------------|
| `lwf_weight` | Output-level LwF distillation weight (0 = disabled) |
| `lwf_feature_agent_weight` | Feature-level LwF on agent encoder (main knob) |
| `lwf_feature_lane_weight` | Feature-level LwF on lane encoder |
| `ttt_frequency` | Update every N steps (999999 = no TTT) |
| `save_adapted` | Save adapted model weights after TTT |

### Forgetting evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config-name=config_test_inter13 \
  datamodule=inter_13 \
  pretrained_weights=<path/to/adapted_model.ckpt> \
  ttt_frequency=999999 \
  desc=forgetting_check
```

## Branches

- `main` — feature-level LwF anti-forgetting
- `grpo-experiments` — GRPO-style reward-guided adaptation (experimental, did not outperform LwF)

## Acknowledgements

Built on top of [T4P](https://github.com/daeheepark/T4P) (Park et al., CVPR 2024), [ForecastMAE](https://arxiv.org/pdf/2308.09882.pdf), and [trajdata](https://arxiv.org/pdf/2307.13924.pdf).
