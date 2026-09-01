#!/usr/bin/env python
"""
domain_discrim_analysis.py — Source vs target domain discriminativeness of
frozen encoder features (Task 5, diagnostic only).

No TTT is run here — the frozen source-pretrained model is probed once on
N_probe source (INTER) scenes and N_probe target (nuScenes) scenes, and the
per-dimension standardized mean difference (Cohen's d, pooled std) between
the two x_agent distributions is computed. Large-d dimensions are the ones
that most separate the two domains in encoder-feature space.

This is a diagnostic script — it does NOT produce a new selective-weighting
method by itself; it only characterizes domain separability, for comparison
against feature_drift_analysis.py's TTT-drift statistics.

Scene-sampling bias: extract_probe_features (reused from
feature_drift_analysis.py) records exactly one x_agent row per probe scene
(the scene's first time step), regardless of scene length, so pooling across
scenes of varying length does not bias the mean/std toward long scenes.

Outputs (in discrim_save_dir/):
  smd_bar.png             — top-20 dims by Cohen's d
  smd_vs_drift_scatter.png — Cohen's d vs TTT source-drift, only if
                             drift_data.npz exists at drift_data_path
  discrim_data.npz        — raw per-dim statistics

Usage (on the server):
  conda activate forecast_mae
  cd /home/ustb/T4P
  CUDA_VISIBLE_DEVICES=0 python domain_discrim_analysis.py \\
      --config-name=config_test_inter13 \\
      datamodule=inter_nus_13 \\
      probe_source=inter_13 \\
      n_probe=80 \\
      discrim_save_dir=outputs/domain_discrim
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from feature_drift_analysis import (
    extract_probe_features,
    _CAT_SRC, _CAT_TGT, _CAT_BOTH,
    _TEXT_PRI, _TEXT_SEC, _SURFACE, _GRID,
    _save,
)

EPS = 1e-8


def cohens_d(mean_src: np.ndarray, std_src: np.ndarray,
             mean_tgt: np.ndarray, std_tgt: np.ndarray) -> np.ndarray:
    """Standardized mean difference with pooled std: sqrt((s1^2+s2^2)/2)."""
    pooled = np.sqrt((std_src ** 2 + std_tgt ** 2) / 2.0)
    return np.abs(mean_src - mean_tgt) / (pooled + EPS)


def plot_discrim(cohens_d_vals: np.ndarray, save_dir: Path,
                  drift_src: Optional[np.ndarray]) -> None:
    D = cohens_d_vals.shape[0]
    TOP_K = min(20, D)
    sort_idx = np.argsort(cohens_d_vals)[::-1]
    top_dims = sort_idx[:TOP_K]

    # ── 1. Top-K Cohen's d bar chart (single series → one hue) ────────────────
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    x = np.arange(TOP_K)
    ax.bar(x, cohens_d_vals[top_dims], width=0.6, color=_CAT_BOTH, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f'd{d}' for d in top_dims], fontsize=7.5,
                        color=_TEXT_SEC, rotation=45, ha='right')
    ax.set_ylabel("Cohen's d (pooled std)", color=_TEXT_SEC, fontsize=10)
    ax.set_title(f"Top-{TOP_K} most domain-discriminative encoder dimensions",
                 color=_TEXT_PRI, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', color=_GRID, linewidth=0.7, alpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_edgecolor(_GRID)
    ax.tick_params(colors=_TEXT_SEC)
    plt.tight_layout()
    _save(fig, save_dir / 'smd_bar.png')

    # ── 2. Cohen's d vs TTT source-drift scatter (only if drift data exists) ──
    if drift_src is not None and drift_src.shape[0] == D:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=_SURFACE)
        ax.set_facecolor(_SURFACE)
        ax.scatter(cohens_d_vals, drift_src, alpha=0.45, s=22,
                   color=_CAT_BOTH, edgecolors='none')
        for d in top_dims[:8]:
            ax.annotate(f'd{d}', (cohens_d_vals[d], drift_src[d]),
                        fontsize=6.5, color=_TEXT_SEC, ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points')
        ax.set_xlabel("Cohen's d (domain discriminativeness)",
                      color=_TEXT_SEC, fontsize=10)
        ax.set_ylabel('TTT source drift |Δmean|', color=_TEXT_SEC, fontsize=10)
        ax.set_title("Discriminativeness vs TTT drift, per dimension",
                     color=_TEXT_PRI, fontsize=11, fontweight='bold')
        ax.grid(True, color=_GRID, linewidth=0.7, alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_edgecolor(_GRID)
        ax.tick_params(colors=_TEXT_SEC)
        plt.tight_layout()
        _save(fig, save_dir / 'smd_vs_drift_scatter.png')
    else:
        print('[discrim] drift_data.npz not found or dim mismatch — '
              'skipping smd_vs_drift_scatter.png')


@hydra.main(version_base=None, config_path="conf", config_name="config_test_ttt")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)

    save_dir = Path(getattr(conf, 'discrim_save_dir', 'outputs/domain_discrim'))
    save_dir.mkdir(parents=True, exist_ok=True)

    n_probe = int(getattr(conf, 'n_probe', 80))
    probe_source = str(getattr(conf, 'probe_source', 'inter_13'))
    drift_data_path = str(getattr(conf, 'drift_data_path',
                                   'outputs/drift_analysis/drift_data.npz'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[discrim] device={device}  n_probe={n_probe}  probe_source={probe_source}')

    model = instantiate(conf.model.target)
    model.net.load_from_checkpoint(conf.pretrained_weights)
    model = model.to(device)
    model.eval()
    model.freeze_layers(conf)

    type_embed_table = model.net.actor_type_embed.detach().clone()

    target_dm = instantiate(conf.datamodule)
    target_dm.setup()

    src_dm_yaml = OmegaConf.load(f'conf/datamodule/{probe_source}.yaml')
    src_conf = OmegaConf.merge(conf, {'datamodule': src_dm_yaml})
    src_dm = instantiate(src_conf.datamodule)
    src_dm.setup()

    print(f'[discrim] Probing {n_probe} source scenes ...')
    feat_src = extract_probe_features(model, src_dm, n_probe, type_embed_table, device)
    print(f'[discrim] Probing {n_probe} target scenes ...')
    feat_tgt = extract_probe_features(model, target_dm, n_probe, type_embed_table, device)
    print(f'[discrim] x_agent shape: src={feat_src.shape} tgt={feat_tgt.shape}')

    mean_src, std_src = feat_src.mean(0), feat_src.std(0)
    mean_tgt, std_tgt = feat_tgt.mean(0), feat_tgt.std(0)
    d_vals = cohens_d(mean_src, std_src, mean_tgt, std_tgt)

    drift_src = None
    drift_path = Path(drift_data_path)
    if drift_path.exists():
        drift_src = np.load(drift_path)['drift_src'][-1]
        print(f'[discrim] Loaded drift reference from {drift_path}')
    else:
        print(f'[discrim] {drift_path} not found — scatter plot will be skipped')

    plot_discrim(d_vals, save_dir, drift_src)

    out = save_dir / 'discrim_data.npz'
    np.savez(str(out), mean_src=mean_src, std_src=std_src,
              mean_tgt=mean_tgt, std_tgt=std_tgt, cohens_d=d_vals,
              sort_idx=np.argsort(d_vals)[::-1])
    print(f'[discrim] {out}')

    top5 = np.argsort(d_vals)[::-1][:5]
    print('\n[discrim] Top-5 domain-discriminative dims:')
    for d in top5:
        print(f'  dim {d:3d}: cohens_d={d_vals[d]:.4f}  '
              f'mean_src={mean_src[d]:.4f} mean_tgt={mean_tgt[d]:.4f}')
    print(f'[discrim] Mean cohens_d over all dims: {d_vals.mean():.4f}')
    print(f'\n[discrim] Done. Outputs in: {save_dir}')


if __name__ == '__main__':
    main()
