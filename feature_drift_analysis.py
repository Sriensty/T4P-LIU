#!/usr/bin/env python
"""
feature_drift_analysis.py — Encoder feature drift during unconstrained TTT.

Runs unconstrained TTT on the target domain (nuScenes) while periodically
probing encoder representations (x_agent, 128-dim) on FIXED probe scenes
from both source and target. Generates four drift plots to identify which
encoder dimensions drift the most — the data foundation for selective LwF.

Outputs (in drift_save_dir/):
  drift_heatmap.png      — dim × checkpoint (blue sequential, sorted by source drift)
  drift_curve.png        — aggregate drift over TTT: source vs target
  top_dims_drift.png     — top-20 most drifting dims, source vs target bars
  src_vs_tgt_scatter.png — per-dim: source drift vs target drift
  drift_data.npz         — raw arrays for further analysis

Usage (on the server):
  conda activate forecast_mae
  cd /home/ustb/T4P
  CUDA_VISIBLE_DEVICES=0 python feature_drift_analysis.py \\
      --config-name=config_test_inter13 \\
      datamodule=inter_nus_13 \\
      ttt_frequency=12 \\
      probe_source=inter_13 \\
      n_probe=80 \\
      snapshot_every=400 \\
      drift_save_dir=outputs/drift_analysis
"""

import os
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

MAX_STEP = 10000

# ── Palette (dataviz skill: blue sequential for heatmap; cat. slots 1+3 for
#    source/target comparison; slot 7 for single-series scatter) ──────────────
_SEQ_BLUE = ['#cde2fb','#9ec5f4','#6da7ec','#3987e5',
             '#2a78d6','#256abf','#1c5cab','#0d366b']
_CAT_SRC  = '#2a78d6'   # slot 1 blue    — source domain
_CAT_TGT  = '#e87ba4'   # slot 3 magenta — target domain
_CAT_BOTH = '#4a3aa7'   # slot 7 violet  — per-dim scatter (single series)
_TEXT_PRI = '#0b0b0b'
_TEXT_SEC = '#52514e'
_SURFACE  = '#fcfcfb'
_GRID     = '#e1e0d9'
BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list('drift_blue', _SEQ_BLUE)


# ── Feature probe ─────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_probe_features(model, datamodule, n_probe: int,
                           type_embed_table, device) -> np.ndarray:
    """
    Eval-mode forward pass on up to n_probe distinct scenes.
    Actor embeds are freshly initialised per scene from type_embed_table
    (so we measure encoder weight drift, not per-scene embed drift).
    Returns np.ndarray [N, D] — one x_agent row per probe scene.
    """
    model.eval()
    feats, SCENE_ID, count = [], None, 0

    for batch in datamodule.test_dataloader():
        if count >= n_probe:
            break
        batch = {k: v.to(device) if hasattr(v, 'to') else v
                 for k, v in batch.items()}
        sid = batch['scenario_id'][0]

        if sid != SCENE_ID:
            SCENE_ID = sid
            count += 1
            if count > n_probe:
                break
            actor_names = batch['actor_names'][0]
            embeds = type_embed_table[batch['x_attr'][0, :, 0].long()].detach().clone()
            model.net.actor_embeds = torch.nn.ParameterDict(
                {actor_names[i]: torch.nn.Parameter(embeds[i])
                 for i in range(len(actor_names))}
            )
            # Only record the first time-step of each new scene
            out = model.net.forward_forecast_peragent_fre(batch)
            feats.append(out['x_agent'][0].detach().cpu().numpy())
        else:
            # Register any new actors that appear mid-scene
            registered = set(model.net.actor_embeds.keys())
            for new_actor in set(batch['actor_names'][0]) - registered:
                idx = batch['actor_names'][0].index(new_actor)
                a_type = batch['x_attr'][0, :, 0].long()[idx]
                model.net.actor_embeds.update(
                    {new_actor: torch.nn.Parameter(
                        type_embed_table[a_type].detach().clone())}
                )

    return np.stack(feats) if feats else np.zeros((1, 128))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_drift_analysis(snapshots_src, snapshots_tgt,
                        step_labels, save_dir: Path) -> None:
    pre_src = snapshots_src[0]          # [N, D]
    pre_tgt = snapshots_tgt[0]          # [N, D]
    T, D    = len(snapshots_src), pre_src.shape[1]

    # Per-dimension mean shift from pre-TTT baseline
    drift_src = np.array([np.abs(s.mean(0) - pre_src.mean(0))
                          for s in snapshots_src])   # [T, D]
    drift_tgt = np.array([np.abs(s.mean(0) - pre_tgt.mean(0))
                          for s in snapshots_tgt])   # [T, D]

    # Sort dims by final source drift (descending) for heatmap readability
    sort_idx = np.argsort(drift_src[-1])[::-1]

    vmax = max(drift_src.max(), drift_tgt.max())

    # ── 1. Drift heatmap ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             facecolor=_SURFACE, sharey=True)
    for ax, drift, title in zip(
            axes,
            [drift_src[:, sort_idx], drift_tgt[:, sort_idx]],
            ['Source (INTER)', 'Target (nuScenes)']):
        im = ax.imshow(drift.T, aspect='auto', cmap=BLUE_CMAP,
                       vmin=0, vmax=vmax, origin='upper',
                       interpolation='nearest')
        ax.set_facecolor(_SURFACE)
        ax.set_xlabel('Checkpoint', color=_TEXT_SEC, fontsize=11)
        ax.set_ylabel('Feature dimension (sorted by source drift)',
                      color=_TEXT_SEC, fontsize=11)
        ax.set_title(f'Encoder drift — {title}',
                     color=_TEXT_PRI, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(range(T))
        ax.set_xticklabels(step_labels, rotation=40, ha='right',
                           fontsize=8, color=_TEXT_SEC)
        ax.tick_params(colors=_TEXT_SEC)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRID)
        cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
        cb.set_label('|Δ mean x_agent|', color=_TEXT_SEC, fontsize=9)
        cb.ax.yaxis.set_tick_params(color=_TEXT_SEC, labelcolor=_TEXT_SEC)

    fig.suptitle('x_agent encoder feature drift during unconstrained TTT\n'
                 '(dims sorted by post-TTT source drift, descending)',
                 color=_TEXT_PRI, fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, save_dir / 'drift_heatmap.png')

    # ── 2. Aggregate drift curve ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    xs = range(T)
    ax.plot(xs, drift_src.mean(1), 'o-', color=_CAT_SRC, lw=2, ms=7,
            label='Source (INTER)')
    ax.plot(xs, drift_tgt.mean(1), 's-', color=_CAT_TGT, lw=2, ms=7,
            label='Target (nuScenes)')
    ax.set_xticks(xs)
    ax.set_xticklabels(step_labels, rotation=30, ha='right',
                       fontsize=9, color=_TEXT_SEC)
    ax.set_ylabel('Mean |Δ mean x_agent| over all dims',
                  color=_TEXT_SEC, fontsize=10)
    ax.set_title('Aggregate encoder drift over TTT',
                 color=_TEXT_PRI, fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=10, labelcolor=_TEXT_PRI)
    ax.grid(True, color=_GRID, linewidth=0.7, alpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_edgecolor(_GRID)
    ax.tick_params(colors=_TEXT_SEC)
    plt.tight_layout()
    _save(fig, save_dir / 'drift_curve.png')

    # ── 3. Top-20 dims bar chart ───────────────────────────────────────────────
    TOP_K    = 20
    top_dims = sort_idx[:TOP_K]
    final_src = drift_src[-1, top_dims]
    final_tgt = drift_tgt[-1, top_dims]

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    x, w = np.arange(TOP_K), 0.38
    ax.bar(x - w/2, final_src, width=w, color=_CAT_SRC,
           label='Source (INTER)', linewidth=0)
    ax.bar(x + w/2, final_tgt, width=w, color=_CAT_TGT,
           label='Target (nuScenes)', linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f'd{d}' for d in top_dims],
                       fontsize=7.5, color=_TEXT_SEC, rotation=45, ha='right')
    ax.set_ylabel('|Δ mean x_agent| (post-TTT)', color=_TEXT_SEC, fontsize=10)
    ax.set_title(f'Top-{TOP_K} most drifting encoder dimensions (source-ranked)',
                 color=_TEXT_PRI, fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=10, labelcolor=_TEXT_PRI)
    ax.grid(True, axis='y', color=_GRID, linewidth=0.7, alpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_edgecolor(_GRID)
    ax.tick_params(colors=_TEXT_SEC)
    plt.tight_layout()
    _save(fig, save_dir / 'top_dims_drift.png')

    # ── 4. Source vs target scatter ───────────────────────────────────────────
    final_src_all = drift_src[-1]
    final_tgt_all = drift_tgt[-1]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.scatter(final_src_all, final_tgt_all, alpha=0.45, s=22,
               color=_CAT_BOTH, edgecolors='none')
    for d in sort_idx[:8]:
        ax.annotate(f'd{d}', (final_src_all[d], final_tgt_all[d]),
                    fontsize=6.5, color=_TEXT_SEC, ha='center', va='bottom',
                    xytext=(0, 3), textcoords='offset points')
    lim = max(final_src_all.max(), final_tgt_all.max()) * 1.12
    ax.plot([0, lim], [0, lim], '--', color=_GRID, lw=1.2, label='y = x')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('Source domain drift |Δmean|', color=_TEXT_SEC, fontsize=10)
    ax.set_ylabel('Target domain drift |Δmean|', color=_TEXT_SEC, fontsize=10)
    ax.set_title('Per-dimension: source vs target drift\n'
                 '(above diagonal → target drifts more than source)',
                 color=_TEXT_PRI, fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=9, labelcolor=_TEXT_SEC)
    ax.grid(True, color=_GRID, linewidth=0.7, alpha=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_edgecolor(_GRID)
    ax.tick_params(colors=_TEXT_SEC)
    plt.tight_layout()
    _save(fig, save_dir / 'src_vs_tgt_scatter.png')

    # ── 5. Raw data ────────────────────────────────────────────────────────────
    out = save_dir / 'drift_data.npz'
    np.savez(str(out),
             drift_src=drift_src,
             drift_tgt=drift_tgt,
             snapshots_src=np.stack(snapshots_src),
             snapshots_tgt=np.stack(snapshots_tgt),
             sort_idx=sort_idx,
             step_labels=np.array(step_labels, dtype=object))
    print(f'[plot] {out}')


def _save(fig, path: Path) -> None:
    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor=_SURFACE)
    plt.close(fig)
    print(f'[plot] {path}')


# ── Main TTT loop with periodic feature probes ───────────────────────────────

@hydra.main(version_base=None, config_path="conf",
            config_name="config_test_ttt")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)

    save_dir = Path(getattr(conf, 'drift_save_dir', 'outputs/drift_analysis'))
    save_dir.mkdir(parents=True, exist_ok=True)

    n_probe        = int(getattr(conf, 'n_probe',        80))
    snapshot_every = int(getattr(conf, 'snapshot_every', 400))
    probe_source   = str(getattr(conf, 'probe_source',   'inter_13'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[drift] device={device}  n_probe={n_probe}  '
          f'snapshot_every={snapshot_every}  probe_source={probe_source}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = instantiate(conf.model.target)
    model.net.load_from_checkpoint(conf.pretrained_weights)
    model = model.to(device)
    model.eval()
    model.freeze_layers(conf)

    # Fixed clone of actor type embed table; won't change when update_type_embed=False
    actor_tyme_embed_clone = deepcopy(
        model.net.actor_type_embed.detach()
    )

    optimizers, _ = model.configure_ttt_optimizers(conf)  # optimizers[0] = model opt

    # ── Datamodules ───────────────────────────────────────────────────────────
    target_dm = instantiate(conf.datamodule)
    target_dm.setup()

    # Merge source dm yaml into conf so ${batch_size} etc. resolve correctly
    src_dm_yaml = OmegaConf.load(f'conf/datamodule/{probe_source}.yaml')
    src_conf    = OmegaConf.merge(conf, {'datamodule': src_dm_yaml})
    src_dm      = instantiate(src_conf.datamodule)
    src_dm.setup()

    # ── Phase 1: pre-TTT probe ────────────────────────────────────────────────
    type_embed_snap = actor_tyme_embed_clone.clone()
    print(f'\n[drift] Pre-TTT: probing {n_probe} source scenes ...')
    feat_src_pre = extract_probe_features(model, src_dm, n_probe,
                                          type_embed_snap, device)
    print(f'[drift] Pre-TTT: probing {n_probe} target scenes ...')
    feat_tgt_pre = extract_probe_features(model, target_dm, n_probe,
                                          type_embed_snap, device)
    print(f'[drift] x_agent shape: {feat_src_pre.shape}  (scenes × embed_dim)')

    snapshots_src = [feat_src_pre]
    snapshots_tgt = [feat_tgt_pre]
    step_labels   = ['pre-TTT']

    # ── Phase 2: unconstrained TTT (reg + mae, no LwF / OPD) ─────────────────
    fut_masks = torch.ones(
        (conf.model.target.future_steps, conf.model.target.future_steps),
        device=device, dtype=torch.bool)
    fut_masks = torch.flip(torch.tril(fut_masks), dims=(0,))

    SCENE_ID       = None
    actor_ns_scene = []     # actor names seen in the current scene
    actor_ts_scene = []     # actor types for each of the above
    output_mae_    = {}
    test_batch_    = {}
    register_maeoutput = 0
    bi_passed  = 0
    ttt_step   = 0
    optimizer1 = None

    print(f'\n[drift] Starting unconstrained TTT (ttt_frequency={conf.ttt_frequency}) ...')

    _SKIP_ACCUM = {'x_agent', 'x_encoder_deep'}

    for bi, test_batch in enumerate(tqdm(target_dm.test_dataloader())):
        if bi_passed > MAX_STEP:
            break

        test_batch = {k: v.to(device) if hasattr(v, 'to') else v
                      for k, v in test_batch.items()}
        scenario_id = test_batch['scenario_id'][0]

        # ── Actor embeds management (mirrors test.py) ─────────────────────────
        if scenario_id != SCENE_ID:
            output_mae_ = {}
            test_batch_ = {}
            actor_ns_scene = []
            actor_ts_scene = []
            SCENE_ID = scenario_id

            if bi_passed == 0:
                actor_names = test_batch['actor_names'][0]
                embeds = actor_tyme_embed_clone[
                    test_batch['x_attr'][0, :, 0].long()]
                model.net.actor_embeds = torch.nn.ParameterDict(
                    {actor_names[i]: torch.nn.Parameter(embeds[i])
                     for i in range(len(actor_names))})
                actor_ns_scene += list(actor_names)
                actor_ts_scene += test_batch['x_attr'][0, :, 0].long().cpu().tolist()
            else:
                if conf.update_type_embed:
                    for a_type in torch.unique(torch.tensor(actor_ts_scene)):
                        mask  = torch.tensor(actor_ts_scene) == a_type
                        names = [actor_ns_scene[i]
                                 for i in mask.nonzero()[:, 0].tolist()]
                        avg   = torch.stack(
                            [model.net.actor_embeds[n] for n in names]).mean(0)
                        actor_tyme_embed_clone[a_type] = avg.detach()

                del model.net.actor_embeds
                actor_names = test_batch['actor_names'][0]
                embeds = actor_tyme_embed_clone[
                    test_batch['x_attr'][0, :, 0].long()]
                model.net.actor_embeds = torch.nn.ParameterDict(
                    {actor_names[i]: torch.nn.Parameter(embeds[i])
                     for i in range(len(actor_names))})
                actor_ns_scene += list(actor_names)
                actor_ts_scene += test_batch['x_attr'][0, :, 0].long().cpu().tolist()

            optimizer1 = torch.optim.AdamW(
                model.net.actor_embeds.parameters(),
                lr=model.lr2, weight_decay=model.weight_decay2)
        else:
            registered = set(model.net.actor_embeds.keys())
            for new_actor in set(test_batch['actor_names'][0]) - registered:
                idx    = test_batch['actor_names'][0].index(new_actor)
                a_type = test_batch['x_attr'][0, :, 0].long()[idx]
                actor_ns_scene.append(new_actor)
                actor_ts_scene.append(a_type.item())
                p = torch.nn.Parameter(actor_tyme_embed_clone[a_type])
                model.net.actor_embeds.update({new_actor: p})
                optimizer1.add_param_group(
                    {'params': p, 'weight_decay': model.weight_decay2})

        optimizers[0].zero_grad()
        optimizer1.zero_grad()

        # ── Forward ───────────────────────────────────────────────────────────
        output_forecast = model.net.forward_forecast_peragent_fre(test_batch)
        output_mae      = model.net.forward_mae_fre(test_batch, output_forecast)
        output_mae.update(output_forecast)

        if register_maeoutput % conf.ttt_real_freq == 0:
            if len(output_mae_) == 0:
                output_mae_.update(
                    {k: v for k, v in output_mae.items()
                     if k not in _SKIP_ACCUM})
                test_batch_.update(test_batch)
            else:
                for key in list(output_mae_.keys()):
                    if key in _SKIP_ACCUM:
                        continue
                    fs = conf.model.target.future_steps
                    if output_mae_[key].size(0) < fs:
                        output_mae_[key] = pad_sequence(
                            [*output_mae_[key], output_mae[key][0]],
                            batch_first=True)
                    else:
                        output_mae_[key] = pad_sequence(
                            [*output_mae_[key][1:], output_mae[key][0]],
                            batch_first=True)
                skip_tb = {'num_actors','num_lanes','scenario_id',
                           'scene_ts','track_id','origin','theta','actor_names'}
                for key in list(test_batch_.keys()):
                    if key in skip_tb:
                        continue
                    fs = conf.model.target.future_steps
                    if test_batch_[key].size(0) < fs:
                        test_batch_[key] = pad_sequence(
                            [*test_batch_[key], test_batch[key][0]],
                            batch_first=True)
                    else:
                        test_batch_[key] = pad_sequence(
                            [*test_batch_[key][1:], test_batch[key][0]],
                            batch_first=True)

        # ── TTT update ────────────────────────────────────────────────────────
        if bi_passed != 0 and bi % conf.ttt_frequency == 0 \
                and len(output_mae_) != 0:
            length = output_mae_['y_hat'].shape[0]
            obs_fut_mask = fut_masks[-length:]
            losses = model.cal_loss_fre_obs(
                output_mae_, test_batch_, obs_fut_mask)
            loss = losses['reg_loss'] + losses['mae_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), conf.gradient_clip_val)
            optimizers[0].step()
            optimizer1.step()
            ttt_step  += 1
            output_mae_ = {}
            test_batch_ = {}

            # ── Snapshot ──────────────────────────────────────────────────────
            if ttt_step % snapshot_every == 0:
                print(f'\n[drift] Snapshot @ TTT step {ttt_step} ...')

                # Save current actor_embeds; restore after probing so TTT
                # can continue within the same scene without mismatched actors.
                saved_embeds = deepcopy(model.net.actor_embeds)
                te_snap      = actor_tyme_embed_clone.clone()

                snapshots_src.append(
                    extract_probe_features(model, src_dm, n_probe,
                                          te_snap, device))
                snapshots_tgt.append(
                    extract_probe_features(model, target_dm, n_probe,
                                          te_snap, device))
                step_labels.append(f'step {ttt_step}')

                # Restore actor_embeds + rebuild optimizer1
                model.net.actor_embeds = saved_embeds
                optimizer1 = torch.optim.AdamW(
                    model.net.actor_embeds.parameters(),
                    lr=model.lr2, weight_decay=model.weight_decay2)
                model.train()

        register_maeoutput += 1
        bi_passed += 1

    # ── Final snapshot ─────────────────────────────────────────────────────────
    print(f'\n[drift] Final probe @ {ttt_step} TTT steps ...')
    te_snap = actor_tyme_embed_clone.clone()
    snapshots_src.append(
        extract_probe_features(model, src_dm, n_probe, te_snap, device))
    snapshots_tgt.append(
        extract_probe_features(model, target_dm, n_probe, te_snap, device))
    step_labels.append(f'post-TTT\n({ttt_step} steps)')

    # ── Phase 3: plot ──────────────────────────────────────────────────────────
    print(f'\n[drift] Plotting {len(snapshots_src)} snapshots ...')
    plot_drift_analysis(snapshots_src, snapshots_tgt, step_labels, save_dir)

    # Summary
    d_src = np.abs(snapshots_src[-1].mean(0) - snapshots_src[0].mean(0))
    d_tgt = np.abs(snapshots_tgt[-1].mean(0) - snapshots_tgt[0].mean(0))
    top5  = np.argsort(d_src)[::-1][:5]
    print('\n[drift] Top-5 source-drifting dims:')
    for d in top5:
        print(f'  dim {d:3d}: src Δ={d_src[d]:.4f}  tgt Δ={d_tgt[d]:.4f}'
              f'  (ratio tgt/src={d_tgt[d]/max(d_src[d],1e-6):.2f})')
    print(f'[drift] Mean src drift: {d_src.mean():.4f}'
          f'  |  mean tgt drift: {d_tgt.mean():.4f}')
    print(f'\n[drift] Done. Outputs in: {save_dir}')


if __name__ == '__main__':
    main()
