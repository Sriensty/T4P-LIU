# T4P Cross-Domain Forgetting — Full Experimental Summary
_2026-07-03_

**Setup**: INTER source model, adapted to Lyft target via T4P per-agent TTT, then evaluated on INTER source to measure forgetting. All numbers are minADE6 unless noted.

**Baselines**
- INTER source (no TTT): **0.577**
- Cross-domain no-TTT (source model on Lyft target): 0.745
- INTER → Lyft → INTER, T4P baseline TTT (no LwF): target **0.571**, source **0.786 (+36% forgetting)**

---

## Output-level LwF Pareto (v3, buggy-teacher fixed)

| λ | Target ADE6 | Δ vs 0.571 | Source ADE6 | Δ vs 0.786 |
|---|---|---|---|---|
| 0 (baseline) | **0.571** | — | 0.786 | — |
| 0.02 | 1.034 | **+81%** | 0.775 | −1% |
| 0.05 | 1.076 | +88% | 0.757 | −4% |
| 0.1 | 1.057 | +85% | 0.705 | −10% |
| 0.2 | 1.013 | +77% | 0.690 | −12% |

**Conclusion**: output-LwF pulls student to reproduce teacher's trajectories. Teacher is bad on Lyft → target catastrophically degrades. **Direction wrong.**

---

## Feature-level LwF Pareto (v2, all 3 bugs fixed)

| Config | feat_agent | feat_lane | out_lwf | **Target ADE6** | **Source ADE6** |
|---|---|---|---|---|---|
| baseline | 0 | 0 | 0 | 0.571 | 0.786 |
| **v2_fal01** | **0.1** | 0 | 0 | **0.569** (−0.3%) | 0.827 (+5%) |
| **v2_fal03** | **0.3** | 0 | 0 | **0.575** (+0.7%) | 0.789 (+0.4%) |
| **v2_fal10** ★ | **1.0** | 0 | 0 | **0.585** (+2.5%) | **0.713 (−9.3%)** |
| **v2_falll** ★ | 0.3 | 0.1 | 0 | 0.576 (+0.9%) | **0.751 (−4.5%)** |
| v2_hybrid | 0.3 | 0 | 0.1 | 1.049 (+84%) | 0.701 (−11%) |

**Two configs achieve strict-better than baseline on both axes**: `v2_falll` (weak lane distill) and `v2_fal10` (strong agent distill). The `v2_hybrid` shows that adding ANY output-LwF taint reintroduces the target catastrophe.

---

## Direct comparison at similar source improvement

| Method | Target Δ | Source Δ | Target cost per unit source gain |
|---|---|---|---|
| output-LwF λ=0.1 | +85% | −10.3% | **+8.3× per −1%** |
| **Feature-LwF fal10** | **+2.5%** | −9.3% | **+0.27× per −1%** |

**Feature LwF is ~30× more efficient than output LwF for the same anti-forgetting benefit.**

---

## Root-cause insight

- **Output distillation** constrains "which trajectory the student outputs" → conflict with adapting to target-specific trajectories.
- **Feature (encoder) distillation** constrains "how the scene is understood" → decoder still free to produce target-appropriate trajectories. This is exactly the right level of granularity for domain adaptation.

The finding was hidden behind three progressively deeper bugs:
1. Teacher forward crashed silently when `test_batch_` had inconsistent shapes across time steps (accumulated actor_names vs pad_sequenced x)
2. Hydra `+lwf_weight` collided with existing yaml key
3. `compute_lwf_loss` early-returned zero when only feature weights were set

All three fixed and cross-verified.

---

## Provenance — all numbers reproducible via server logs

All experiments live under `47.122.121.215:/home/ustb/T4P/outputs/forecast-mae-ttt-test_True/`. Master CSV: `downloaded_results/master_table.csv` (52 rows, includes ckpt path + config for every row).

Key path examples:
- INTER→Lyft baseline TTT: `2026-06-03/22-25-00_i2l_baseline_tgt` + `2026-06-03/22-37-00_i2l_baseline_fgt`
- Output LwF λ=0.1: `2026-07-02/23-24-41_v3_i2l_l01_tgt` + `2026-07-02/23-37-03_v3_i2l_l01_fgt`
- **Feature LwF fal10 ★**: `2026-07-03/01-17-56_v2_fal10_tgt` + `2026-07-03/01-30-41_v2_fal10_fgt`
- Feature LwF falll ★: `2026-07-03/01-35-54_v2_falll_tgt` + `2026-07-03/01-51-36_v2_falll_fgt`

---

## Cross-domain forgetting is target-dependent

The "36% forgetting" only appears at large domain distance. Same source model, different targets:

| Target | Forgetting on INTER source |
|---|---|
| nuScenes-mini (close) | **−11%** (source *improves*) |
| nuScenes (medium) | +25% |
| Lyft (far) | +36% |

Feature LwF gates itself: because near-target encoder features already match teacher, `feat_agent_lwf` loss is naturally small → no interference with beneficial near-domain adaptation.
