#!/usr/bin/env bash
# =============================================================================
# sweep_lwf.sh — LwF + long-horizon TTT sweep for T4P
#
# What it does (one cross-domain pair, configurable via env vars):
#   (0) src_baseline    : eval pretrained on SOURCE, no TTT
#   (0) tgt_noTTT       : eval pretrained on TARGET, no TTT
#   (1..5) per method   : run TTT on TARGET (saves adapted ckpt),
#                         then load adapted ckpt and eval on SOURCE
#                         (= "forgetting" check)
#
# Methods swept (matches my recommendation):
#   baseline       lwf=0.0  pi=0.0   gamma=0.0   ← pure T4P
#   horizon_only   lwf=0.0  pi=0.0   gamma=2.0   ← isolate long-horizon weighting
#   lwf_only       lwf=0.3  pi=0.05  gamma=0.0   ← isolate LwF
#   lwf_horizon    lwf=0.3  pi=0.05  gamma=2.0   ← main combo (table headline)
#   lwf_strong     lwf=1.0  pi=0.10  gamma=2.0   ← upper-bound LwF
#
# Override defaults from the command line, e.g.:
#   PRETRAINED=path/to/source.ckpt \
#   SRC_DATAMODULE=nus  TGT_DATAMODULE=nus_lyft  MODEL=model_ttt \
#   bash scripts/sweep_lwf.sh
# =============================================================================
set -euo pipefail

# ---- Required: pretrained source weights (override via env) -----------------
PRETRAINED="${PRETRAINED:-}"
if [[ -z "${PRETRAINED}" ]]; then
  echo "[sweep] ERROR: set PRETRAINED=path/to/source/checkpoint.ckpt" >&2
  echo "[sweep] Example:" >&2
  echo "  PRETRAINED=outputs/.../checkpoints/epoch=23.ckpt bash $0" >&2
  exit 1
fi
if [[ ! -f "${PRETRAINED}" ]]; then
  echo "[sweep] ERROR: PRETRAINED not found: ${PRETRAINED}" >&2
  exit 1
fi

# ---- Configurable: cross-domain pair + model variant ------------------------
SRC_DATAMODULE="${SRC_DATAMODULE:-nus}"            # source-only eval (no TTT)
TGT_DATAMODULE="${TGT_DATAMODULE:-nus_lyft_sample}" # target adaptation (with TTT) — matches your existing _lt runs
MODEL="${MODEL:-model_ttt}"                        # model_ttt (50/60) or model_ttt_13 (10/30)
TTT_FREQ_ON="${TTT_FREQ_ON:-12}"                   # the default T4P frequency
TTT_FREQ_OFF="${TTT_FREQ_OFF:-999999}"             # >=10000 disables TTT (no_grad path)

# ---- Output bookkeeping ------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${T4P_SWEEP_DIR:-sweep_${TS}}"
mkdir -p "${SWEEP_DIR}"
INDEX="${SWEEP_DIR}/runs.tsv"
printf "run_id\tphase\tmethod\tlwf\tpi\tgamma\toutput_dir\n" > "${INDEX}"
echo "[sweep] writing index to: ${INDEX}"

# Helper: run a single experiment, capture Hydra output_dir from logs ----------
run_exp () {
  local label="$1"; shift
  local desc="$1"; shift
  local logf="${SWEEP_DIR}/_stdout_${label}.log"
  echo "------------------------------------------------------------"
  echo "[sweep] launching: ${label}  desc=${desc}"
  echo "[sweep]   args: $*"
  # tee both stdout/stderr to a file so we can recover the Hydra output_dir.
  # NOTE: callers must pre-escape "=" in any ckpt path values (see below).
  python test.py "$@" "desc=${desc}" 2>&1 | tee "${logf}"
  # Hydra prints the output dir line like:
  #   "[YYYY-MM-DD HH:MM:SS,ms][hydra][INFO] - Working directory: <path>"
  # but the easier signal in test.py is:
  #   "Result of exp <basename>" (which we can map via desc)
  # so we just glob the latest outputs/.../<HH-MM-SS_${desc}>/ dir.
  local outdir
  outdir=$(find outputs -maxdepth 4 -type d -name "*_${desc}" 2>/dev/null | sort | tail -1 || true)
  echo "${label}\t-\t-\t-\t-\t-\t${outdir}" >> "${INDEX}.tmp_${label}" || true
  echo "[sweep] output dir: ${outdir}"
  echo "${outdir}"
}

# Helper: write index row ------------------------------------------------------
index_row () {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${INDEX}"
}

# =============================================================================
# (0) Background runs — done once, shared by all methods
# =============================================================================
SRC_DESC="${TS}_src_baseline"
SRC_DIR=$(run_exp "src_baseline" "${SRC_DESC}" \
  "pretrained_weights=${PRETRAINED//=/\\=}" \
  datamodule="${SRC_DATAMODULE}" \
  model="${MODEL}" \
  ttt_frequency="${TTT_FREQ_OFF}" \
  save_adapted=false \
  lwf_weight=0.0 lwf_pi_weight=0.0 \
  long_horizon_gamma=0.0 long_horizon_floor=1.0)
index_row "src_baseline" "src" "—" "0" "0" "0" "${SRC_DIR}"

TGT_DESC="${TS}_tgt_noTTT"
TGT_DIR=$(run_exp "tgt_noTTT" "${TGT_DESC}" \
  "pretrained_weights=${PRETRAINED//=/\\=}" \
  datamodule="${TGT_DATAMODULE}" \
  model="${MODEL}" \
  ttt_frequency="${TTT_FREQ_OFF}" \
  save_adapted=false \
  lwf_weight=0.0 lwf_pi_weight=0.0 \
  long_horizon_gamma=0.0 long_horizon_floor=1.0)
index_row "tgt_noTTT" "tgt" "—" "0" "0" "0" "${TGT_DIR}"

# =============================================================================
# (1..5) Methods: target-adapt → save adapted ckpt → forgetting eval
# =============================================================================
# method   lwf     pi      gamma
METHODS=(
  "baseline      0.0  0.0   0.0"
  "horizon_only  0.0  0.0   2.0"
  "lwf_only      0.3  0.05  0.0"
  "lwf_horizon   0.3  0.05  2.0"
  "lwf_strong    1.0  0.10  2.0"
)

for row in "${METHODS[@]}"; do
  # parse row
  read -r METHOD LWF LWF_PI GAMMA <<< "${row}"

  # ---------- phase 1: target adaptation ----------
  PH1_DESC="${TS}_${METHOD}_target"
  PH1_DIR=$(run_exp "${METHOD}_target" "${PH1_DESC}" \
    "pretrained_weights=${PRETRAINED//=/\\=}" \
    datamodule="${TGT_DATAMODULE}" \
    model="${MODEL}" \
    ttt_frequency="${TTT_FREQ_ON}" \
    save_adapted=true \
    lwf_weight="${LWF}" lwf_pi_weight="${LWF_PI}" \
    long_horizon_gamma="${GAMMA}" long_horizon_floor=1.0)
  index_row "${METHOD}_target" "tgt" "${METHOD}" "${LWF}" "${LWF_PI}" "${GAMMA}" "${PH1_DIR}"

  ADAPTED="${PH1_DIR}/adapted_model.ckpt"
  if [[ ! -f "${ADAPTED}" ]]; then
    echo "[sweep] WARN: adapted ckpt not found for ${METHOD}: ${ADAPTED}" >&2
    echo "[sweep] skipping forgetting eval"
    continue
  fi

  # ---------- phase 2: forgetting eval on source ----------
  PH2_DESC="${TS}_${METHOD}_forgetting"
  PH2_DIR=$(run_exp "${METHOD}_forgetting" "${PH2_DESC}" \
    "pretrained_weights=${ADAPTED//=/\\=}" \
    datamodule="${SRC_DATAMODULE}" \
    model="${MODEL}" \
    ttt_frequency="${TTT_FREQ_OFF}" \
    save_adapted=false \
    lwf_weight=0.0 lwf_pi_weight=0.0 \
    long_horizon_gamma=0.0 long_horizon_floor=1.0)
  index_row "${METHOD}_forgetting" "src" "${METHOD}" "${LWF}" "${LWF_PI}" "${GAMMA}" "${PH2_DIR}"
done

# =============================================================================
# Aggregate everything into a CSV
# =============================================================================
echo "============================================================"
echo "[sweep] all runs done. Parsing results..."
python scripts/parse_sweep.py --index "${INDEX}" --out "${SWEEP_DIR}/results.csv"
echo "[sweep] done.  See:"
echo "  ${INDEX}"
echo "  ${SWEEP_DIR}/results.csv"
