#!/bin/bash
# Source forgetting check for 5 selected methods.
# For methods that lack an adapted checkpoint (save_adapted=false),
# re-run TTT with save_adapted=true first, then evaluate on INTER.
set -e
cd /home/ustb/T4P
source /home/ustb/miniconda/etc/profile.d/conda.sh
conda activate forecast_mae

LOG=outputs/forget_queue.log

forget_check_only() {
  local ckpt=$1
  local desc=$2
  echo "$(date) === FORGET CHECK: $desc ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=0 python test.py \
    --config-name=config_test_inter13 \
    datamodule=inter_13 \
    pretrained_weights="$ckpt" \
    ttt_frequency=999999 \
    save_adapted=false \
    desc="forget_${desc}" \
    2>&1 | tee outputs/forget_${desc}.log
  echo "$(date) === DONE FORGET: $desc ===" | tee -a $LOG
}

retrain_and_forget() {
  local desc=$1
  shift
  echo "$(date) === RE-TRAIN (save): $desc ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=0 python test.py \
    --config-name=config_test_inter13 \
    datamodule=inter_nus_13 \
    ttt_frequency=12 "$@" \
    save_adapted=true \
    desc="${desc}" \
    2>&1 | tee outputs/${desc}_resave.log
  echo "$(date) === DONE RE-TRAIN: $desc ===" | tee -a $LOG

  # Hydra writes output under outputs/forecast-mae-ttt-test_True/<date>/<time>_<desc>/
  ckpt=$(find outputs/forecast-mae-ttt-test_True -path "*${desc}*" \
    -name "adapted_model.ckpt" | sort | tail -1)
  if [ -z "$ckpt" ]; then
    echo "$(date) ERROR: checkpoint not found for $desc" | tee -a $LOG
    return 1
  fi
  echo "$(date) Found ckpt: $ckpt" | tee -a $LOG
  forget_check_only "$ckpt" "$desc"
}

# -----------------------------------------------------------------------
# Task 0: uniform feature LwF w=0.3  (checkpoint already saved)
# -----------------------------------------------------------------------
T0_CKPT=$(find outputs/forecast-mae-ttt-test_True \
  -path "*uniform_lwf_feat03_inter2nus/adapted_model.ckpt" | sort | tail -1)
echo "Task0 ckpt: $T0_CKPT"
if [ -n "$T0_CKPT" ]; then
  forget_check_only "$T0_CKPT" "uniform_lwf_feat03"
else
  echo "WARN: Task0 checkpoint not found, re-running" | tee -a $LOG
  retrain_and_forget "uniform_lwf_feat03_inter2nus" \
    lwf_feature_agent_weight=0.3
fi

# -----------------------------------------------------------------------
# Task 1-B: encoder LR x0.01
# -----------------------------------------------------------------------
retrain_and_forget "encLR001_inter2nus" encoder_lr_scale=0.01

# -----------------------------------------------------------------------
# Task 2-C: L2-SP 1e-3
# -----------------------------------------------------------------------
retrain_and_forget "l2sp_1e3_inter2nus" l2sp_weight=1e-3

# -----------------------------------------------------------------------
# Task 3-C2: lane LwF only
# -----------------------------------------------------------------------
retrain_and_forget "mlwf_C2_lane_inter2nus" lwf_feature_lane_weight=0.3

# -----------------------------------------------------------------------
# Task 3-C5: full_enc LwF only
# -----------------------------------------------------------------------
retrain_and_forget "mlwf_C5_full_enc_inter2nus" lwf_feature_full_encoder_weight=0.3

echo "$(date) ALL FORGET CHECKS DONE" | tee -a $LOG
