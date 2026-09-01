#!/bin/bash
# Deep residual adapter ablation (5 groups), target (nuScenes) + source (INTER) forgetting.
# Group 1 (full_enc LwF baseline) reuses existing results:
#   target: mlwf_C5_full_enc_inter2nus       -> MR=0.214 ADE6=0.685 FDE6=1.470
#   source: forget_mlwf_C5_full_enc_inter2nus -> MR=0.344 ADE6=0.711 FDE6=1.867
# Groups 2-5 all use adapter_lr_scale=10 (zero-init adapter + few TTT steps
# needs a boosted LR to move at all; see 2026-08-28 diagnostic confirming
# adapter_delta_norm/grad_norm are genuinely nonzero and growing under this
# setting, not just noise).
set -e
cd /home/ustb/T4P
source /home/ustb/miniconda/etc/profile.d/conda.sh
conda activate forecast_mae

LOG=outputs/adapter_ablation_queue.log

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

  ckpt=$(find outputs/forecast-mae-ttt-test_True -path "*${desc}*" \
    -name "adapted_model.ckpt" | sort | tail -1)
  if [ -z "$ckpt" ]; then
    echo "$(date) ERROR: checkpoint not found for $desc" | tee -a $LOG
    return 1
  fi
  echo "$(date) Found ckpt: $ckpt" | tee -a $LOG

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

# -----------------------------------------------------------------------
# Group 1: full_enc LwF baseline -- REUSED, not re-run.
# target MR=0.214 ADE6=0.685 FDE6=1.470 / source MR=0.344 ADE6=0.711 FDE6=1.867
# -----------------------------------------------------------------------
echo "$(date) Group 1 (full_enc LwF baseline) reused from mlwf_C5_full_enc_inter2nus, not re-run" | tee -a $LOG

# -----------------------------------------------------------------------
# Group 2: adapter only, adapter_lr_scale=10 (no LwF)
# -----------------------------------------------------------------------
retrain_and_forget "adapter_only_lr10_inter2nus" \
  use_deep_adapter=true adapter_lr_scale=10.0

# -----------------------------------------------------------------------
# Group 3: full_enc LwF + adapter, adapter_lr_scale=10
# -----------------------------------------------------------------------
retrain_and_forget "adapter_fullenc_lr10_inter2nus" \
  lwf_feature_full_encoder_weight=0.3 use_deep_adapter=true adapter_lr_scale=10.0

# -----------------------------------------------------------------------
# Group 4: full_enc LwF + adapter + encoder_lr_scale=0.1, adapter_lr_scale=10
# -----------------------------------------------------------------------
retrain_and_forget "adapter_fullenc_enclr01_lr10_inter2nus" \
  lwf_feature_full_encoder_weight=0.3 use_deep_adapter=true adapter_lr_scale=10.0 \
  encoder_lr_scale=0.1

# -----------------------------------------------------------------------
# Group 5: full_enc LwF + adapter + freeze_encoder_backbone + detach_base,
#          adapter_lr_scale=10 (clean isolation: task loss only trains adapter)
# -----------------------------------------------------------------------
retrain_and_forget "adapter_fullenc_freeze_detach_lr10_inter2nus" \
  lwf_feature_full_encoder_weight=0.3 use_deep_adapter=true adapter_lr_scale=10.0 \
  freeze_encoder_backbone=true deep_adapter_detach_base=true

echo "$(date) ALL ADAPTER ABLATION GROUPS DONE" | tee -a $LOG
