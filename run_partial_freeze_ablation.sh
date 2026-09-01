#!/bin/bash
# Task A (partial encoder block freezing) + Task B (larger adapter dim),
# 4 priority groups, target (nuScenes) TTT+save then source (INTER) forget-check.
# The forget-check step passes "$@" (each group's adapter/freeze config) so that
# use_deep_adapter/deep_adapter_alpha/deep_adapter_detach_base/deep_adapter_dim
# stay consistent with training -- the bug fixed in run_adapter_on_forget_check.sh
# is baked in here from the start.
set -e
cd /home/ustb/T4P
source /home/ustb/miniconda/etc/profile.d/conda.sh
conda activate forecast_mae

LOG=outputs/partial_freeze_ablation_queue.log

retrain_and_forget() {
  local desc=$1; shift
  echo "$(date) === RE-TRAIN (save): $desc ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=0 python test.py --config-name=config_test_inter13 \
    datamodule=inter_nus_13 ttt_frequency=12 "$@" save_adapted=true desc="${desc}" \
    2>&1 | tee outputs/${desc}_resave.log
  echo "$(date) === DONE RE-TRAIN: $desc ===" | tee -a $LOG

  ckpt=$(find outputs/forecast-mae-ttt-test_True -path "*${desc}*" -name "adapted_model.ckpt" | sort | tail -1)
  if [ -z "$ckpt" ]; then
    echo "$(date) ERROR: ckpt not found for $desc" | tee -a $LOG
    return 1
  fi
  echo "$(date) Found ckpt: $ckpt" | tee -a $LOG

  echo "$(date) === FORGET CHECK: $desc ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=0 python test.py --config-name=config_test_inter13 \
    datamodule=inter_13 pretrained_weights="$ckpt" ttt_frequency=999999 save_adapted=false \
    desc="forget_${desc}" "$@" \
    2>&1 | tee outputs/forget_${desc}.log
  echo "$(date) === DONE FORGET: $desc ===" | tee -a $LOG
}

# -----------------------------------------------------------------------
# Task A: freeze first 50% / 75% of encoder blocks, full_enc LwF + adapter
# -----------------------------------------------------------------------
retrain_and_forget "adapter_fullenc_freeze50_lr10_inter2nus" \
  lwf_feature_full_encoder_weight=0.3 use_deep_adapter=true adapter_lr_scale=10.0 \
  freeze_encoder_blocks=first_50%

retrain_and_forget "adapter_fullenc_freeze75_lr10_inter2nus" \
  lwf_feature_full_encoder_weight=0.3 use_deep_adapter=true adapter_lr_scale=10.0 \
  freeze_encoder_blocks=first_75%

# -----------------------------------------------------------------------
# Task B: larger adapter (dim=64), one fully-frozen config + one partial-freeze
# -----------------------------------------------------------------------
retrain_and_forget "adapter_fullenc_freezeall_detach_adim64_inter2nus" \
  lwf_feature_full_encoder_weight=0.3 use_deep_adapter=true adapter_lr_scale=10.0 \
  freeze_encoder_backbone=true deep_adapter_detach_base=true \
  model.target.deep_adapter_dim=64

retrain_and_forget "adapter_fullenc_freeze50_adim64_inter2nus" \
  lwf_feature_full_encoder_weight=0.3 use_deep_adapter=true adapter_lr_scale=10.0 \
  freeze_encoder_blocks=first_50% \
  model.target.deep_adapter_dim=64

echo "$(date) ALL PARTIAL-FREEZE ABLATION GROUPS DONE" | tee -a $LOG
