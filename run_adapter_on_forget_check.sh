#!/bin/bash
# Adapter-ON source (INTER) forgetting re-check for groups 2-5.
# The original run_adapter_ablation.sh forget-check step never passed
# use_deep_adapter=true, so those source MR numbers reflect the adapter-off
# base path, not the true adapted-model source performance. This script
# re-evaluates the SAME already-saved adapted_model.ckpt files with the
# adapter enabled (matching each group's training-time adapter config).
set -e
cd /home/ustb/T4P
source /home/ustb/miniconda/etc/profile.d/conda.sh
conda activate forecast_mae

LOG=outputs/adapter_on_forget_check_queue.log

forget_check_adapter_on() {
  local desc=$1
  local ckpt=$2
  shift 2
  echo "$(date) === FORGET CHECK (adapter-on): $desc ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=0 python test.py \
    --config-name=config_test_inter13 \
    datamodule=inter_13 \
    pretrained_weights="$ckpt" \
    ttt_frequency=999999 \
    save_adapted=false \
    desc="forget_adapteron_${desc}" \
    use_deep_adapter=true \
    "$@" \
    2>&1 | tee outputs/forget_adapteron_${desc}.log
  echo "$(date) === DONE FORGET (adapter-on): $desc ===" | tee -a $LOG
}

forget_check_adapter_on \
  "adapter_only_lr10_inter2nus" \
  "outputs/forecast-mae-ttt-test_True/2026-08-29/18-32-05_adapter_only_lr10_inter2nus/adapted_model.ckpt" \
  deep_adapter_alpha=1.0 deep_adapter_detach_base=false

forget_check_adapter_on \
  "adapter_fullenc_lr10_inter2nus" \
  "outputs/forecast-mae-ttt-test_True/2026-08-29/18-49-17_adapter_fullenc_lr10_inter2nus/adapted_model.ckpt" \
  deep_adapter_alpha=1.0 deep_adapter_detach_base=false

forget_check_adapter_on \
  "adapter_fullenc_enclr01_lr10_inter2nus" \
  "outputs/forecast-mae-ttt-test_True/2026-08-29/19-06-22_adapter_fullenc_enclr01_lr10_inter2nus/adapted_model.ckpt" \
  deep_adapter_alpha=1.0 deep_adapter_detach_base=false

forget_check_adapter_on \
  "adapter_fullenc_freeze_detach_lr10_inter2nus" \
  "outputs/forecast-mae-ttt-test_True/2026-08-29/19-23-28_adapter_fullenc_freeze_detach_lr10_inter2nus/adapted_model.ckpt" \
  deep_adapter_alpha=1.0 deep_adapter_detach_base=true

echo "$(date) ALL ADAPTER-ON FORGET CHECKS DONE" | tee -a $LOG
