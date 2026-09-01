#!/bin/bash
# Task C: post-hoc encoder interpolation between the source-pretrained checkpoint
# and Group 3's adapted checkpoint (full_enc LwF + adapter, encoder not frozen --
# the only group with real, meaningful encoder drift to interpolate away).
# Pure eval (ttt_frequency=999999), no re-training. Both target and source evals
# explicitly pass Group 3's adapter config so the merged checkpoint's adapter
# weights actually participate in the forward pass.
set -e
cd /home/ustb/T4P
source /home/ustb/miniconda/etc/profile.d/conda.sh
conda activate forecast_mae

LOG=outputs/encoder_interp_queue.log

SOURCE_CKPT="outputs/forecast-mae-ttt-test_False/2025-12-11/09-58-41_NLL_13_tr_inter_vl_lyftsample/checkpoints/epoch=1185.ckpt"
ADAPTED_CKPT="outputs/forecast-mae-ttt-test_True/2026-08-29/18-49-17_adapter_fullenc_lr10_inter2nus/adapted_model.ckpt"

eval_both() {
  local desc=$1
  local ckpt=$2

  echo "$(date) === TARGET EVAL: $desc ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=0 python test.py --config-name=config_test_inter13 \
    datamodule=inter_nus_13 pretrained_weights="$ckpt" ttt_frequency=999999 save_adapted=false \
    desc="target_${desc}" \
    use_deep_adapter=true deep_adapter_alpha=1.0 deep_adapter_detach_base=false \
    2>&1 | tee outputs/target_${desc}.log

  echo "$(date) === SOURCE EVAL: $desc ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=0 python test.py --config-name=config_test_inter13 \
    datamodule=inter_13 pretrained_weights="$ckpt" ttt_frequency=999999 save_adapted=false \
    desc="source_${desc}" \
    use_deep_adapter=true deep_adapter_alpha=1.0 deep_adapter_detach_base=false \
    2>&1 | tee outputs/source_${desc}.log
  echo "$(date) === DONE: $desc ===" | tee -a $LOG
}

for rho in 0.25 0.5 0.75; do
  desc="interp_g3_rho${rho}"
  merged="outputs/interp/adapted_model_rho${rho}.ckpt"
  mkdir -p outputs/interp
  echo "$(date) === MERGE: rho=$rho ===" | tee -a $LOG
  python interpolate_encoder.py --source "$SOURCE_CKPT" --adapted "$ADAPTED_CKPT" --rho "$rho" --out "$merged" \
    2>&1 | tee outputs/interp_merge_rho${rho}.log
  eval_both "$desc" "$merged"
done

echo "$(date) ALL ENCODER INTERP EVALS DONE" | tee -a $LOG
