#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-./data/tiny-imagenet-200}
SAVE_DIR=${SAVE_DIR:-./outputs/tiny_resnet18_w2a4}

python train_classification.py \
  --dataset tiny-imagenet \
  --data_dir "$DATA_DIR" \
  --model_arch resnet18 \
  --pretrained \
  --save_dir "$SAVE_DIR" \
  --epochs 120 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --weight_decay 0.001 \
  --label_smoothing 0.1 \
  --noise_start_epoch_frac 0.0 \
  --noise_ramp_up_epochs 20 \
  --error_max_intensity 0.8 \
  --swa_rate 0.5 \
  --wqn --aqn \
  --w_bits 2 \
  --a_bits 4 \
  --num_calib_samples 200 \
  --num_calib_batches 2 \
  --activation_mask salience \
  --activation_error_density 0.25

