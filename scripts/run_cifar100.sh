#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-./data}
SAVE_DIR=${SAVE_DIR:-./outputs/cifar100_resnet18_w2a4}

python train_classification.py \
  --dataset cifar100 \
  --data_dir "$DATA_DIR" \
  --model_arch resnet18 \
  --save_dir "$SAVE_DIR" \
  --epochs 120 \
  --batch_size 256 \
  --learning_rate 0.015 \
  --weight_decay 0.0005 \
  --label_smoothing 0.1 \
  --noise_start_epoch_frac 0.0 \
  --noise_ramp_up_epochs 20 \
  --error_max_intensity 1.0 \
  --swa_rate 0.5 \
  --wqn --aqn \
  --w_bits 2 \
  --a_bits 4 \
  --num_calib_samples 100 \
  --num_calib_batches 1 \
  --activation_mask salience \
  --activation_error_density 0.5

