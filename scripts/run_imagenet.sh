#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-./data/imagenet}
SAVE_DIR=${SAVE_DIR:-./outputs/imagenet_resnet18_w2a4}

python train_classification.py \
  --dataset imagenet \
  --data_dir "$DATA_DIR" \
  --model_arch resnet18 \
  --pretrained \
  --save_dir "$SAVE_DIR" \
  --epochs 80 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.0 \
  --noise_start_epoch_frac 0.0 \
  --noise_ramp_up_epochs 3 \
  --error_max_intensity 0.05 \
  --swa_rate 0.1 \
  --freeze_bn_in_noise_phase \
  --wqn --aqn \
  --w_bits 2 \
  --a_bits 4 \
  --num_calib_samples 1024 \
  --num_calib_batches 8 \
  --activation_mask salience \
  --activation_error_density 0.05

