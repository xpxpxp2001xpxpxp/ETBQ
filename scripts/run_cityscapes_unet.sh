#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-./data/cityscapes}
SAVE_DIR=${SAVE_DIR:-./outputs/cityscapes_unet_w4a4}

python train_cityscapes_unet.py \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --encoder resnet34 \
  --epochs_clean 300 \
  --epochs_noise 100 \
  --epochs_swa 100 \
  --batch_size 4 \
  --learning_rate 0.01 \
  --swa_lr 0.001 \
  --noise_ramp_up_epochs 10 \
  --error_max_intensity 0.8 \
  --wqn --aqn \
  --w_bits 4 \
  --a_bits 4 \
  --num_calib_batches 4 \
  --activation_mask salience \
  --activation_error_density 0.5

