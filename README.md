# ETBQ: Efficient Tuning Before Low-Bit PTQ

This folder contains one-click training scripts for ETBQ-style full-precision
pre-conditioning before post-training quantization.

ETBQ updates the full-precision model with target quantization-error statistics:

- **WQN**: channel-wise weight quantization error conditioning.
- **AQN**: activation quantization error conditioning with salience-aware masking.
- **SWA**: late-stage stochastic weight averaging.

The saved checkpoint is still a full-precision model and can be sent to a
downstream PTQ backend such as QDrop or QEP.

## Installation

```bash
cd ETBQ_Code
pip install -r requirements.txt
```

## One-click Runs

Set `DATA_DIR` when your dataset is not under the default `./data`.

```bash
# CIFAR-100 / ResNet-18 / W2A4 pre-conditioning
DATA_DIR=/path/to/data bash scripts/run_cifar100.sh

# Tiny-ImageNet / ResNet-18 / W2A4 pre-conditioning
DATA_DIR=/path/to/tiny-imagenet-200 bash scripts/run_tiny_imagenet.sh

# ImageNet / ResNet-18 / W2A4 pre-conditioning
DATA_DIR=/path/to/imagenet bash scripts/run_imagenet.sh

# Cityscapes / U-Net-ResNet34 / W4A4 pre-conditioning
DATA_DIR=/path/to/cityscapes bash scripts/run_cityscapes_unet.sh
```

You can override the output folder:

```bash
SAVE_DIR=./outputs/debug DATA_DIR=/path/to/data bash scripts/run_cifar100.sh
```

## Dataset Layout

### CIFAR-100

The script uses `torchvision.datasets.CIFAR100` and can download the dataset
automatically into `DATA_DIR`.

### Tiny-ImageNet

Expected layout:

```text
tiny-imagenet-200/
  train/
  val/
    images/
    val_annotations.txt
  wnids.txt
```

### ImageNet

Either ImageFolder-style:

```text
imagenet/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

or the standard `torchvision.datasets.ImageNet` layout.

### Cityscapes

Expected torchvision Cityscapes layout under `DATA_DIR`, with `leftImg8bit`
and `gtFine` folders.

## Main Arguments

Common classification entry:

```bash
python train_classification.py --help
```

Important flags:

- `--dataset`: `cifar100`, `tiny-imagenet`, or `imagenet`.
- `--model_arch`: `resnet18`, `resnet50`, `mobilenet_v1`, `mobilenet_v2`,
  `ghostnet_100`, `ghostnet_130`, or `ghostnetv2_100`.
- `--wqn`: enable weight quantization error conditioning.
- `--aqn`: enable activation quantization error conditioning.
- `--w_bits`, `--a_bits`: target bit-widths used to estimate ETBQ errors.
- `--num_calib_samples`: calibration subset size for AQN statistics.
- `--error_max_intensity`: maximum ETBQ error intensity.

Cityscapes entry:

```bash
python train_cityscapes_unet.py --help
```

## Notes

The code follows the paper convention:

```text
weight error:     E_w = W_q - W
activation error: E_a = A_q - A
```

During WQN, the injected weight-side error is temporally differenced to control
the drift caused by non-zero-mean quantization errors. Before evaluation and
saving, the residual sampled weight error is removed so that checkpoints contain
clean full-precision weights.

