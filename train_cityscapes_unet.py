from __future__ import annotations

import argparse
import functools
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets

from etbq import utils


TARGET_WEIGHT_LAYERS = (nn.Conv2d, nn.Linear)
TARGET_ACTIVATION_LAYERS = (nn.ReLU, nn.ReLU6)
CITYSCAPES_MEAN = [0.485, 0.456, 0.406]
CITYSCAPES_STD = [0.229, 0.224, 0.225]


class CityscapesWrapper(datasets.Cityscapes):
    def __init__(self, root, split="train", target_type="semantic", crop_size=(512, 512)):
        super().__init__(root, split=split, mode="fine", target_type=target_type)
        self.crop_size = crop_size
        self.ignore_index = 255
        self.id_to_train_id = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
            26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
        }

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        target = TF.crop(target, i, j, h, w)
        if random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)
        image = TF.normalize(TF.to_tensor(image), mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD)
        target = torch.from_numpy(np.array(target)).long()
        mapped = torch.ones_like(target) * self.ignore_index
        for raw_id, train_id in self.id_to_train_id.items():
            mapped[target == raw_id] = train_id
        return image, mapped


class ETBQHooks:
    def __init__(self, args):
        self.args = args
        self.weight_mean: Dict[str, torch.Tensor] = {}
        self.weight_var: Dict[str, torch.Tensor] = {}
        self.ema_weight_mean: Dict[str, torch.Tensor] = {}
        self.ema_weight_var: Dict[str, torch.Tensor] = {}
        self.prev_weight_error: Dict[str, torch.Tensor] = {}
        self.act_mean: Dict[str, torch.Tensor] = {}
        self.act_var: Dict[str, torch.Tensor] = {}
        self.ema_act_mean: Dict[str, torch.Tensor] = {}
        self.ema_act_var: Dict[str, torch.Tensor] = {}
        self.handles = []
        self.error_intensity = 0.0

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def remove_residual_weight_error(self, model):
        if not self.prev_weight_error:
            return
        for name, module in model.named_modules():
            if name in self.prev_weight_error and hasattr(module, "weight"):
                module.weight.data.sub_(self.prev_weight_error[name].to(module.weight.device))
        self.prev_weight_error = {}

    def _weight_pre_hook(self, module, inputs, layer_name):
        if not module.training or self.error_intensity == 0.0 or layer_name not in self.weight_mean:
            return
        mean = self.weight_mean[layer_name].to(module.weight.device)
        var = self.weight_var[layer_name].to(module.weight.device)
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        current_error = torch.randn_like(module.weight) * std * self.error_intensity + mean * self.error_intensity
        previous_error = self.prev_weight_error.get(layer_name, torch.zeros_like(module.weight))
        previous_error = previous_error.to(module.weight.device)
        module.weight.data.add_(current_error - previous_error)
        self.prev_weight_error[layer_name] = current_error.detach().clone()

    def _activation_hook(self, module, inputs, output, layer_name):
        if not module.training or self.error_intensity == 0.0:
            return output
        if layer_name not in self.act_mean or not torch.is_tensor(output):
            return output
        mean = self.act_mean[layer_name].to(output.device)
        var = self.act_var[layer_name].to(output.device)
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        error = torch.randn_like(output) * std * self.error_intensity + mean * self.error_intensity
        rho = self.args.activation_error_density
        if rho >= 1.0:
            return output + error
        if self.args.activation_mask == "uniform" or output.dim() != 4:
            mask = (torch.rand_like(output) < rho).to(output.dtype)
            return output + mask * error
        importance = output.detach().abs().mean(dim=(2, 3), keepdim=True)
        score = torch.softmax(importance / max(self.args.activation_temperature, 1e-6), dim=1)
        prob = torch.clamp(score * output.size(1) * rho, 0.0, 1.0)
        mask = torch.bernoulli(prob).to(output.dtype)
        return output + mask * error

    def update_error_statistics(self, model, calib_loader, device, epoch, noise_start_epoch):
        if self.args.wqn:
            self.remove_residual_weight_error(model)
            q_weights = utils.simulate_quantized_weights(model, self.args.w_bits, TARGET_WEIGHT_LAYERS)
            mean, var = utils.calculate_weight_quantization_error(model, q_weights, TARGET_WEIGHT_LAYERS)
            for name in mean:
                self.ema_weight_mean[name] = (
                    self.args.weight_ema * self.ema_weight_mean.get(name, mean[name])
                    + (1.0 - self.args.weight_ema) * mean[name]
                )
                self.ema_weight_var[name] = (
                    self.args.weight_ema * self.ema_weight_var.get(name, var[name])
                    + (1.0 - self.args.weight_ema) * var[name]
                )
            self.weight_mean = self.ema_weight_mean
            self.weight_var = self.ema_weight_var
            if epoch == noise_start_epoch:
                self.prev_weight_error = {}

        if self.args.aqn:
            act_names = [name for name, module in model.named_modules() if isinstance(module, TARGET_ACTIVATION_LAYERS)]
            stats = utils.calculate_activation_quantization_error_stats(
                model,
                calib_loader,
                self.args.a_bits,
                act_names,
                device,
                self.args.num_calib_batches,
                task_type="segmentation",
            )
            for name in stats["mean"]:
                self.ema_act_mean[name] = (
                    self.args.act_ema * self.ema_act_mean.get(name, stats["mean"][name])
                    + (1.0 - self.args.act_ema) * stats["mean"][name]
                )
                self.ema_act_var[name] = (
                    self.args.act_ema * self.ema_act_var.get(name, stats["variance"][name])
                    + (1.0 - self.args.act_ema) * stats["variance"][name]
                )
            self.act_mean = self.ema_act_mean
            self.act_var = self.ema_act_var

    def register(self, model):
        self.remove_hooks()
        if self.error_intensity <= 0.0:
            return
        if self.args.wqn:
            for name, module in model.named_modules():
                if isinstance(module, TARGET_WEIGHT_LAYERS) and name in self.weight_mean:
                    self.handles.append(
                        module.register_forward_pre_hook(
                            functools.partial(self._weight_pre_hook, layer_name=name)
                        )
                    )
        if self.args.aqn:
            for name, module in model.named_modules():
                if isinstance(module, TARGET_ACTIVATION_LAYERS) and name in self.act_mean:
                    self.handles.append(
                        module.register_forward_hook(
                            functools.partial(self._activation_hook, layer_name=name)
                        )
                    )


def fast_hist(label, pred, n):
    mask = (label >= 0) & (label < n)
    return np.bincount(n * label[mask].astype(int) + pred[mask], minlength=n ** 2).reshape(n, n)


@torch.no_grad()
def evaluate_miou(model, loader, device, num_classes):
    was_training = model.training
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    for image, mask_true in tqdm(loader, desc="mIoU", leave=False):
        image = image.to(device, dtype=torch.float32, non_blocking=True)
        pred = model(image).argmax(dim=1).cpu().numpy().flatten()
        label = mask_true.numpy().flatten()
        hist += fast_hist(label, pred, num_classes)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
    model.train(was_training)
    return float(np.nanmean(iu))


def parse_args():
    parser = argparse.ArgumentParser(description="ETBQ one-click Cityscapes U-Net pre-conditioning")
    parser.add_argument("--data_dir", default="./data/cityscapes")
    parser.add_argument("--save_dir", default="./outputs/cityscapes_unet")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder_weights", default="imagenet")
    parser.add_argument("--num_classes", default=19, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--crop_h", default=512, type=int)
    parser.add_argument("--crop_w", default=512, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--resume", default="")

    parser.add_argument("--epochs_clean", default=300, type=int)
    parser.add_argument("--epochs_noise", default=100, type=int)
    parser.add_argument("--epochs_swa", default=100, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--swa_lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)

    parser.add_argument("--wqn", action="store_true")
    parser.add_argument("--aqn", action="store_true")
    parser.add_argument("--w_bits", default=4, type=int)
    parser.add_argument("--a_bits", default=4, type=int)
    parser.add_argument("--weight_ema", default=0.9, type=float)
    parser.add_argument("--act_ema", default=0.9, type=float)
    parser.add_argument("--num_calib_batches", default=4, type=int)
    parser.add_argument("--noise_ramp_up_epochs", default=10, type=int)
    parser.add_argument("--error_max_intensity", default=0.8, type=float)
    parser.add_argument("--activation_error_density", default=0.5, type=float)
    parser.add_argument("--activation_mask", default="salience", choices=["salience", "uniform"])
    parser.add_argument("--activation_temperature", default=1.0, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Cityscapes U-Net training requires segmentation_models_pytorch. "
            "Install it with: pip install segmentation-models-pytorch"
        ) from exc

    seed = utils.seed_everything(args.seed)
    print(f"Seed: {seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    utils.mkdir(args.save_dir)

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=args.num_classes,
        activation=None,
    ).to(device)
    if args.resume:
        state = utils.load_state_dict_file(args.resume, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.resume}")

    train_ds = CityscapesWrapper(args.data_dir, split="train", crop_size=(args.crop_h, args.crop_w))
    val_ds = CityscapesWrapper(args.data_dir, split="val", crop_size=(args.crop_h, args.crop_w))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    calib_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=8, factor=0.5)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    hooks = ETBQHooks(args)

    clean_end = args.epochs_clean
    noise_end = args.epochs_clean + args.epochs_noise
    total_epochs = args.epochs_clean + args.epochs_noise + args.epochs_swa
    best_score = 0.0

    for epoch in range(total_epochs):
        is_noise = clean_end <= epoch < noise_end
        is_swa = epoch >= noise_end
        phase = "clean" if epoch < clean_end else ("noise" if is_noise else "swa")
        print(f"\nEpoch {epoch + 1}/{total_epochs} | phase={phase}")

        hooks.error_intensity = 0.0
        if is_noise:
            if args.noise_ramp_up_epochs <= 1:
                ramp = 1.0
            else:
                ramp = min(1.0, (epoch - clean_end + 1) / args.noise_ramp_up_epochs)
            hooks.error_intensity = ramp * args.error_max_intensity
            hooks.update_error_statistics(model, calib_loader, device, epoch, clean_end)
        hooks.register(model)

        model.train()
        epoch_loss = 0.0
        iterator = tqdm(train_loader, desc="Train", leave=False)
        for images, masks in iterator:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            masks = masks.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss += loss.item()
            iterator.set_postfix(loss=f"{epoch_loss / max(1, iterator.n + 1):.4f}")

        hooks.remove_hooks()
        hooks.remove_residual_weight_error(model)

        if is_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        score = evaluate_miou(model, val_loader, device, args.num_classes)
        if not is_swa:
            scheduler.step(score)
        print(f"Validation mIoU: {score:.4f}")

        if score > best_score:
            best_score = score
            path = os.path.join(args.save_dir, f"best_{phase}_w{args.w_bits}a{args.a_bits}.pth")
            torch.save(model.state_dict(), path)
            print(f"Best updated -> {path}")

    if args.epochs_swa > 0:
        print("Updating BN for final SWA model...")
        update_bn(train_loader, swa_model, device=device)
        swa_score = evaluate_miou(swa_model, val_loader, device, args.num_classes)
        path = os.path.join(args.save_dir, f"swa_final_w{args.w_bits}a{args.a_bits}.pth")
        torch.save(swa_model.module.state_dict(), path)
        print(f"Final SWA mIoU: {swa_score:.4f}; saved -> {path}")


if __name__ == "__main__":
    main()
