from __future__ import annotations

import argparse
import functools
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

from etbq.data import build_calib_loader, build_classification_loaders
from etbq.models import build_classification_model
from etbq import utils


TARGET_WEIGHT_LAYERS = (nn.Conv2d, nn.Linear)
TARGET_ACTIVATION_LAYERS = (nn.ReLU, nn.ReLU6)


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
        if not module.training or self.error_intensity == 0.0:
            return
        if layer_name not in self.weight_mean:
            return

        mean = self.weight_mean[layer_name].to(module.weight.device)
        var = self.weight_var[layer_name].to(module.weight.device)
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        current_error = torch.randn_like(module.weight) * std * self.error_intensity + mean * self.error_intensity
        previous_error = self.prev_weight_error.get(
            layer_name,
            torch.zeros_like(module.weight, device=module.weight.device),
        )
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

        if self.args.aqn and calib_loader is not None:
            act_names = [
                name for name, module in model.named_modules()
                if isinstance(module, TARGET_ACTIVATION_LAYERS)
            ]
            stats = utils.calculate_activation_quantization_error_stats(
                model,
                calib_loader,
                self.args.a_bits,
                act_names,
                device,
                self.args.num_calib_batches,
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


def freeze_batchnorm_only(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def parse_args():
    parser = argparse.ArgumentParser(description="ETBQ one-click classification pre-conditioning")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "tiny-imagenet", "imagenet"])
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--model_arch", default="resnet18",
                        choices=["resnet18", "resnet50", "mobilenet_v1", "mobilenet_v2",
                                 "ghostnet_100", "ghostnet_130", "ghostnetv2_100"])
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights when available.")
    parser.add_argument("--load_checkpoint", default="", help="Optional FP checkpoint to start from.")
    parser.add_argument("--save_dir", default="./outputs/classification")

    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=0.015, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--noise_start_epoch_frac", default=0.0, type=float)
    parser.add_argument("--noise_ramp_up_epochs", default=20, type=int)
    parser.add_argument("--error_max_intensity", default=1.0, type=float)
    parser.add_argument("--freeze_bn_in_noise_phase", action="store_true")

    parser.add_argument("--wqn", action="store_true", help="Enable weight quantization error conditioning.")
    parser.add_argument("--aqn", action="store_true", help="Enable activation quantization error conditioning.")
    parser.add_argument("--w_bits", default=2, type=int)
    parser.add_argument("--a_bits", default=4, type=int)
    parser.add_argument("--weight_ema", default=0.9, type=float)
    parser.add_argument("--act_ema", default=0.9, type=float)
    parser.add_argument("--num_calib_samples", default=1024, type=int)
    parser.add_argument("--num_calib_batches", default=16, type=int)
    parser.add_argument("--activation_error_density", default=0.25, type=float)
    parser.add_argument("--activation_mask", default="salience", choices=["salience", "uniform"])
    parser.add_argument("--activation_temperature", default=1.0, type=float)

    parser.add_argument("--swa_rate", default=0.5, type=float)
    parser.add_argument("--swa_bn_batches", default=20, type=int)
    parser.add_argument("--eval_initial", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.wqn and not args.aqn:
        raise ValueError("Enable at least one of --wqn or --aqn.")
    if not 0.0 <= args.error_max_intensity <= 1.0:
        raise ValueError("--error_max_intensity must be in [0, 1].")

    seed = utils.seed_everything(args.seed)
    print(f"Seed: {seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, num_classes = build_classification_loaders(args)
    calib_loader = None
    if args.aqn:
        calib_loader = build_calib_loader(
            train_loader.dataset,
            args.batch_size,
            args.num_calib_samples,
            args.num_workers,
            seed,
        )

    model = build_classification_model(
        args.model_arch,
        num_classes=num_classes,
        dataset=args.dataset,
        pretrained=args.pretrained,
    ).to(device)

    if args.load_checkpoint:
        state = utils.load_state_dict_file(args.load_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.load_checkpoint}")
        if missing:
            print(f"Missing keys (first 8): {missing[:8]}")
        if unexpected:
            print(f"Unexpected keys (first 8): {unexpected[:8]}")

    if args.eval_initial:
        acc = utils.classification_accuracy(model, val_loader, device)
        print(f"Initial FP32 accuracy: {acc:.2f}%")

    optimizer = SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    swa_start_epoch = int(args.epochs * args.swa_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(len(train_loader) * max(swa_start_epoch, 1), 1))
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.learning_rate * 0.01)
    hooks = ETBQHooks(args)

    noise_start_epoch = int(args.epochs * args.noise_start_epoch_frac)
    best_acc = 0.0
    utils.mkdir(args.save_dir)

    print(f"ETBQ: dataset={args.dataset}, model={args.model_arch}, W{args.w_bits}A{args.a_bits}")
    print(f"Noise starts at epoch {noise_start_epoch + 1}; SWA starts at epoch {swa_start_epoch + 1}.")

    for epoch in range(args.epochs):
        if epoch >= noise_start_epoch:
            if args.noise_ramp_up_epochs <= 1:
                ramp = 1.0
            else:
                ramp = min(1.0, (epoch - noise_start_epoch + 1) / args.noise_ramp_up_epochs)
        else:
            ramp = 0.0
        hooks.error_intensity = ramp * args.error_max_intensity

        print(f"\nEpoch {epoch + 1}/{args.epochs} | error_intensity={hooks.error_intensity:.4f}")

        if epoch >= noise_start_epoch:
            hooks.update_error_statistics(model, calib_loader, device, epoch, noise_start_epoch)
        hooks.register(model)

        model.train()
        if args.freeze_bn_in_noise_phase and hooks.error_intensity > 0:
            freeze_batchnorm_only(model)

        total_loss = 0.0
        total_seen = 0
        iterator = tqdm(train_loader, desc="Train", leave=False)
        for inputs, labels in iterator:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = utils.smooth_crossentropy(logits, labels, smoothing=args.label_smoothing).mean()
            loss.backward()
            optimizer.step()
            if epoch < swa_start_epoch:
                scheduler.step()
            total_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)
            iterator.set_postfix(loss=f"{total_loss / max(total_seen, 1):.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.6g}")

        hooks.remove_hooks()
        hooks.remove_residual_weight_error(model)

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        eval_model = swa_model if epoch >= swa_start_epoch else model
        if epoch >= swa_start_epoch and args.swa_bn_batches > 0:
            utils.update_bn(train_loader, eval_model, device=device, max_batches=args.swa_bn_batches)

        acc = utils.classification_accuracy(eval_model, val_loader, device)
        print(f"Validation accuracy: {acc:.2f}%")

        state_model = eval_model.module if isinstance(eval_model, AveragedModel) else eval_model
        acc_tag = f"{acc:.2f}".replace(".", "p")
        filename = f"epoch_{epoch + 1:03d}_acc_{acc_tag}_w{args.w_bits}a{args.a_bits}_{args.model_arch}.pt"
        path = os.path.join(args.save_dir, filename)
        torch.save(state_model.state_dict(), path)
        print(f"Saved: {path}")

        if acc > best_acc:
            best_acc = acc
            best_path = os.path.join(args.save_dir, f"best_w{args.w_bits}a{args.a_bits}_{args.model_arch}.pt")
            torch.save(state_model.state_dict(), best_path)
            print(f"Best updated: {best_acc:.2f}% -> {best_path}")

    print(f"Training done. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
