from __future__ import annotations

import os
import random
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import update_bn as swa_update_bn


def seed_everything(seed: int | None = None) -> int:
    if seed is None:
        seed = random.randint(1, 2000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_state_dict_file(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return strip_module_prefix(ckpt)


def smooth_crossentropy(pred: torch.Tensor, target: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    if smoothing <= 0.0:
        return F.cross_entropy(pred, target, reduction="none")

    n_class = pred.size(1)
    log_prob = F.log_softmax(pred, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(smoothing / (n_class - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    return -(true_dist * log_prob).sum(dim=1)


def update_bn(loader, model, device=None, max_batches: int | None = None):
    if max_batches is None:
        return swa_update_bn(loader, model, device=device)

    class LimitedLoader:
        def __iter__(self):
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                yield batch

    return swa_update_bn(LimitedLoader(), model, device=device)


@torch.no_grad()
def quantize_weight_symmetric_per_channel(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Signed symmetric per-output-channel quant-dequant for Conv2d/Linear weights."""
    if bits >= 32:
        return x.clone()
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    if qmax <= 0:
        raise ValueError(f"Invalid weight bit-width: {bits}")

    if x.dim() == 4:
        reduce_dims = (1, 2, 3)
    elif x.dim() == 2:
        reduce_dims = (1,)
    else:
        reduce_dims = tuple(range(x.dim()))

    max_abs = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.clamp(max_abs / float(qmax), min=1e-8)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale


@torch.no_grad()
def quantize_activation_asymmetric_per_tensor(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Unsigned asymmetric per-tensor quant-dequant for activations."""
    if bits >= 32:
        return x.clone()
    qmin, qmax = 0, 2 ** bits - 1
    x_min = x.detach().amin()
    x_max = x.detach().amax()
    if torch.isclose(x_min, x_max):
        return x.clone()
    scale = torch.clamp((x_max - x_min) / float(qmax - qmin), min=1e-8)
    zero_point = torch.clamp(torch.round(qmin - x_min / scale), qmin, qmax)
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    return (q - zero_point) * scale


@torch.no_grad()
def simulate_quantized_weights(
    model: nn.Module,
    bits: int,
    target_layers: Iterable[type],
) -> Dict[str, torch.Tensor]:
    q_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, tuple(target_layers)) and hasattr(module, "weight") and module.weight is not None:
            q_weights[name] = quantize_weight_symmetric_per_channel(module.weight.detach(), bits).cpu()
    return q_weights


@torch.no_grad()
def calculate_weight_quantization_error(
    model: nn.Module,
    q_weights: Dict[str, torch.Tensor],
    target_layers: Iterable[type],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return channel-wise statistics of E_w = W_q - W."""
    mean_dict, var_dict = {}, {}
    for name, module in model.named_modules():
        if name not in q_weights:
            continue
        if not isinstance(module, tuple(target_layers)) or not hasattr(module, "weight"):
            continue
        fp_w = module.weight.detach()
        q_w = q_weights[name].to(fp_w.device)
        err = q_w - fp_w
        if err.dim() == 4:
            reduce_dims = (1, 2, 3)
        elif err.dim() == 2:
            reduce_dims = (1,)
        else:
            reduce_dims = tuple(range(err.dim()))
        mean_dict[name] = err.mean(dim=reduce_dims, keepdim=True).detach().cpu()
        var_dict[name] = err.var(dim=reduce_dims, keepdim=True, unbiased=False).detach().cpu()
    return mean_dict, var_dict


@torch.no_grad()
def calculate_activation_quantization_error_stats(
    model: nn.Module,
    calib_loader,
    bits: int,
    act_names,
    device,
    num_batches: int,
    task_type: str = "classification",
):
    """Return per-tensor activation statistics of E_a = A_q - A."""
    stats = {"mean": {}, "variance": {}}
    was_training = model.training
    model.eval()

    sums: Dict[str, float] = {}
    sq_sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    hooks = []
    act_name_set = set(act_names)

    def make_hook(layer_name):
        def hook_fn(module, inputs, output):
            if not torch.is_tensor(output):
                return
            act = output.detach()
            q_act = quantize_activation_asymmetric_per_tensor(act, bits)
            err = q_act - act
            sums[layer_name] = sums.get(layer_name, 0.0) + err.sum().item()
            sq_sums[layer_name] = sq_sums.get(layer_name, 0.0) + (err * err).sum().item()
            counts[layer_name] = counts.get(layer_name, 0) + err.numel()

        return hook_fn

    for name, module in model.named_modules():
        if name in act_name_set:
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        for batch_idx, batch in enumerate(calib_loader):
            if batch_idx >= num_batches:
                break
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(inputs.to(device, non_blocking=True))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    for name, n in counts.items():
        if n <= 0:
            continue
        mean = sums[name] / n
        ex2 = sq_sums[name] / n
        var = max(ex2 - mean * mean, 1e-12)
        stats["mean"][name] = torch.tensor(mean, dtype=torch.float32)
        stats["variance"][name] = torch.tensor(var, dtype=torch.float32)

    return stats


@torch.no_grad()
def classification_accuracy(model, loader, device) -> float:
    was_training = model.training
    model.eval()
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        pred = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    model.train(was_training)
    return 100.0 * correct / max(total, 1)

