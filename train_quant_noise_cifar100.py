# train_quant_noise.py
import argparse
import time
import random
import os
import copy
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F  # [Added] 需要用到 softmax
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import model.resnet as resnet
import model.mobilenetv1 as mobilenetv1
import model.mobilenetv2 as mobilenetv2
import utils 


current_weight_noise_mean, current_weight_noise_variance = {}, {}
ema_weight_noise_mean, ema_weight_noise_variance = {}, {}
previous_step_weight_noise_per_layer = {}
weight_noise_hooks_handles = []
TARGET_WEIGHT_NOISE_LAYERS = (nn.Conv2d, nn.Linear)

# 激活噪声
current_activation_noise_mean_ema, current_activation_noise_variance_ema = {}, {}
activation_noise_hooks_handles = []
TARGET_ACTIVATION_NOISE_LAYERS = (nn.ReLU, nn.ReLU6)

# 通用
current_noise_ramp_factor = 0.0

# --- 噪声注入的 Hook 函数 ---

def training_weight_noise_hook(module, inputs, layer_name):
    """权重差分噪声注入钩子 (前向预挂钩)"""
    global current_weight_noise_mean, current_weight_noise_variance, previous_step_weight_noise_per_layer
    if not module.training or current_noise_ramp_factor == 0.0: return
    if not current_weight_noise_mean: return

    if layer_name in current_weight_noise_mean:
        try:
            original_weight = module.weight.data
            device_w = original_weight.device
            noise_mean = current_weight_noise_mean[layer_name].to(device_w)
            noise_var = current_weight_noise_variance[layer_name].to(device_w)
            
            # 应用强度因子
            scaled_mean = noise_mean * current_noise_ramp_factor
            scaled_std = torch.sqrt(torch.clamp(noise_var, min=1e-10)) * current_noise_ramp_factor
            scaled_std = torch.clamp(scaled_std, min=1e-10 if current_noise_ramp_factor > 0 else 0)

            shape = (-1, 1, 1, 1) if original_weight.dim() == 4 else (-1, 1)
            if scaled_mean.dim() > 1: shape = None
                
            if shape:
                scaled_mean = scaled_mean.view(shape)
                scaled_std = scaled_std.view(shape)

            current_noise = torch.randn_like(original_weight) * scaled_std + scaled_mean
            prev_noise = previous_step_weight_noise_per_layer.get(layer_name, torch.zeros_like(original_weight, device=device_w))
            
            module.weight.data += (current_noise - prev_noise)
            previous_step_weight_noise_per_layer[layer_name] = current_noise.clone()
        except Exception: pass

def training_activation_noise_hook(module, inputs, output, layer_name, noise_density, strategy='random', temperature=1.0):
    global current_activation_noise_mean_ema, current_activation_noise_variance_ema
    
    # 1. 基础检查
    if not module.training or current_noise_ramp_factor == 0.0: return output
    if layer_name not in current_activation_noise_mean_ema: return output

    try:
        # --- A. 生成基础噪声 (模拟量化误差) ---
        mean = current_activation_noise_mean_ema[layer_name].to(output.device)
        var = current_activation_noise_variance_ema[layer_name].to(output.device)
        
        # 应用强度预热因子
        scaled_mean = mean * current_noise_ramp_factor
        scaled_std = torch.sqrt(torch.clamp(var, min=1e-10)) * current_noise_ramp_factor
        scaled_std = torch.clamp(scaled_std, min=1e-10 if current_noise_ramp_factor > 0 else 0)
        
        noise = torch.randn_like(output) * scaled_std + scaled_mean
        
        # 如果密度 >= 1，全量注入
        if noise_density >= 1.0: return output + noise
            
        # --- B. 计算注入 Mask ---
        if strategy == 'random':
            mask = torch.rand_like(output) < noise_density
            return torch.where(mask, output + noise, output)
            
        elif strategy == 'salience':
            # === [核心修复] AQES Channel-Wise 逻辑 ===
            
            # 1. 确定计算维度 N
            if output.dim() == 4: # Conv2d [B, C, H, W]
                # 计算每个通道的重要性 (绝对值的均值) -> [B, C, 1, 1]
                # 这样 N 就是通道数 (比如 64)，而不是像素数 (65536)
                importance = output.abs().mean(dim=(2, 3), keepdim=True)
                N = output.size(1) 
            else: # Linear [B, N]
                importance = output.abs()
                N = output.size(1)
            
            # 2. 温度缩放 + 数值稳定保护
            # [重要] 之前你的代码里漏了除以 temperature！
            val = importance / temperature
            
            # 减去最大值防止 exp 溢出 (Log-Sum-Exp 技巧)
            val_max, _ = torch.max(val, dim=1, keepdim=True)
            scores = F.softmax(val - val_max.detach(), dim=1) # [B, C, 1, 1]
            
            # 3. 广播回原图大小 (让同一个通道共享相同的显著性分数)
            if output.dim() == 4:
                scores = scores.expand_as(output)
            
            # 4. 计算概率: p = score * N * rho
            probs = torch.clamp(scores * N * noise_density, 0.0, 1.0)
            
            # 5. 采样
            mask = torch.bernoulli(probs).bool()
            return torch.where(mask, output + noise, output)
            
        else:
            return output

    except Exception as e:
        return output

# --- 主训练与评估流程 ---
def main():
    parser = argparse.ArgumentParser(description="AQES 训练脚本")
    # --- 模型与超参数 ---
    parser.add_argument("--model_arch", default="resnet18", type=str, choices=["resnet18", "resnet50", "mobilenet_v1", "mobilenet_v2"], help="模型架构")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--learning_rate", default=0.015, type=float, help="初始学习率")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="权重衰减")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--swa_rate", default=0.75, type=float, help="开始SWA的epoch比例")
    
    # --- 噪声控制参数 ---
    parser.add_argument("--noise_start_epoch_frac", default=0.5, type=float, help="开始噪声注入的epoch比例")
    parser.add_argument("--noise_ramp_up_epochs", default=80, type=int, help="噪声强度预热的轮数")
    parser.add_argument("--noise_max_intensity", default=0.8, type=float, help="噪声强度的最大比例上限 [0.0, 1.0]")
    
    # --- AQES 核心消融参数 (New) ---
    parser.add_argument("--aqes_strategy", default="random", type=str, choices=['random', 'salience'], 
                        help="激活噪声注入策略: 'random' (Baseline) 或 'salience' (Ours)")
    parser.add_argument("--aqes_temperature", default=1.0, type=float, help="AQES 温度 T")
    
    # --- 权重量化噪声 ---
    parser.add_argument("--inject_differential_weight_noise", action='store_true', help="启用差分权重量化噪声")
    parser.add_argument("--quant_w_bits", default=2, type=int)
    parser.add_argument("--ema_decay_weight", default=0.9, type=float)

    # --- 激活量化噪声 ---
    parser.add_argument("--inject_activation_noise", action='store_true', help="启用激活量化噪声")
    parser.add_argument("--quant_a_bits", default=4, type=int)
    parser.add_argument("--num_calib_batches_act", default=4, type=int)
    parser.add_argument("--ema_decay_act", default=0.9, type=float)
    parser.add_argument("--activation_noise_application_prob", default=0.5, type=float, help="噪声注入密度 (rho)")
    
    args = parser.parse_args()
    if not (0.0 <= args.noise_max_intensity <= 1.0): raise ValueError("noise_max_intensity 必须在 [0.0, 1.0] 之间")

    # --- 初始化 ---
    seed = random.randint(1, 2000)
    print(f"**Seed**: {seed}")
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"**Using device**: {device}")
    
    # --- 数据加载器 ---
    print("准备数据加载器...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
    
    # [注意] 请根据你的环境修改 root 路径
    full_train_dataset = datasets.CIFAR100(root='/home/xp/idea1/data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='/home/xp/idea1/data', train=False, download=True, transform=transform_test)
    num_classes = 100
    
    train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(os.cpu_count(), 4), pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(os.cpu_count(), 4), pin_memory=True)
    
    calib_loader_act = None
    if args.inject_activation_noise and args.num_calib_batches_act > 0:
        num_samples = args.num_calib_batches_act * args.batch_size
        indices = np.random.choice(len(full_train_dataset), min(num_samples, len(full_train_dataset)), replace=False)
        calib_loader_act = DataLoader(Subset(full_train_dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- 模型、优化器、调度器 ---
    print("创建模型...")
    MODEL_ZOO = {
        "resnet18": resnet.ResNet18, "resnet50": resnet.ResNet50, "mobilenet_v1": mobilenetv1.MobileNetV1_CIFAR, "mobilenet_v2": mobilenetv2.MobileNetV2_CIFAR
    }
    model_creator_fn = MODEL_ZOO[args.model_arch]
    model = model_creator_fn(num_classes=num_classes).to(device)
    print(f"**已创建模型**: {args.model_arch}")

    SWA_START_EPOCH = int(args.epochs * args.swa_rate)
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    
    t_max_cosine = len(train_loader) * (SWA_START_EPOCH if SWA_START_EPOCH > 0 else args.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max_cosine if t_max_cosine > 0 else 1)
    
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.learning_rate * 0.01)

    # --- 训练循环 ---
    best_acc = 0.0
    global current_noise_ramp_factor
    global ema_weight_noise_mean, ema_weight_noise_variance, current_weight_noise_mean, current_weight_noise_variance, previous_step_weight_noise_per_layer
    global ema_activation_noise_mean, ema_activation_noise_variance, current_activation_noise_mean_ema, current_activation_noise_variance_ema
    
    NOISE_START_EPOCH = int(args.epochs * args.noise_start_epoch_frac)
    
    # 打印当前配置
    if args.inject_activation_noise:
        print(f"** Activation Noise Config **: Strategy={args.aqes_strategy}, T={args.aqes_temperature}, Rho={args.activation_noise_application_prob}")

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # 1. 更新噪声强度因子
        base_ramp = 0.0
        if epoch >= NOISE_START_EPOCH:
            if args.noise_ramp_up_epochs <= 1: base_ramp = 1.0
            else:
                epochs_in = epoch - NOISE_START_EPOCH
                base_ramp = min(1.0, (epochs_in + 1) / args.noise_ramp_up_epochs)
        current_noise_ramp_factor = base_ramp * args.noise_max_intensity
        if base_ramp > 0: print(f"  当前噪声强度预热因子: {current_noise_ramp_factor:.3f}")
        
        # 2. 如果需要，计算和更新噪声统计量
        if epoch >= NOISE_START_EPOCH:
            # -- 权重量化噪声 --
            if args.inject_differential_weight_noise:
                fp_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                q_weights = utils.simulate_quantized_weights(fp_dict, model_creator_fn, num_classes, args.quant_w_bits, TARGET_WEIGHT_NOISE_LAYERS, device)
                if q_weights:
                    mean, var = utils.calculate_weight_quantization_noise(model, q_weights, TARGET_WEIGHT_NOISE_LAYERS, device)
                    for k in mean:
                        ema_weight_noise_mean[k] = args.ema_decay_weight * ema_weight_noise_mean.get(k, mean[k]) + (1 - args.ema_decay_weight) * mean[k]
                        ema_weight_noise_variance[k] = args.ema_decay_weight * ema_weight_noise_variance.get(k, var[k]) + (1 - args.ema_decay_weight) * var[k]
                    current_weight_noise_mean, current_weight_noise_variance = ema_weight_noise_mean, ema_weight_noise_variance
                    if current_noise_ramp_factor > 0: previous_step_weight_noise_per_layer = {}
            
            # -- 激活量化噪声 --
            if args.inject_activation_noise and calib_loader_act:
                act_names = [name for name, mod in model.named_modules() if isinstance(mod, TARGET_ACTIVATION_NOISE_LAYERS)]
                if act_names:
                    act_stats = utils.calculate_activation_quantization_noise_stats(model, calib_loader_act, args.quant_a_bits, act_names, device, args.num_calib_batches_act)
                    if act_stats and act_stats['mean']:
                        for k in act_stats['mean']:
                            current_activation_noise_mean_ema[k] = args.ema_decay_act * current_activation_noise_mean_ema.get(k, act_stats['mean'][k]) + (1 - args.ema_decay_act) * act_stats['mean'][k]
                            current_activation_noise_variance_ema[k] = args.ema_decay_act * current_activation_noise_variance_ema.get(k, act_stats['variance'][k]) + (1 - args.ema_decay_act) * act_stats['variance'][k]

        # 3. 注册或移除 Hooks
        for h in weight_noise_hooks_handles + activation_noise_hooks_handles: h.remove()
        weight_noise_hooks_handles.clear(); activation_noise_hooks_handles.clear()
        
        if current_noise_ramp_factor > 0:
            # 注册权重噪声 Hook
            if args.inject_differential_weight_noise:
                for name, mod in model.named_modules():
                    if isinstance(mod, TARGET_WEIGHT_NOISE_LAYERS) and name in current_weight_noise_mean:
                        weight_noise_hooks_handles.append(mod.register_forward_pre_hook(functools.partial(training_weight_noise_hook, layer_name=name)))
            
            # 注册激活噪声 Hook (使用新的参数)
            if args.inject_activation_noise:
                for name, mod in model.named_modules():
                    if isinstance(mod, TARGET_ACTIVATION_NOISE_LAYERS) and name in current_activation_noise_mean_ema:
                        # [Modified] 传入 strategy 和 temperature
                        activation_noise_hooks_handles.append(mod.register_forward_hook(
                            functools.partial(training_activation_noise_hook, 
                                              layer_name=name, 
                                              noise_density=args.activation_noise_application_prob,
                                              strategy=args.aqes_strategy,
                                              temperature=args.aqes_temperature)
                        ))

        # 4. 训练一个 Epoch
        model.train()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        for inputs, labels in train_iterator:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = utils.smooth_crossentropy(outputs, labels, smoothing=args.label_smoothing).mean()
            loss.backward()
            optimizer.step()
            if epoch < SWA_START_EPOCH: scheduler.step()
            train_iterator.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
        # 5. SWA 更新
        if epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # 6. 评估
        eval_model = swa_model if epoch >= SWA_START_EPOCH else model
        if epoch >= SWA_START_EPOCH: utils.update_bn(train_loader, eval_model, device)
        eval_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = eval_model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] **Validation Accuracy**: {acc:.2f}%")

        # 7. 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            # 自动创建 saved_models 目录
            save_dir = "saved_models"
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成带策略和参数的文件名
            noise_suffix = f"_{args.aqes_strategy}"
            if args.aqes_strategy == 'salience':
                noise_suffix += f"_T{args.aqes_temperature}"
            
            filename = f"best_{args.model_arch}{noise_suffix}_w{args.quant_w_bits}a{args.quant_a_bits}.pt"
            save_path = os.path.join(save_dir, filename)
            
            torch.save(eval_model.state_dict(), save_path)
            print(f"最佳模型已更新 (Acc: {best_acc:.2f}%) 并保存至 {save_path}")

    print(f"**训练完成。最佳验证准确率**: {best_acc:.2f}%")

if __name__ == "__main__":
    if not os.path.isdir('model'):
        print("关键警告: 'model' 目录未找到。请确保该目录存在，并包含所需的模型文件。")
    else:
        main()
