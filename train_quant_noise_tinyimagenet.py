import argparse
import time
import random
import os
import copy
import functools
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as tv_models
import timm
import utils

# ==========================================
# 1. AQES 全局变量
# ==========================================
current_weight_noise_mean, current_weight_noise_variance = {}, {}
ema_weight_noise_mean, ema_weight_noise_variance = {}, {}
previous_step_weight_noise_per_layer = {}
weight_noise_hooks_handles = []
TARGET_WEIGHT_NOISE_LAYERS = (nn.Conv2d, nn.Linear)

current_activation_noise_mean_ema, current_activation_noise_variance_ema = {}, {}
activation_noise_hooks_handles = []
TARGET_ACTIVATION_NOISE_LAYERS = (nn.ReLU, nn.ReLU6)

current_noise_ramp_factor = 0.0

# ==========================================
# 2. 自定义 Dataset (针对 Tiny-ImageNet val)
# ==========================================
class TinyImageNetValDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.val_dir = os.path.join(root_dir, 'val')
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.class_to_idx = class_to_idx
        if self.class_to_idx is None:
            # 兼容老逻辑（以防万一）
            wnids_path = os.path.join(root_dir, 'wnids.txt')
            with open(wnids_path, 'r') as f:
                wnids = [line.strip() for line in f.readlines()]
            self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
        
        
        val_anno_path = os.path.join(self.val_dir, 'val_annotations.txt')
        with open(val_anno_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                img_name = parts[0]
                wnid = parts[1]
                if wnid in self.class_to_idx:
                    self.image_paths.append(os.path.join(self.val_dir, 'images', img_name))
                    self.labels.append(self.class_to_idx[wnid])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ==========================================
# 3. 噪声注入 Hooks
# ==========================================
def training_weight_noise_hook(module, inputs, layer_name):
    global current_weight_noise_mean, current_weight_noise_variance, previous_step_weight_noise_per_layer
    if not module.training or current_noise_ramp_factor == 0.0: return
    if not current_weight_noise_mean: return

    if layer_name in current_weight_noise_mean:
        try:
            original_weight = module.weight.data
            device_w = original_weight.device
            noise_mean = current_weight_noise_mean[layer_name].to(device_w)
            noise_var = current_weight_noise_variance[layer_name].to(device_w)
            
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

def training_activation_noise_hook(module, inputs, output, layer_name, prob):
    global current_activation_noise_mean_ema, current_activation_noise_variance_ema
    if not module.training or current_noise_ramp_factor == 0.0: return output
    if layer_name in current_activation_noise_mean_ema:
        try:
            mean = current_activation_noise_mean_ema[layer_name].to(output.device)
            var = current_activation_noise_variance_ema[layer_name].to(output.device)
            
            scaled_mean = mean * current_noise_ramp_factor
            scaled_std = torch.sqrt(torch.clamp(var, min=1e-10)) * current_noise_ramp_factor
            scaled_std = torch.clamp(scaled_std, min=1e-10 if current_noise_ramp_factor > 0 else 0)
            
            noise = torch.randn_like(output) * scaled_std + scaled_mean
            
            if prob < 1.0:
                mask = torch.rand_like(output) < prob
                return torch.where(mask, output + noise, output)
            else:
                return output + noise
        except Exception: return output
    return output

def _freeze_batchnorm_only(model):
    """训练时保持 model.train()，但冻结 BN 的 running 统计，减轻强噪声对 BN 的破坏。"""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

# ==========================================
# 4. 主流程 main()
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Tiny-ImageNet 预训练模型量化噪声注入训练脚本")
    
    # 路径与模型架构
    parser.add_argument("--data_dir", default="/home/xp/data/tiny-imagenet-200", type=str)
    parser.add_argument("--model_arch", default="resnet18", type=str,
    choices=["resnet18", "resnet50", "mobilenet_v2",
             "ghostnet_100", "ghostnet_130", "ghostnetv2_100"])
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="从已有全精度 state_dict (.pt) 加载权重；通常与 --noise_only 联用，不再从 ImageNet 预训练初始化",
    )
    parser.add_argument(
        "--noise_only",
        action="store_true",
        help="仅做加噪微调：从第 1 个 epoch 起即进入噪声阶段，--epochs 表示加噪总轮数（需 --load_checkpoint，且至少开一种噪声注入）",
    )
    # 基础超参数
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--swa_rate", default=0.9, type=float)
    
    # 噪声控制参数 (默认更保守：慢 ramp + 降低满量程，避免 GhostNet 等结构在噪声阶段崩溃)
    parser.add_argument("--noise_start_epoch_frac", default=0.8, type=float)
    parser.add_argument("--noise_ramp_up_epochs", default=80, type=int)
    parser.add_argument("--noise_max_intensity", default=0.5, type=float)
    parser.add_argument("--freeze_bn_in_noise_phase", action="store_true",
                        help="噪声阶段训练时冻结 BN 的 running 统计（仍更新 Conv/Linear 权重）")
    
    # 权重量化参数
    parser.add_argument("--inject_differential_weight_noise", action='store_true')
    parser.add_argument("--quant_w_bits", default=2, type=int)
    parser.add_argument("--ema_decay_weight", default=0.9, type=float)

    # 激活量化参数
    parser.add_argument("--inject_activation_noise", action='store_true')
    parser.add_argument("--quant_a_bits", default=4, type=int)
    parser.add_argument("--num_calib_batches_act", default=16, type=int)
    parser.add_argument("--ema_decay_act", default=0.9, type=float)
    parser.add_argument("--activation_noise_application_prob", default=0.25, type=float)
    
    args = parser.parse_args()
    if not (0.0 <= args.noise_max_intensity <= 1.0): raise ValueError("noise_max_intensity 必须在 [0.0, 1.0] 之间")
    if args.noise_only:
        if not args.load_checkpoint:
            raise ValueError("--noise_only 需要指定 --load_checkpoint（例如 best_clean_*_fp32.pt）")
        if not (args.inject_differential_weight_noise or args.inject_activation_noise):
            raise ValueError("--noise_only 需要至少开启 --inject_differential_weight_noise 或 --inject_activation_noise")

    # 初始化设备与种子
    seed = random.randint(1, 2000)
    print(f"**Seed**: {seed}")
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"**Using device**: {device}")
    
    # 数据加载 (Resize 到 224 以适配 ImageNet 预训练模型)
    print("准备 Tiny-ImageNet 数据加载器...")
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    train_dir = os.path.join(args.data_dir, 'train')
    full_train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = TinyImageNetValDataset(
        args.data_dir, 
        transform=transform_test, 
        class_to_idx=full_train_dataset.class_to_idx 
    )
    num_classes = 200 
    
    train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(os.cpu_count(), 4), pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(os.cpu_count(), 4), pin_memory=True)
    
    calib_loader_act = None
    if args.inject_activation_noise and args.num_calib_batches_act > 0:
        num_samples = args.num_calib_batches_act * args.batch_size
        indices = np.random.choice(len(full_train_dataset), min(num_samples, len(full_train_dataset)), replace=False)
        calib_loader_act = DataLoader(Subset(full_train_dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 模型加载逻辑
    use_pretrained_init = not args.load_checkpoint
    print("构建模型结构..." + ("" if use_pretrained_init else "（将从 checkpoint 加载权重，跳过 ImageNet 预训练初始化）"))
    def get_pretrained_model(num_classes=200, pretrained=True):
        if args.model_arch == "resnet18":
            m = tv_models.resnet18(pretrained=pretrained)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        elif args.model_arch == "resnet50":
            m = tv_models.resnet50(pretrained=pretrained)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        elif args.model_arch == "mobilenet_v2":
            m = tv_models.mobilenet_v2(pretrained=pretrained)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

        elif args.model_arch in ("ghostnet_100", "ghostnet_130", "ghostnetv2_100"):
            m = timm.create_model(
                args.model_arch,
                pretrained=pretrained,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"不支持的模型架构: {args.model_arch}")
        return m
        
    model = get_pretrained_model(num_classes=num_classes, pretrained=use_pretrained_init).to(device)
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"  [load_checkpoint] 未匹配的键 (missing 前 8 个): {list(missing)[:8]}")
        if unexpected:
            print(f"  [load_checkpoint] 多余键 (unexpected 前 8 个): {list(unexpected)[:8]}")
        print(f"**已从 checkpoint 加载权重**: {args.load_checkpoint}")
    else:
        print(f"**已加载预训练模型**: {args.model_arch}, 分类数: {num_classes}")

    # 优化器与调度器
    SWA_START_EPOCH = int(args.epochs * args.swa_rate)
    
    optimizer = SGD([
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 
        'lr': args.learning_rate * 0.1},   # backbone 用 0.0001
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 
        'lr': args.learning_rate}           # 分类头用 0.001
    ], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    t_max_cosine = len(train_loader) * (SWA_START_EPOCH if SWA_START_EPOCH > 0 else args.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max_cosine if t_max_cosine > 0 else 1)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.learning_rate * 0.01)

    # 训练与双重保存状态
    best_acc_clean = 0.0 
    best_acc_noise = 0.0 
    
    global current_noise_ramp_factor
    global ema_weight_noise_mean, ema_weight_noise_variance, current_weight_noise_mean, current_weight_noise_variance, previous_step_weight_noise_per_layer
    global ema_activation_noise_mean, ema_activation_noise_variance, current_activation_noise_mean_ema, current_activation_noise_variance_ema
    
    NOISE_START_EPOCH = 0 if args.noise_only else int(args.epochs * args.noise_start_epoch_frac)
    if args.noise_only:
        print(f"**noise_only**：仅加噪微调，共 {args.epochs} 个 epoch，噪声从 epoch 1 开始 ramp")
    
    for epoch in range(args.epochs):
        is_noise_phase = (epoch >= NOISE_START_EPOCH) and (args.inject_differential_weight_noise or args.inject_activation_noise)
        phase_name = "Noise Stage" if is_noise_phase else "Clean SGD Stage"
        print(f"\n--- Epoch {epoch+1}/{args.epochs} [{phase_name}] ---")
        
        # 1. 更新噪声强度
        base_ramp = 0.0
        if epoch >= NOISE_START_EPOCH:
            if args.noise_ramp_up_epochs <= 1: base_ramp = 1.0
            else:
                epochs_in = epoch - NOISE_START_EPOCH
                base_ramp = min(1.0, (epochs_in + 1) / args.noise_ramp_up_epochs)
        current_noise_ramp_factor = base_ramp * args.noise_max_intensity
        if base_ramp > 0: print(f"  当前噪声强度预热因子: {current_noise_ramp_factor:.3f}")
        
        # 2. 统计噪声
        if epoch >= NOISE_START_EPOCH:
            # -- 权重量化 --
            if args.inject_differential_weight_noise:
                fp_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                q_weights = utils.simulate_quantized_weights(
                    fp_dict,
                    lambda num_classes: get_pretrained_model(num_classes, pretrained=False),
                    num_classes, args.quant_w_bits, TARGET_WEIGHT_NOISE_LAYERS, device
                )
                if q_weights:
                    mean, var = utils.calculate_weight_quantization_noise(model, q_weights, TARGET_WEIGHT_NOISE_LAYERS, device)
                    for k in mean:
                        ema_weight_noise_mean[k] = args.ema_decay_weight * ema_weight_noise_mean.get(k, mean[k]) + (1 - args.ema_decay_weight) * mean[k]
                        ema_weight_noise_variance[k] = args.ema_decay_weight * ema_weight_noise_variance.get(k, var[k]) + (1 - args.ema_decay_weight) * var[k]
                    current_weight_noise_mean, current_weight_noise_variance = ema_weight_noise_mean, ema_weight_noise_variance
                    # 仅在「刚进入噪声阶段」时清空差分状态；若每个 epoch 都清空，第一个 batch 会等价于一次性加满噪声，易训练崩溃
                    if current_noise_ramp_factor > 0 and epoch == NOISE_START_EPOCH:
                        previous_step_weight_noise_per_layer = {}
            
            # -- 激活量化 (已修复语法错误) --
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
            if args.inject_differential_weight_noise:
                for name, mod in model.named_modules():
                    if isinstance(mod, TARGET_WEIGHT_NOISE_LAYERS) and name in current_weight_noise_mean:
                        weight_noise_hooks_handles.append(mod.register_forward_pre_hook(functools.partial(training_weight_noise_hook, layer_name=name)))
            if args.inject_activation_noise:
                for name, mod in model.named_modules():
                    if isinstance(mod, TARGET_ACTIVATION_NOISE_LAYERS) and name in current_activation_noise_mean_ema:
                        activation_noise_hooks_handles.append(mod.register_forward_hook(functools.partial(training_activation_noise_hook, layer_name=name, prob=args.activation_noise_application_prob)))

        # 4. 训练
        model.train()
        if args.freeze_bn_in_noise_phase and is_noise_phase:
            _freeze_batchnorm_only(model)
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
            
        for h in weight_noise_hooks_handles + activation_noise_hooks_handles: 
            h.remove()
        weight_noise_hooks_handles.clear()
        activation_noise_hooks_handles.clear()
        
        # 5. SWA 
        if epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # 6. 评估
        eval_model = swa_model if epoch >= SWA_START_EPOCH else model
        if epoch >= SWA_START_EPOCH: 
            mini_loader = DataLoader(Subset(full_train_dataset, range(0, args.batch_size * 10)), batch_size=args.batch_size, num_workers=2)
            utils.update_bn(mini_loader, eval_model, device)
        eval_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Validation", leave=False):
                outputs = eval_model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] **Validation Accuracy**: {acc:.2f}%")

        # 7. 双重保存机制
        save_dir = "../trained_tiny_imagenet_pretrained/"
        os.makedirs(save_dir, exist_ok=True)
        
        if not is_noise_phase:
            if acc > best_acc_clean:
                best_acc_clean = acc
                filename_clean = f"best_clean_{args.model_arch}_fp32.pt"
                torch.save(eval_model.state_dict(), os.path.join(save_dir, filename_clean))
                print(f"  [Clean SGD 阶段] 最佳 Baseline 模型已更新 (Acc: {best_acc_clean:.2f}%) -> {filename_clean}")
        else:
            if acc > best_acc_noise:
                best_acc_noise = acc
                filename_noise = f"best_noise_w{args.quant_w_bits}_a{args.quant_a_bits}_{args.model_arch}.pt"
                torch.save(eval_model.state_dict(), os.path.join(save_dir, filename_noise))
                print(f"  [Noise 阶段] 最佳加噪模型已更新 (Acc: {best_acc_noise:.2f}%) -> {filename_noise}")

    print(f"\n**训练全部完成**")
    if args.noise_only:
        print(f" -> 已跳过干净阶段（由 checkpoint 提供全精度起点）")
    else:
        print(f" -> Clean SGD 阶段最佳验证准确率: {best_acc_clean:.2f}%")
    if args.inject_differential_weight_noise or args.inject_activation_noise:
        print(f" -> Noise Injection 阶段最佳验证准确率: {best_acc_noise:.2f}%")

if __name__ == "__main__":
    main()
