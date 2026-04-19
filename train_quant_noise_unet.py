import argparse
import random
import os
import functools
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
# SWA
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets

# === 核心修改：引入 SMP 库 ===
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Error: 请先安装 segmentation_models_pytorch 库")
    print("pip install segmentation_models_pytorch")
    exit(1)

import utils 

# --- 全局变量 (AQES 核心逻辑) ---
current_weight_noise_mean, current_weight_noise_variance = {}, {}
ema_weight_noise_mean, ema_weight_noise_variance = {}, {}
previous_step_weight_noise_per_layer = {}
weight_noise_hooks_handles = []
# ResNet 里的卷积也是 Conv2d
TARGET_WEIGHT_NOISE_LAYERS = (nn.Conv2d, nn.Linear)

current_activation_noise_mean_ema, current_activation_noise_variance_ema = {}, {}
activation_noise_hooks_handles = []
# ResNet 使用 ReLU
TARGET_ACTIVATION_NOISE_LAYERS = (nn.ReLU, nn.ReLU6) 

current_noise_ramp_factor = 0.0

# --- Dataset Wrapper for Cityscapes ---
CITYSCAPES_MEAN = [0.485, 0.456, 0.406]
CITYSCAPES_STD = [0.229, 0.224, 0.225]

class CityscapesWrapper(datasets.Cityscapes):
    def __init__(self, root, split='train', target_type='semantic', crop_size=(512, 512)):
        super().__init__(root, split=split, mode='fine', target_type=target_type)
        self.crop_size = crop_size
        self.ignore_index = 255
        
        self.id_to_train_id = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
            26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
        }

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        
        # 1. Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.crop_size
        )
        image = TF.crop(image, i, j, h, w)
        target = TF.crop(target, i, j, h, w)

        # 2. Random Horizontal Flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)

        # 3. Transform Image
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD)

        # 4. Map Targets
        target = torch.from_numpy(np.array(target)).long()
        target_mapped = torch.ones_like(target) * self.ignore_index
        for k, v in self.id_to_train_id.items():
            target_mapped[target == k] = v
        
        return image, target_mapped

# --- Hooks (AQES) ---
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

def training_activation_noise_hook(module, inputs, output, layer_name, noise_density, strategy='random', temperature=1.0):
    global current_activation_noise_mean_ema, current_activation_noise_variance_ema
    if not module.training or current_noise_ramp_factor == 0.0: return output
    if layer_name not in current_activation_noise_mean_ema: return output
    try:
        mean = current_activation_noise_mean_ema[layer_name].to(output.device)
        var = current_activation_noise_variance_ema[layer_name].to(output.device)
        scaled_mean = mean * current_noise_ramp_factor
        scaled_std = torch.sqrt(torch.clamp(var, min=1e-10)) * current_noise_ramp_factor
        scaled_std = torch.clamp(scaled_std, min=1e-10 if current_noise_ramp_factor > 0 else 0)
        noise = torch.randn_like(output) * scaled_std + scaled_mean
        if noise_density >= 1.0: return output + noise
        if strategy == 'random':
            mask = torch.rand_like(output) < noise_density
            return torch.where(mask, output + noise, output)
        elif strategy == 'salience':
            if output.dim() == 4: 
                importance = output.abs().mean(dim=(2, 3), keepdim=True)
                N = output.size(1) 
            else: 
                importance = output.abs()
                N = output.size(1)
            val = importance / temperature
            val_max, _ = torch.max(val, dim=1, keepdim=True)
            scores = F.softmax(val - val_max.detach(), dim=1)
            if output.dim() == 4: scores = scores.expand_as(output)
            probs = torch.clamp(scores * N * noise_density, 0.0, 1.0)
            mask = torch.bernoulli(probs).bool()
            return torch.where(mask, output + noise, output)
        else: return output
    except Exception as e: return output

# --- Evaluation (mIoU) ---
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def evaluate_unet_miou(net, dataloader, device, num_classes):
    net.eval()
    hist = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for image, mask_true in tqdm(dataloader, desc='Calculating mIoU', leave=False):
            image = image.to(device, dtype=torch.float32)
            mask_pred = net(image)
            mask_pred = torch.argmax(mask_pred, dim=1)
            pred_np = mask_pred.cpu().numpy().flatten()
            label_np = mask_true.numpy().flatten()
            hist += fast_hist(label_np, pred_np, num_classes)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
    mean_iou = np.nanmean(iu)
    net.train()
    return mean_iou

# ==========================================
# 4. Main Function (三阶段独立保存版)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="AQES ResNet34-UNet Cityscapes")
    parser.add_argument("--n_channels", default=3, type=int)
    parser.add_argument("--num_classes", default=19, type=int) 
    parser.add_argument("--data_dir", default="/home/xp/data/cityscapes", help="Cityscapes根目录")
    
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--crop_h", default=512, type=int)
    parser.add_argument("--crop_w", default=512, type=int)
    
    parser.add_argument("--learning_rate", default=0.01, type=float)
    
    parser.add_argument("--epochs_phase1", default=300, type=int)
    parser.add_argument("--epochs_phase2", default=100, type=int)
    parser.add_argument("--epochs_phase3", default=100, type=int)
    parser.add_argument("--swa_lr", default=0.001, type=float)

    # AQES 参数
    parser.add_argument("--noise_ramp_up_epochs", default=10, type=int)
    parser.add_argument("--noise_max_intensity", default=0.8, type=float)
    parser.add_argument("--aqes_strategy", default="salience", choices=['random', 'salience'])
    parser.add_argument("--aqes_temperature", default=1.0, type=float)
    parser.add_argument("--inject_differential_weight_noise", action='store_true')
    parser.add_argument("--quant_w_bits", default=4, type=int)
    parser.add_argument("--ema_decay_weight", default=0.9, type=float)
    parser.add_argument("--inject_activation_noise", action='store_true')
    parser.add_argument("--quant_a_bits", default=4, type=int)
    parser.add_argument("--activation_noise_application_prob", default=0.5, type=float)
    parser.add_argument("--num_calib_batches_act", default=4, type=int)
    parser.add_argument("--ema_decay_act", default=0.9, type=float)
    
    # 断点续训
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"**Using device**: {device}")

    PHASE1_END = args.epochs_phase1
    PHASE2_END = args.epochs_phase1 + args.epochs_phase2
    TOTAL_EPOCHS = args.epochs_phase1 + args.epochs_phase2 + args.epochs_phase3
    
    print(f"Plan: Clean[0-{PHASE1_END}] -> Noise[{PHASE1_END}-{PHASE2_END}] -> Finetune/SWA[{PHASE2_END}-{TOTAL_EPOCHS}]")

    # 1. 创建模型 (SMP ResNet34 U-Net)
    def create_model_smp(n_channels, num_classes):
        print("Creating SMP U-Net with ResNet34 encoder (ImageNet pre-trained)...")
        return smp.Unet(
            encoder_name="resnet34",        
            encoder_weights="imagenet",     
            in_channels=n_channels,
            classes=num_classes,
            activation=None                 
        )

    model = create_model_smp(args.n_channels, args.num_classes)
    model.to(device)

    # 2. 加载权重
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            try:
                model.load_state_dict(checkpoint, strict=True)
                print(f"Loaded successfully (strict=True).")
            except Exception as e:
                print(f"Loading failed strict=True, trying strict=False. Error: {e}")
                model.load_state_dict(checkpoint, strict=False)
            print(f"Resuming with start_epoch: {args.start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # 3. 数据加载
    print("Loading Cityscapes dataset...")
    train_ds = CityscapesWrapper(root=args.data_dir, split='train', crop_size=(args.crop_h, args.crop_w))
    val_ds = CityscapesWrapper(root=args.data_dir, split='val', crop_size=(args.crop_h, args.crop_w))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train Size: {len(train_ds)}, Val Size: {len(val_ds)}")

    calib_loader_act = None
    if args.inject_activation_noise:
        calib_loader_act = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 4. 优化器
    print("Using SGD Optimizer")
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=8, factor=0.5, verbose=True)
    
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 5. 全局变量
    global current_noise_ramp_factor
    global ema_weight_noise_mean, ema_weight_noise_variance, current_weight_noise_mean, current_weight_noise_variance, previous_step_weight_noise_per_layer
    global current_activation_noise_mean_ema, current_activation_noise_variance_ema

    best_score = 0.0
    
    # 状态标志位，用于在切阶段时重置 best_score
    has_reset_for_phase2 = False 
    has_reset_for_phase3 = False

    # 6. 训练循环
    for epoch in range(args.start_epoch, TOTAL_EPOCHS):
        is_noise_phase = (epoch >= PHASE1_END) and (epoch < PHASE2_END)
        is_swa_phase = (epoch >= PHASE2_END)
        
        # [逻辑修改] Phase 2 重置逻辑
        if epoch >= PHASE1_END and epoch < PHASE2_END and not has_reset_for_phase2:
            print(f"\n>>> Entering Phase 2 (Noise Injection). Resetting Best Score (Prev Best: {best_score:.4f})")
            best_score = 0.0
            has_reset_for_phase2 = True
            
        # [逻辑修改] Phase 3 重置逻辑 (如果你想单独保存 Phase 3 的最佳模型)
        if epoch >= PHASE2_END and not has_reset_for_phase3:
            print(f"\n>>> Entering Phase 3 (Finetune/SWA). Resetting Best Score (Prev Best: {best_score:.4f})")
            best_score = 0.0
            has_reset_for_phase3 = True
        
        phase_name = "Clean"
        if is_noise_phase: phase_name = "Noise/AQES"
        if is_swa_phase: phase_name = "SWA/Finetune"

        print(f"\n--- Epoch {epoch+1}/{TOTAL_EPOCHS} [{phase_name}] ---")
        model.train()
        
        # --- AQES Noise Logic ---
        current_noise_ramp_factor = 0.0
        if is_noise_phase:
            epochs_in_noise = epoch - PHASE1_END
            base_ramp = 1.0
            if args.noise_ramp_up_epochs > 1:
                base_ramp = min(1.0, (epochs_in_noise + 1) / args.noise_ramp_up_epochs)
            current_noise_ramp_factor = base_ramp * args.noise_max_intensity
            print(f"  Noise Factor: {current_noise_ramp_factor:.3f}")

            # 权重噪声统计
            if args.inject_differential_weight_noise:
                fp_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                re_creator = lambda: create_model_smp(args.n_channels, args.num_classes)
                q_weights = utils.simulate_quantized_weights(fp_dict, re_creator, None, args.quant_w_bits, TARGET_WEIGHT_NOISE_LAYERS, device)
                if q_weights:
                    mean, var = utils.calculate_weight_quantization_noise(model, q_weights, TARGET_WEIGHT_NOISE_LAYERS, device)
                    for k in mean:
                        ema_weight_noise_mean[k] = args.ema_decay_weight * ema_weight_noise_mean.get(k, mean[k]) + (1 - args.ema_decay_weight) * mean[k]
                        ema_weight_noise_variance[k] = args.ema_decay_weight * ema_weight_noise_variance.get(k, var[k]) + (1 - args.ema_decay_weight) * var[k]
                    current_weight_noise_mean, current_weight_noise_variance = ema_weight_noise_mean, ema_weight_noise_variance
                    if current_noise_ramp_factor > 0: previous_step_weight_noise_per_layer = {}

            # 激活噪声统计
            if args.inject_activation_noise and calib_loader_act:
                 act_names = [name for name, mod in model.named_modules() if isinstance(mod, TARGET_ACTIVATION_NOISE_LAYERS)]
                 act_stats = utils.calculate_activation_quantization_noise_stats(model, calib_loader_act, args.quant_a_bits, act_names, device, args.num_calib_batches_act, task_type='segmentation')
                 if act_stats and act_stats['mean']:
                    for k in act_stats['mean']:
                        current_activation_noise_mean_ema[k] = args.ema_decay_act * current_activation_noise_mean_ema.get(k, act_stats['mean'][k]) + (1 - args.ema_decay_act) * act_stats['mean'][k]
                        current_activation_noise_variance_ema[k] = args.ema_decay_act * current_activation_noise_variance_ema.get(k, act_stats['variance'][k]) + (1 - args.ema_decay_act) * act_stats['variance'][k]

        # Register Hooks
        for h in weight_noise_hooks_handles + activation_noise_hooks_handles: h.remove()
        weight_noise_hooks_handles.clear(); activation_noise_hooks_handles.clear()

        if is_noise_phase and current_noise_ramp_factor > 0:
            if args.inject_differential_weight_noise:
                 for name, mod in model.named_modules():
                    if isinstance(mod, TARGET_WEIGHT_NOISE_LAYERS) and name in current_weight_noise_mean:
                        weight_noise_hooks_handles.append(mod.register_forward_pre_hook(functools.partial(training_weight_noise_hook, layer_name=name)))
            if args.inject_activation_noise:
                for name, mod in model.named_modules():
                    if isinstance(mod, TARGET_ACTIVATION_NOISE_LAYERS) and name in current_activation_noise_mean_ema:
                        activation_noise_hooks_handles.append(mod.register_forward_hook(functools.partial(training_activation_noise_hook, layer_name=name, noise_density=args.activation_noise_application_prob, strategy=args.aqes_strategy, temperature=args.aqes_temperature)))

        # Training Loop
        epoch_loss = 0
        with tqdm(total=len(train_ds), desc=f'Ep {epoch+1}', unit='img') as pbar:
            for images, true_masks in train_loader:
                images = images.to(device, dtype=torch.float32)
                true_masks = true_masks.to(device, dtype=torch.long)

                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)

                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss': loss.item()})
        
        # Validation & Saving (核心修改部分)
        if is_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"  SWA Update Done. LR: {swa_scheduler.get_last_lr()[0]:.6f}")
        
        # 注意：即使在 SWA 阶段，我们也跑一遍常规验证，保存该阶段最好的单体模型
        val_score = evaluate_unet_miou(model, val_loader, device, args.num_classes)
        
        # 如果不是 SWA 阶段，更新常规 Scheduler；如果是 SWA，Scheduler 已经由 swa_scheduler 接管
        if not is_swa_phase:
            scheduler.step(val_score)
            
        print(f"Validation mIoU: {val_score:.4f} (Current Phase Best: {best_score:.4f})")
        
        if val_score > best_score:
            best_score = val_score
            os.makedirs('checkpoints', exist_ok=True)
            
            # === 三阶段文件名区分 ===
            if epoch < PHASE1_END:
                # Phase 1: 纯净 SGD
                save_name = 'best_unet_resnet34_cityscapes_phase1_clean.pth'
                print(f"Saved [Phase 1 Clean] Best: {save_name}")
                
            elif epoch < PHASE2_END:
                # Phase 2: 加噪训练 (Noise)
                save_name = 'best_unet_resnet34_cityscapes_phase2_noise.pth'
                print(f"Saved [Phase 2 Noise] Best: {save_name}")
                
            else:
                # Phase 3: 收尾微调 (Finetune/SWA Stage)
                # 这里保存的是 SWA 阶段中的"单体最佳"，不是 SWA 平均模型
                save_name = 'best_unet_resnet34_cityscapes_phase3_finetune.pth'
                print(f"Saved [Phase 3 Finetune] Best: {save_name}")
            
            torch.save(model.state_dict(), f'checkpoints/{save_name}')

    # Final SWA Save
    print("\nTraining Finished. Processing SWA model...")
    print("Updating SWA BatchNorm statistics...")
    update_bn(train_loader, swa_model, device=device)
    
    print("Evaluating Final SWA model...")
    swa_val_score = evaluate_unet_miou(swa_model, val_loader, device, args.num_classes)
    print(f"Final SWA Validation mIoU: {swa_val_score:.4f}")
    
    # 最终的 SWA 模型 (平均后的模型)
    torch.save(swa_model.state_dict(), 'checkpoints/unet/w2a2/swa_unet_resnet34_cityscapes_final.pth')
    print("Saved Final SWA Model.")

if __name__ == '__main__':
    main()
