# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import functools
import torch.optim as optim

# --- 损失函数 ---
def smooth_crossentropy(logits, target, smoothing=0.1):
    """标签平滑交叉熵损失函数"""
    num_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - smoothing)
    loss = torch.sum(-true_dist * log_probs, dim=1)
    return loss

# --- SWA 相关工具 ---
def update_bn(loader, model, device=None, task_type='classification'):
    """为SWA模型更新批量归一化层的统计数据"""
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
    if not momenta:
        return
    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0
    with torch.no_grad():
        num_bn_update_batches = 20
        batch_count = 0
        bn_update_iter = tqdm(loader, total=min(num_bn_update_batches, len(loader)), desc="BN Update", leave=False, disable=True)
        if task_type == 'classification':
            for inputs, _ in bn_update_iter:
                if device is not None:
                    inputs = inputs.to(device)
                model(inputs)
                batch_count += 1
                if batch_count >= num_bn_update_batches:
                    break
        
        elif task_type == 'detection':
            for images, *_ in bn_update_iter: # 使用 *_ 忽略所有标签和元数据
                if device is not None:
                    images = images.to(device)
                model(images) # 只将图像传入模型
                batch_count += 1
                if batch_count >= num_bn_update_batches:
                    break
    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
    

# --- 量化噪声模拟与统计 ---
def simulate_quantized_weights(fp_model_state_dict, model_arch_fn, num_classes, w_bits, target_types_w, device_sim):
    """模拟权重的PTQ过程，生成量化后的权重"""
    simulated_q_weights = {}
    temp_model = model_arch_fn(num_classes=num_classes).to(device_sim)
    has_module_prefix = any(k.startswith('module.') for k in fp_model_state_dict.keys())
    if has_module_prefix:
        fp_model_state_dict = {k.replace("module.", ""): v for k, v in fp_model_state_dict.items()}
    try:
        temp_model.load_state_dict(fp_model_state_dict, strict=False)
    except RuntimeError as e:
        print(f"加载状态字典到临时模型时出错: {e}.")
        return {}
    temp_model.eval()
    q_min, q_max = -(2**(w_bits - 1)), 2**(w_bits - 1) - 1
    with torch.no_grad():
        for name, module in temp_model.named_modules():
            if isinstance(module, target_types_w) and hasattr(module, 'weight'):
                W_fp = module.weight.data
                try:
                    if isinstance(module, nn.Conv2d):
                        max_abs = W_fp.abs().amax(dim=(1, 2, 3), keepdim=True)
                        scale = max_abs / q_max
                        scale = torch.where(scale == 0, torch.tensor(1e-9, device=scale.device), scale)
                    elif isinstance(module, nn.Linear):
                        max_abs = W_fp.abs().max()
                        scale = max_abs / q_max
                        scale = torch.tensor(1e-9, device=scale.device) if scale == 0 else scale
                    else: continue
                    W_int = torch.round(W_fp / scale)
                    W_int_clamped = torch.clamp(W_int, q_min, q_max)
                    W_q = W_int_clamped * scale
                    simulated_q_weights[name] = W_q.cpu()
                except Exception as e_quant:
                    print(f"模拟量化层 '{name}' 时出错: {e_quant}")
                    if name in simulated_q_weights: del simulated_q_weights[name]
    del temp_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return simulated_q_weights

def calculate_weight_quantization_noise(fp_model, simulated_q_weights_dict, target_types_w, device_calc):
    """计算权重量化误差的均值和方差"""
    noise_mean_w, noise_variance_w = {}, {}
    fp_model.eval()
    with torch.no_grad():
        for name, fp_module in fp_model.named_modules():
            if isinstance(fp_module, target_types_w) and name in simulated_q_weights_dict:
                try:
                    fp_weight = fp_module.weight.data.to(device_calc)
                    q_weight = simulated_q_weights_dict[name].to(device_calc)
                    if fp_weight.shape != q_weight.shape: continue
                    error = fp_weight - q_weight
                    if isinstance(fp_module, nn.Conv2d):
                        dims_reduce = tuple(range(1, error.dim()))
                        mean_err_ch = torch.mean(error, dim=dims_reduce)
                        var_err_ch = torch.var(error, dim=dims_reduce, unbiased=False)
                        noise_mean_w[name], noise_variance_w[name] = mean_err_ch, var_err_ch
                    elif isinstance(fp_module, nn.Linear):
                        noise_mean_w[name] = torch.mean(error)
                        noise_variance_w[name] = torch.var(error, unbiased=False)
                except Exception as e:
                    print(f"警告：处理层 '{name}' 计算权重量化噪声时出错: {e}")
    return noise_mean_w, noise_variance_w

def calculate_activation_quantization_noise_stats( model_calib, calib_loader, a_bits, act_names, device, num_batches, task_type='classification'):
    """使用校准集计算激活量化误差的均值和方差"""
    model_calib.eval()
    raw_stats = {'mean': {}, 'variance': {}}
    q_min, q_max = 0, 2**a_bits - 1

    def get_stats(tensor):
        if tensor.numel() == 0: return None, None
        min_val, max_val = tensor.min(), tensor.max()
        if min_val >= max_val - 1e-9: return torch.tensor(0.0, device=device), torch.tensor(1e-9, device=device)
        
        scale = torch.clamp((max_val - min_val) / (q_max - q_min), min=1e-9)
        zero_point = torch.round(q_min - min_val / scale).clamp(q_min, q_max).to(torch.int32)
        
        q_tensor = torch.round(tensor / scale + zero_point).clamp(q_min, q_max)
        dq_tensor = (q_tensor - zero_point.float()) * scale
        error = tensor - dq_tensor
        return torch.mean(error), torch.clamp(torch.var(error, unbiased=False), min=1e-9)

    captured_outputs = {name: [] for name in act_names}
    hooks = []
    
    def capture_hook(module, input, output, name):
        if name in captured_outputs:
            captured_outputs[name].append(output.detach().cpu())
            
    for name, mod in model_calib.named_modules():
        if name in act_names:
            hooks.append(mod.register_forward_hook(functools.partial(capture_hook, name=name)))

    if hooks and calib_loader:
        with torch.no_grad():
            if task_type == 'classification':
                for i, (inputs, _) in enumerate(tqdm(calib_loader, total=min(num_batches, len(calib_loader)), desc="激活校准", leave=False, disable=True)):
                    if i >= num_batches: break
                    model_calib(inputs.to(device, non_blocking=True))
            elif task_type in ['detection', 'segmentation']: 
                for i, (images, *_) in enumerate(tqdm(calib_loader, total=min(num_batches, len(calib_loader)), desc="激活校准", leave=False, disable=True)):
                    if i >= num_batches: break
                    model_calib(images.to(device, non_blocking=True))
            
    for h in hooks: h.remove()
    
    for name, out_list in captured_outputs.items():
        if not out_list: continue
        all_outputs = torch.cat(out_list, dim=0).to(device)
        mean_stat, var_stat = get_stats(all_outputs)
        if mean_stat is not None: raw_stats['mean'][name] = mean_stat.cpu()
        if var_stat is not None: raw_stats['variance'][name] = var_stat.cpu()
        del all_outputs

    return raw_stats
