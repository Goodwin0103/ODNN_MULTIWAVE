# utils/save_utils.py
import torch
import os
import json
import numpy as np
from datetime import datetime

def save_model_checkpoint(model, epoch, loss, num_layers, config, optimizer=None, scheduler=None):
    """
    保存模型检查点
    
    Parameters:
    -----------
    model : torch.nn.Module
        要保存的模型
    epoch : int
        当前epoch
    loss : float
        当前损失
    num_layers : int
        层数
    config : ODNNConfig
        配置对象
    optimizer : torch.optim.Optimizer, optional
        优化器
    scheduler : torch.optim.lr_scheduler, optional
        学习率调度器
    """
    # 确保保存目录存在
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # 构建保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{num_layers}layers_epoch{epoch}_loss{loss:.6f}_{timestamp}.pth"
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    
    # 准备保存的数据
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'num_layers': num_layers,
        'config': {
            'FIELD_SIZE': config.FIELD_SIZE,
            'LAYER_SIZE': config.LAYER_SIZE,
            'NUM_MODES': config.NUM_MODES,
            'WAVELENGTHS': config.WAVELENGTHS.tolist() if hasattr(config.WAVELENGTHS, 'tolist') else config.WAVELENGTHS,
            'PIXEL_SIZE': config.PIXEL_SIZE,
            'Z_LAYERS': config.Z_LAYERS,
            'Z_PROP': config.Z_PROP,
        },
        'timestamp': timestamp
    }
    
    # 添加优化器状态
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # 添加调度器状态
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存检查点
    torch.save(checkpoint, filepath)
    
    # 同时保存最佳模型
    best_filepath = os.path.join(config.CHECKPOINT_DIR, f"best_model_{num_layers}layers.pth")
    torch.save(checkpoint, best_filepath)
    
    print(f"模型已保存到: {filepath}")
    return filepath

def load_model_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cuda:0'):
    """
    加载模型检查点
    
    Parameters:
    -----------
    filepath : str
        检查点文件路径
    model : torch.nn.Module
        模型对象
    optimizer : torch.optim.Optimizer, optional
        优化器
    scheduler : torch.optim.lr_scheduler, optional
        学习率调度器
    device : str
        设备
    
    Returns:
    --------
    dict : 检查点信息
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")
    
    # 加载检查点
    checkpoint = torch.load(filepath, map_location=device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"模型已从 {filepath} 加载")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.6f}")
    
    return checkpoint

def save_training_results(results, config, num_layers):
    """
    保存训练结果
    
    Parameters:
    -----------
    results : dict
        训练结果字典
    config : ODNNConfig
        配置对象
    num_layers : int
        层数
    """
    # 确保保存目录存在
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # 构建文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_results_{num_layers}layers_{timestamp}.json"
    filepath = os.path.join(config.SAVE_DIR, filename)
    
    # 转换numpy数组为列表（JSON序列化）
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            serializable_results[key] = value.detach().cpu().numpy().tolist()
        else:
            serializable_results[key] = value
    
    # 保存结果
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"训练结果已保存到: {filepath}")
    return filepath

def save_phase_masks(phase_masks, config, num_layers):
    """
    保存相位掩膜
    
    Parameters:
    -----------
    phase_masks : list
        相位掩膜列表
    config : ODNNConfig
        配置对象
    num_layers : int
        层数
    """
    # 确保保存目录存在
    save_dir = os.path.join(config.SAVE_DIR, "phase_masks")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存每个层的相位掩膜
    for i, phase_mask in enumerate(phase_masks):
        filename = f"phase_mask_layer{i+1}_{num_layers}layers.npy"
        filepath = os.path.join(save_dir, filename)
        np.save(filepath, phase_mask)
    
    print(f"相位掩膜已保存到: {save_dir}")

def save_predictions(predictions, config, num_layers):
    """
    保存预测结果
    
    Parameters:
    -----------
    predictions : np.ndarray
        预测结果
    config : ODNNConfig
        配置对象
    num_layers : int
        层数
    """
    # 确保保存目录存在
    save_dir = os.path.join(config.SAVE_DIR, "predictions")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存预测结果
    filename = f"predictions_{num_layers}layers.npy"
    filepath = os.path.join(save_dir, filename)
    np.save(filepath, predictions)
    
    print(f"预测结果已保存到: {filepath}")
    return filepath

def save_losses(losses, config, num_layers):
    """
    保存损失历史
    
    Parameters:
    -----------
    losses : list
        损失列表
    config : ODNNConfig
        配置对象
    num_layers : int
        层数
    """
    # 确保保存目录存在
    save_dir = os.path.join(config.SAVE_DIR, "losses")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存损失
    filename = f"losses_{num_layers}layers.npy"
    filepath = os.path.join(save_dir, filename)
    np.save(filepath, np.array(losses))
    
    # 同时保存为文本文件
    txt_filename = f"losses_{num_layers}layers.txt"
    txt_filepath = os.path.join(save_dir, txt_filename)
    with open(txt_filepath, 'w') as f:
        for i, loss in enumerate(losses):
            f.write(f"Epoch {i}: {loss:.10f}\n")
    
    print(f"损失历史已保存到: {filepath}")
    return filepath

def create_experiment_summary(config, results_dict):
    """
    创建实验总结
    
    Parameters:
    -----------
    config : ODNNConfig
        配置对象
    results_dict : dict
        结果字典
    """
    # 确保保存目录存在
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # 创建实验总结
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device': config.DEVICE,
            'num_modes': config.NUM_MODES,
            'wavelengths_nm': [wl * 1e9 for wl in config.WAVELENGTHS],
            'field_size': config.FIELD_SIZE,
            'layer_size': config.LAYER_SIZE,
            'num_layers_tested': config.NUM_LAYER_OPTIONS,
        },
        'training_params': {
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'gamma': config.GAMMA,
        },
        'optical_params': {
            'pixel_size_um': config.PIXEL_SIZE * 1e6,
            'z_layers_um': config.Z_LAYERS * 1e6,
            'z_prop_um': config.Z_PROP * 1e6,
            'focus_radius': config.FOCUS_RADIUS,
            'detect_size': config.DETECT_SIZE,
        },
        'results': results_dict
    }
    
    # 保存总结
    filename = f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(config.SAVE_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"实验总结已保存到: {filepath}")
    return filepath

def load_experiment_results(filepath):
    """
    加载实验结果
    
    Parameters:
    -----------
    filepath : str
        结果文件路径
    
    Returns:
    --------
    dict : 实验结果
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"结果文件不存在: {filepath}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results

def cleanup_old_checkpoints(config, keep_last_n=5):
    """
    清理旧的检查点文件
    
    Parameters:
    -----------
    config : ODNNConfig
        配置对象
    keep_last_n : int
        保留最近的N个检查点
    """
    if not os.path.exists(config.CHECKPOINT_DIR):
        return
    
    # 获取所有检查点文件
    checkpoint_files = []
    for filename in os.listdir(config.CHECKPOINT_DIR):
        if filename.startswith("model_") and filename.endswith(".pth"):
            filepath = os.path.join(config.CHECKPOINT_DIR, filename)
            mtime = os.path.getmtime(filepath)
            checkpoint_files.append((filepath, mtime))
    
    # 按修改时间排序
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # 删除旧文件
    for filepath, _ in checkpoint_files[keep_last_n:]:
        try:
            os.remove(filepath)
            print(f"已删除旧检查点: {os.path.basename(filepath)}")
        except OSError as e:
            print(f"删除文件失败 {filepath}: {e}")
