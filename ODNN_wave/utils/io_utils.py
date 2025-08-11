import torch
import scipy.io as sio
import numpy as np
import os
import time
from datetime import datetime

def save_to_mat(data, filename, key='data'):
    """
    保存数据到.mat文件
    
    参数:
        data: 要保存的数据
        filename: 文件名
        key: 数据键名
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # 转换数据类型
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # 保存数据
    sio.savemat(filename, {key: data})
    print(f"数据已保存至: {filename}")

def save_to_mat_MC(data_list, filenames, keys=None):
    """
    保存多通道数据到多个.mat文件
    
    参数:
        data_list: 数据列表
        filenames: 文件名列表
        keys: 数据键名列表
    """
    if keys is None:
        keys = ['data'] * len(data_list)
    
    for data, filename, key in zip(data_list, filenames, keys):
        save_to_mat(data, filename, key)

def save_model(model, config, optimizer=None, epoch=None, loss=None):
    """
    保存模型和训练状态
    
    参数:
        model: 模型
        config: 配置
        optimizer: 优化器
        epoch: 当前轮数
        loss: 当前损失
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.save_dir, f"model_{timestamp}")
    if epoch is not None:
        path += f"_epoch{epoch}"
    path += ".pth"
    
    state = {
        'model_state_dict': model.state_dict(),
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    }
    
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        state['epoch'] = epoch
    if loss is not None:
        state['loss'] = loss
    
    torch.save(state, path)
    print(f"模型已保存至: {path}")
    
    return path

def load_model(model, path, optimizer=None, device=None):
    """
    加载模型
    
    参数:
        model: 模型
        path: 模型路径
        optimizer: 优化器
        device: 设备
        
    返回:
        加载的状态
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    
    print(f"模型已从 {path} 加载")
    if 'epoch' in state:
        print(f"训练轮数: {state['epoch']}")
    if 'loss' in state:
        print(f"损失: {state['loss']}")
    
    return state
