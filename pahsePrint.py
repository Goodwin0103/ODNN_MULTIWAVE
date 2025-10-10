#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
相位掩膜检查脚本
此脚本用于加载深度衍射神经网络模型并打印每一层的相位掩膜信息
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径，以便导入模块
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入您的模型和配置
from ODNN_WAVE.model import WavelengthDependentDiffractionLayer, WavelengthDependentD2NNModel
from ODNN_WAVE.config import Config

def print_phase_masks(model_path=None, config=None, num_layers=3, visualize=True):
    """
    打印模型中每一层的相位掩膜信息
    
    参数:
        model_path: 模型权重文件路径，如果为None则使用随机初始化的模型
        config: 模型配置对象
        num_layers: 模型层数
        visualize: 是否可视化相位掩膜
    """
    # 如果没有提供配置，使用默认配置
    if config is None:
        config = Config()
    
    # 创建模型
    model = WavelengthDependentD2NNModel(config, num_layers=num_layers)
    
    # 如果提供了模型路径，加载模型权重
    if model_path is not None and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print("使用随机初始化的模型")
    
    # 打印模型信息
    print("\n模型信息:")
    print(f"  层数: {len(model.layers)}")
    print(f"  波长数量: {len(model.layers[0].lam_list) if hasattr(model.layers[0], 'lam_list') else 1}")
    print(f"  波长列表: {model.layers[0].lam_list.cpu().numpy() * 1e9} nm" if hasattr(model.layers[0], 'lam_list') else "未知")
    
    # 打印每一层的相位掩膜信息
    print("\n相位掩膜信息:")
    total_params = 0
    
    for i, layer in enumerate(model.layers):
        print(f"\n层 {i+1}:")
        
        # 检查是否有相位掩膜
        if hasattr(layer, 'phase'):
            phase = layer.phase.detach().cpu().numpy()
            phase_shape = phase.shape
            phase_params = np.prod(phase_shape)
            total_params += phase_params
            
            print(f"  相位掩膜形状: {phase_shape}")
            print(f"  参数数量: {phase_params}")
            print(f"  相位值范围: [{phase.min():.4f}, {phase.max():.4f}]")
            print(f"  相位值平均: {phase.mean():.4f}")
            print(f"  相位值标准差: {phase.std():.4f}")
            
            # 检查是否有多个相位掩膜
            if hasattr(layer, '_phase_per_wavelength'):
                print("  警告: 发现每个波长有独立的相位掩膜!")
                for j, wavelength in enumerate(layer.lam_list.cpu().numpy()):
                    phase_wl = layer._phase_per_wavelength[j].detach().cpu().numpy()
                    print(f"    波长 {wavelength*1e9:.0f} nm 相位掩膜形状: {phase_wl.shape}")
        else:
            print("  没有找到相位掩膜参数!")
    
    print(f"\n总相位掩膜参数数量: {total_params}")
    
    # 可视化相位掩膜
    if visualize:
        visualize_phase_masks(model)

def visualize_phase_masks(model, save_path=None):
    """可视化模型中的相位掩膜"""
    num_layers = len(model.layers)
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 5, 5))
    
    if num_layers == 1:
        axes = [axes]
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'phase'):
            phase = layer.phase.detach().cpu().numpy() % (2 * np.pi)
            im = axes[i].imshow(phase, cmap='hsv', vmin=0, vmax=2*np.pi)
            axes[i].set_title(f'Layer {i+1} Phase Mask')
            axes[i].axis('off')
    
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"相位掩膜已保存到: {save_path}")
    else:
        plt.savefig('phase_masks.png')
        print("相位掩膜已保存到: phase_masks.png")
    
    plt.show()

def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='打印模型相位掩膜信息')
    parser.add_argument('--model', type=str, default=None, help='模型权重文件路径')
    parser.add_argument('--layers', type=int, default=3, help='模型层数')
    parser.add_argument('--no-vis', action='store_true', help='不可视化相位掩膜')
    args = parser.parse_args()
    
    # 打印相位掩膜信息
    print_phase_masks(
        model_path=args.model,
        num_layers=args.layers,
        visualize=not args.no_vis
    )

if __name__ == "__main__":
    main()
