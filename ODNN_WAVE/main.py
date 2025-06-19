# -*- coding: utf-8 -*-
"""
多模式多波长光场调制系统 - 主程序
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# 导入自定义模块
from config import Config
from data_generator import MultiModeMultiWavelengthDataGenerator
from model import MultiModeMultiWavelengthModel
from trainer import Trainer
from simulator import Simulator
from visualizer import Visualizer

def main():
    # 设置随机种子，确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建配置
    config = Config(
        # 基本参数
        num_modes=3,                                # 模式数量
        wavelengths=np.array([450e-9, 550e-9, 650e-9]),  # 波长列表(m)
        
        # 空间参数
        field_size=50,                              # 场大小(像素)
        layer_size=200,                             # 层大小(像素)
        focus_radius=5,                             # 焦点半径(像素)
        detectsize=15,                              # 检测区域大小(像素)
        
        # 物理参数
        z_layers=40e-6,                             # 层间距离(m)
        z_prop=300e-6,                              # 传播距离(m)
        z_step=20e-6,                               # 传播步长(m)
        pixel_size=1e-6,                            # 像素大小(m)
        
        # 检测区域偏移 - 为每个波长定义不同的偏移
        offsets=[(0,0), (20,0), (-20,0)],           # 每个波长的检测区域偏移
        
        # 训练参数
        learning_rate=0.01,                         # 学习率
        lr_decay=0.99,                              # 学习率衰减
        epochs=400,                                # 训练轮数
        batch_size=3,                               # 批量大小
        
        # 保存参数
        save_dir="./results_multi_mode_multi_wl/",  # 保存目录
        flag_savemat=True                           # 是否保存.mat文件
    )
    
    # 创建数据生成器
    print("创建数据生成器...")
    data_generator = MultiModeMultiWavelengthDataGenerator(config)
    
    # 创建可视化器
    print("创建可视化器...")
    visualizer = Visualizer(config)
    
    # 可视化检测区域
    print("可视化检测区域...")
    visualizer.visualize_detector_regions(save_path=f"{config.save_dir}/detector_regions.png")
    
    # 创建训练器
    print("创建训练器...")
    trainer = Trainer(config, data_generator)
    
    # 定义要训练的层数选项
    num_layer_options = [1, 2, 3]
    
    # 训练多个模型
    print(f"开始训练 {len(num_layer_options)} 个不同层数的模型...")
    results = trainer.train_multiple_models(MultiModeMultiWavelengthModel, num_layer_options)
    
    # 可视化训练损失
    print("可视化训练损失...")
    plt.figure(figsize=(10, 6))
    for i, num_layers in enumerate(num_layer_options):
        plt.plot(results['losses'][i], label=f'{num_layers} 层')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('不同层数模型的训练损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{config.save_dir}/training_losses.png", dpi=300)
    plt.close()
    
    # 可视化能量分布
    print("可视化能量分布...")
    for i, num_layers in enumerate(num_layer_options):
        fig, visibility = visualizer.plot_energy_distribution(
            [results['weights_pred'][i]], [num_layers]
        )
        plt.savefig(f"{config.save_dir}/energy_distribution_{num_layers}_layers.png", dpi=300)
        plt.close(fig)
    
    # 可视化可见度比较
    print("可视化可见度比较...")
    # 重新组织可见度数据，按模式组织
    visibility_by_mode = []
    for m in range(config.num_modes):
        mode_vis = []
        for i in range(len(num_layer_options)):
            mode_vis.append(results['visibility'][i][m])
        visibility_by_mode.append(mode_vis)
    
    visualizer.plot_visibility_comparison_by_mode(visibility_by_mode, num_layer_options)
    plt.savefig(f"{config.save_dir}/visibility_comparison.png", dpi=300)
    plt.close()
    
    # 选择最佳模型进行光场传播模拟
    print("选择最佳模型进行光场传播模拟...")
    best_model_idx = np.argmax([np.mean(vis) for vis in results['visibility']])
    best_num_layers = num_layer_options[best_model_idx]
    best_model = results['models'][best_model_idx]
    best_phase_masks = results['phase_masks'][best_model_idx]
    
    print(f"最佳模型: {best_num_layers} 层, 平均可见度: {np.mean(results['visibility'][best_model_idx]):.4f}")
    
    # 创建模拟器
    print("创建模拟器...")
    simulator = Simulator(config)
    
    # 生成输入场
    print("生成输入场...")
    input_field = data_generator.generate_input_data()
    
    # 为每个模式生成专用相位掩膜
    print("为每个模式生成专用相位掩膜...")
    mode_specific_masks = simulator.generate_mode_specific_masks(best_phase_masks, config.num_modes)
    
    # 模拟光场传播
    print("模拟光场传播...")
    simulator.simulate_propagation(
        best_phase_masks, 
        input_field, 
        process_all_modes=True,
        mode_specific_masks=mode_specific_masks
    )
    
    # 打印相位掩膜信息
    print("打印相位掩膜信息...")
    best_model.print_phase_masks(save_path=config.save_dir)
    
    # 保存结果
    print("保存结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    torch.save({
        'config': config,
        'models': [model.state_dict() for model in results['models']],
        'losses': results['losses'],
        'visibility': results['visibility'],
        'best_model_idx': best_model_idx,
        'num_layer_options': num_layer_options
    }, f"{config.save_dir}/results_{timestamp}.pth")
    
    print("程序执行完成!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总执行时间: {end_time - start_time:.2f} 秒")
