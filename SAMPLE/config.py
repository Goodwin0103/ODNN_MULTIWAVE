import numpy as np
import os
import torch

class Config:
    def __init__(self, **kwargs):
        # 基本参数
        self.num_modes = kwargs.get('num_modes', 1)  # 默认只有1个模式
        self.wavelengths = kwargs.get('wavelengths', np.array([450e-9, 650e-9]))  # 默认两个波长
        
        # 空间参数
        self.field_size = kwargs.get('field_size', 50)  # 场大小(像素)
        self.layer_size = kwargs.get('layer_size', 200)  # 层大小(像素)
        self.focus_radius = kwargs.get('focus_radius', 5)  # 焦点半径(像素)
        self.detect_size = kwargs.get('detect_size', 15)  # 检测区域大小(像素)
        
        # 物理参数
        self.z_layers = kwargs.get('z_layers', 40e-6)  # 层间距离(m)
        self.z_prop = kwargs.get('z_prop', 300e-6)  # 传播距离(m)
        self.z_step = kwargs.get('z_step', 20e-6)  # 传播步长(m)
        self.pixel_size = kwargs.get('pixel_size', 1e-6)  # 像素大小(m)
        
        # 检测区域偏移 - 为每个波长定义不同的偏移
        self.offsets = kwargs.get('offsets', [(0,0), (20,0)])  # 每个波长的检测区域偏移
        
        # 训练参数
        self.learning_rate = kwargs.get('learning_rate', 0.01)  # 学习率
        self.lr_decay = kwargs.get('lr_decay', 0.99)  # 学习率衰减
        self.epochs = kwargs.get('epochs', 200)  # 训练轮数
        self.num_epochs = kwargs.get('num_epochs', 200)  # 训练轮数(别名)
        self.batch_size = kwargs.get('batch_size', 2)  # 批量大小
        
        # 设备配置
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 保存参数
        self.save_dir = kwargs.get('save_dir', "./results_simple/")  # 保存目录
        self.flag_savemat = kwargs.get('flag_savemat', False)  # 是否保存.mat文件
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
    def print_config(self):
        """打印当前配置信息"""
        print("\n===== 系统配置 =====")
        print(f"模式数量: {self.num_modes}")
        print(f"波长: {[f'{wl*1e9:.0f}nm' for wl in self.wavelengths]}")
        print(f"场大小: {self.field_size}×{self.field_size} 像素")
        print(f"层大小: {self.layer_size}×{self.layer_size} 像素")
        print(f"像素尺寸: {self.pixel_size*1e6:.1f} μm")
        print(f"训练轮数: {self.num_epochs}")
        print(f"保存目录: {self.save_dir}")
        print(f"计算设备: {self.device}")
        print("===================\n")
