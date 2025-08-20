# config.py
import numpy as np
import os
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

def default_wavelengths():
    """默认波长列表工厂函数"""
    return np.array([450e-9, 550e-9, 650e-9])

def default_offsets():
    """默认偏移列表工厂函数"""
    return [(0,0), (20,0), (-20,0)]

def default_device():
    """默认设备工厂函数"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class Config:
    # 基本参数
    num_modes: int = 3                                # 模式数量
    wavelengths: np.ndarray = field(default_factory=default_wavelengths)  # 波长列表(m)
    
    # 空间参数
    field_size: int = 50                              # 场大小(像素)
    layer_size: int = 200                             # 层大小(像素)
    focus_radius: int = 5                             # 焦点半径(像素)
    detectsize: int = 15                              # 检测区域大小(像素)
    
    # 物理参数
    z_layers: float = 40e-6                           # 层间距离(m)
    z_prop: float = 300e-6                            # 传播距离(m)
    z_step: float = 20e-6                             # 传播步长(m)
    pixel_size: float = 1e-6                          # 像素大小(m)
    
    # 检测区域偏移 - 为每个波长定义不同的偏移
    offsets: List[Tuple[int, int]] = field(default_factory=default_offsets)  # 每个波长的检测区域偏移
    
    # 训练参数
    learning_rate: float = 0.01                       # 学习率
    lr_decay: float = 0.99                            # 学习率衰减
    epochs: int = 400                                # 训练轮数
    batch_size: int = 3                               # 批量大小
    
    # 保存参数
    save_dir: str = "./results_multi_mode_multi_wl/"  # 保存目录
    flag_savemat: bool = True                         # 是否保存.mat文件
    
    # *** 新增：设备配置 ***
    device: torch.device = field(default_factory=default_device)  # 计算设备
    
    def __post_init__(self):
        # 确保offsets数量与波长数量一致
        if len(self.offsets) != len(self.wavelengths):
            # 如果不一致，则调整offsets列表
            if len(self.offsets) < len(self.wavelengths):
                # 如果offsets少于波长数，则添加默认偏移(0,0)
                for _ in range(len(self.wavelengths) - len(self.offsets)):
                    self.offsets.append((0, 0))
            else:
                # 如果offsets多于波长数，则截断
                self.offsets = self.offsets[:len(self.wavelengths)]
            
            print(f"已调整offsets数量以匹配波长数量: {len(self.wavelengths)}")
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # *** 新增：打印设备信息 ***
        print(f"配置完成，使用设备: {self.device}")
        
        # *** 新增：验证设备可用性 ***
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print("⚠ CUDA不可用，自动切换到CPU")
            self.device = torch.device('cpu')
