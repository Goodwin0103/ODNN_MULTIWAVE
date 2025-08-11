import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Config:
    """配置类"""
    
    # 基本参数
    num_modes: int = 3                                     # 模式数量
    wavelengths: np.ndarray = field(default_factory=lambda: np.array([450e-9, 550e-9, 650e-9]))  # 波长列表(m)
    num_layers: int = 3                                    # 衍射层数量
    
    # 空间参数
    field_size: int = 50                                   # 场大小(像素)
    layer_size: int = 200                                  # 层大小(像素)
    focus_radius: int = 5                                  # 焦点半径(像素)
    detection_size: int = 15                               # 检测区域大小(像素)
    
    # 物理参数
    z_layers: float = 40e-6                                # 层间距离(m)
    z_prop: float = 300e-6                                 # 传播距离(m)
    pixel_size: float = 1e-6                               # 像素大小(m)
    
    # 检测区域偏移
    offsets: List[Tuple[int, int]] = field(default_factory=lambda: [(0,0), (20,0), (-20,0)])
    
    # 训练参数
    learning_rate: float = 0.01                            # 学习率
    lr_decay: float = 0.99                                 # 学习率衰减
    epochs: int = 400                                      # 训练轮数
    batch_size: int = 3                                    # 批量大小
    
    # 保存参数
    save_dir: str = "./results/"                           # 保存目录
    save_interval: int = 50                                # 保存间隔
    
    # 设备参数
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        """初始化后检查配置"""
        import os
        os.makedirs(self.save_dir, exist_ok=True)
