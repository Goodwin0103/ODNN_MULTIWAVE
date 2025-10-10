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

def default_fallback():
    return [40e-6, 60e-6, 80e-6]
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
    
    # *** 新增：基准波长配置 ***
    base_wavelength_idx: int = 1                      # 基准波长索引（默认选择中间波长）
    
    # 检测区域偏移 - 为每个波长定义不同的偏移
    offsets: List[Tuple[int, int]] = field(default_factory=default_offsets)  # 每个波长的检测区域偏移
    
    # *** 新增：Zero Padding 参数 ***
    use_zero_padding: bool = True                     # 是否启用Zero Padding
    padding_ratio: float = 0.5                       # 填充比例 (相对于原始尺寸)
    use_apodization: bool = True                      # 是否使用边界衰减
    apodization_width: int = 20                       # 边界衰减宽度 (像素)
    apodization_type: str = 'cosine'                  # 衰减类型: 'cosine', 'gaussian', 'linear'

    # *** MaskLoader 参数 ***
    fallback_focal_lengths: List[float] = field(default_factory=default_fallback)  # 备用掩码的焦距列表
    default_num_layers: int = 3                       # 默认层数

    # 训练参数
    learning_rate: float = 0.01                       # 学习率
    lr_decay: float = 0.99                            # 学习率衰减
    epochs: int = 400                                 # 训练轮数
    batch_size: int = 3                               # 批量大小
    
    # 保存参数
    save_dir: str = "./results_multi_mode_multi_wl/"  # 保存目录
    flag_savemat: bool = True                         # 是否保存.mat文件
    
    # 设备配置
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
        
        # *** 新增：验证基准波长索引 ***
        if self.base_wavelength_idx < 0 or self.base_wavelength_idx >= len(self.wavelengths):
            print(f"⚠ 基准波长索引 {self.base_wavelength_idx} 超出范围，自动调整为中间波长")
            self.base_wavelength_idx = len(self.wavelengths) // 2
        
        # 显示基准波长信息
        base_wl = self.wavelengths[self.base_wavelength_idx]
        print(f"✓ 基准波长设置: 索引 {self.base_wavelength_idx} -> {base_wl*1e9:.1f}nm")
        
        # *** 新增：Zero Padding 参数验证 ***
        if self.use_zero_padding:
            # 验证填充比例
            if not (0 < self.padding_ratio <= 2.0):
                print(f"⚠ 填充比例 {self.padding_ratio} 超出合理范围 (0, 2.0]，调整为 0.5")
                self.padding_ratio = 0.5
            
            # 验证衰减宽度
            if self.use_apodization and self.apodization_width <= 0:
                print(f"⚠ 衰减宽度 {self.apodization_width} 无效，调整为 20")
                self.apodization_width = 20
            
            # 验证衰减类型
            valid_apodization_types = ['cosine', 'gaussian', 'linear']
            if self.apodization_type not in valid_apodization_types:
                print(f"⚠ 衰减类型 '{self.apodization_type}' 无效，调整为 'cosine'")
                self.apodization_type = 'cosine'
            
            print(f"✓ Zero Padding 配置: 填充比例={self.padding_ratio}, 衰减={self.use_apodization}")
            if self.use_apodization:
                print(f"  - 衰减类型: {self.apodization_type}, 宽度: {self.apodization_width}像素")
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 打印设备信息
        print(f"配置完成，使用设备: {self.device}")
        
        # 验证设备可用性
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print("⚠ CUDA不可用，自动切换到CPU")
            self.device = torch.device('cpu')
    
    # *** 新增：Zero Padding 相关方法 ***
    def get_padded_size(self, original_size: Optional[int] = None) -> int:
        """
        获取填充后的尺寸
        
        Args:
            original_size: 原始尺寸，如果为None则使用layer_size
            
        Returns:
            填充后的尺寸
        """
        if not self.use_zero_padding:
            return original_size or self.layer_size
        
        size = original_size or self.layer_size
        padding_size = int(size * self.padding_ratio)
        return size + 2 * padding_size
    
    def get_padding_size(self, original_size: Optional[int] = None) -> int:
        """
        获取单边填充尺寸
        
        Args:
            original_size: 原始尺寸，如果为None则使用layer_size
            
        Returns:
            单边填充尺寸
        """
        if not self.use_zero_padding:
            return 0
        
        size = original_size or self.layer_size
        return int(size * self.padding_ratio)
    
    def print_config(self):
        """打印详细配置信息"""
        print("\n" + "="*60)
        print("📋 多模式多波长衍射神经网络配置")
        print("="*60)
        
        print("\n🌈 光学参数:")
        print(f"  - 模式数量: {self.num_modes}")
        print(f"  - 波长: {[f'{w*1e9:.0f}nm' for w in self.wavelengths]}")
        print(f"  - 基准波长: 索引{self.base_wavelength_idx} -> {self.wavelengths[self.base_wavelength_idx]*1e9:.1f}nm")
        print(f"  - 像素尺寸: {self.pixel_size*1e6:.1f} μm")
        
        print("\n📏 空间参数:")
        print(f"  - 场大小: {self.field_size} × {self.field_size}")
        print(f"  - 层大小: {self.layer_size} × {self.layer_size}")
        print(f"  - 焦点半径: {self.focus_radius} 像素")
        print(f"  - 检测区域: {self.detectsize} × {self.detectsize}")
        print(f"  - 检测偏移: {self.offsets}")
        
        print("\n🔲 Zero Padding 参数:")
        print(f"  - 启用 Zero Padding: {self.use_zero_padding}")
        if self.use_zero_padding:
            print(f"  - 填充比例: {self.padding_ratio}")
            print(f"  - 填充后尺寸: {self.get_padded_size()} × {self.get_padded_size()}")
            print(f"  - 启用边界衰减: {self.use_apodization}")
            if self.use_apodization:
                print(f"  - 衰减类型: {self.apodization_type}")
                print(f"  - 衰减宽度: {self.apodization_width} 像素")
        
        print("\n📐 物理参数:")
        print(f"  - 层间距离: {self.z_layers*1e6:.1f} μm")
        print(f"  - 传播距离: {self.z_prop*1e6:.1f} μm")
        print(f"  - 传播步长: {self.z_step*1e6:.1f} μm")
        
        print("\n🎯 训练参数:")
        print(f"  - 学习率: {self.learning_rate}")
        print(f"  - 学习率衰减: {self.lr_decay}")
        print(f"  - 训练轮数: {self.epochs}")
        print(f"  - 批量大小: {self.batch_size}")
        
        print("\n💾 输出参数:")
        print(f"  - 保存目录: {self.save_dir}")
        print(f"  - 保存MAT文件: {self.flag_savemat}")
        print(f"  - 计算设备: {self.device}")
        
        print("="*60)
    
    # *** 新增：获取基准波长 ***
    @property
    def base_wavelength(self) -> float:
        """获取基准波长"""
        return self.wavelengths[self.base_wavelength_idx]
    
    # *** 新增：获取波长相对比例 ***
    def get_wavelength_ratios(self) -> np.ndarray:
        """
        获取相对于基准波长的比例
        
        Returns:
            波长比例数组
        """
        return self.wavelengths / self.base_wavelength