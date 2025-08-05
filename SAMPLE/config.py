import torch
import numpy as np
import os

class Config:
    def __init__(self, field_size=50, layer_size=None, wavelengths=None, offsets=None, 
                 detectsize=10, num_layers=3, epochs=200, batch_size=1, 
                 learning_rate=0.01, save_dir='results', num_epochs=None):
        # 基本参数
        self.field_size = field_size
        self.layer_size = layer_size if layer_size is not None else field_size
        self.pixel_size = 1e-6  # 1 μm per pixel
        
        # 兼容性处理：num_epochs 和 epochs
        if num_epochs is not None:
            self.epochs = num_epochs
            self.num_epochs = num_epochs
        else:
            self.epochs = epochs
            self.num_epochs = epochs
        
        # 波长设置
        if wavelengths is None:
            self.wavelengths = np.array([450e-9, 650e-9])  # 默认双波长
        else:
            self.wavelengths = np.array(wavelengths)
        
        # 检测区域偏移 [(x1,y1), (x2,y2), ...]
        if offsets is None:
            if len(self.wavelengths) == 2:
                self.offsets = [(-10, 0), (10, 0)]  # 默认双波长左右分离
            else:
                # 为多波长生成默认偏移
                self.offsets = []
                for i, wl in enumerate(self.wavelengths):
                    angle = 2 * np.pi * i / len(self.wavelengths)
                    radius = 15
                    x_offset = int(radius * np.cos(angle))
                    y_offset = int(radius * np.sin(angle))
                    self.offsets.append((x_offset, y_offset))
        else:
            self.offsets = offsets
        
        # 检测区域大小
        self.detectsize = detectsize
        self.detect_size = detectsize  # 兼容性别名
        
        # 训练参数
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 保存路径
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        # 检查波长数量和偏移数量是否匹配
        if len(self.wavelengths) != len(self.offsets):
            raise ValueError(f"Number of wavelengths ({len(self.wavelengths)}) "
                           f"must match number of offsets ({len(self.offsets)})")
        
        # 检查检测区域是否在场范围内
        for i, (offset_x, offset_y) in enumerate(self.offsets):
            center_x = self.field_size // 2 + offset_x
            center_y = self.field_size // 2 + offset_y
            
            if (center_x - self.detectsize // 2 < 0 or 
                center_x + self.detectsize // 2 >= self.field_size or
                center_y - self.detectsize // 2 < 0 or 
                center_y + self.detectsize // 2 >= self.field_size):
                print(f"Warning: Detection region for wavelength {i} "
                      f"({int(self.wavelengths[i]*1e9)}nm) may be outside field boundaries")
        
        print("Configuration validation completed.")
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 50)
        print("CONFIGURATION SUMMARY")
        print("=" * 50)
        print(f"Field size: {self.field_size} x {self.field_size} pixels")
        print(f"Layer size: {self.layer_size} x {self.layer_size} pixels")
        print(f"Pixel size: {self.pixel_size*1e6:.1f} μm")
        print(f"Wavelengths: {[int(wl*1e9) for wl in self.wavelengths]} nm")
        print(f"Detection offsets: {self.offsets}")
        print(f"Detection size: {self.detectsize} x {self.detectsize} pixels")
        print(f"Number of layers: {self.num_layers}")
        print(f"Training epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")
        print("=" * 50)
