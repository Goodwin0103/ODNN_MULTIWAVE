import numpy as np
import torch

class SimpleDataGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config.device  # 使用配置中的设备
        self.wavelengths = config.wavelengths  # 从配置中获取波长
        self.field_size = config.field_size  # 从配置中获取场大小
        
    def generate_labels(self):
        """生成简单的标签 - 每个波长对应一个目标区域"""
        num_wavelengths = len(self.wavelengths)
        
        # 创建标签 - 形状为 [波长数, 区域数]
        # 对于每个波长，只有一个区域的值为1，其他为0
        labels = np.zeros((num_wavelengths, num_wavelengths))
        for i in range(num_wavelengths):
            labels[i, i] = 1.0
            
        print(f"生成标签形状: {labels.shape}")
        return labels

    def generate_input_fields(self):
        """生成不同波长的输入场"""
        input_fields = {}
        
        for wavelength in self.wavelengths:
            # 创建平面波场
            field = torch.ones((self.field_size, self.field_size), 
                               dtype=torch.complex64, 
                               device=self.device)
            
            # 可以添加一些随机相位扰动
            random_phase = torch.rand((self.field_size, self.field_size), 
                                      device=self.device) * 0.1 * np.pi
            field = field * torch.exp(1j * random_phase)
            
            input_fields[wavelength] = field
        
        return input_fields

    def generate_input_data(self):
        """生成输入场 - 简单的高斯光束"""
        field_size = self.field_size
        
        # 创建网格
        x = np.linspace(-1, 1, field_size)
        y = np.linspace(-1, 1, field_size)
        xx, yy = np.meshgrid(x, y)
        r2 = xx**2 + yy**2
        
        # 创建高斯光束
        sigma = 0.5
        amplitude = np.exp(-r2/(2*sigma**2))
        
        # 归一化
        amplitude = amplitude / np.max(amplitude)
        
        # 创建复场 (振幅+零相位)
        field = amplitude * np.exp(1j * 0)
        
        # 转为PyTorch张量
        field_tensor = torch.tensor(field, dtype=torch.complex64, device=self.device)
        
        # 对于多波长情况，创建一个列表包含相同的场
        fields = [field_tensor for _ in range(len(self.wavelengths))]
        
        return fields
    
    def get_detector_regions(self):
        """获取检测区域坐标"""
        field_size = self.field_size
        
        # 检查配置中是否有 detectsize 和 offsets
        if hasattr(self.config, 'detectsize') and hasattr(self.config, 'offsets'):
            detect_size = self.config.detectsize
            num_wavelengths = len(self.wavelengths)
            
            regions = []
            
            # 为每个波长创建一个检测区域
            for i, offset in enumerate(self.config.offsets):
                # 计算中心点
                center_x = field_size // 2 + offset[0]
                center_y = field_size // 2 + offset[1]
                
                # 计算检测区域边界
                half_size = detect_size // 2
                x_start = max(center_x - half_size, 0)
                x_end = min(center_x + half_size, field_size)
                y_start = max(center_y - half_size, 0)
                y_end = min(center_y + half_size, field_size)
                
                regions.append((x_start, x_end, y_start, y_end))
        else:
            # 如果配置中没有相关参数，使用默认值
            num_wavelengths = len(self.wavelengths)
            regions = []
            
            # 默认在场的不同区域创建检测区域
            for i in range(num_wavelengths):
                size = field_size // 4
                offset = field_size // (num_wavelengths + 1) * (i + 1) - field_size // 2
                
                # 计算中心点
                center_x = field_size // 2 + offset
                center_y = field_size // 2
                
                # 计算检测区域边界
                half_size = size // 2
                x_start = max(center_x - half_size, 0)
                x_end = min(center_x + half_size, field_size)
                y_start = max(center_y - half_size, 0)
                y_end = min(center_y + half_size, field_size)
                
                regions.append((x_start, x_end, y_start, y_end))
                
        return regions
    
    def get_detector_region(self, wavelength):
        """获取特定波长的检测区域"""
        detector_region = torch.zeros((self.field_size, self.field_size), device=self.device)
        
        # 获取所有检测区域
        regions = self.get_detector_regions()
        
        # 找到当前波长对应的索引
        wavelength_index = self.wavelengths.index(wavelength)
        
        # 设置检测区域
        if wavelength_index < len(regions):
            x_start, x_end, y_start, y_end = regions[wavelength_index]
            detector_region[y_start:y_end, x_start:x_end] = 1.0
        
        return detector_region
