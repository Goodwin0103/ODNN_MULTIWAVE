import torch
import torch.nn as nn
import numpy as np
from propagation import angular_spectrum_propagation

class SimpleMultiWavelengthModel(nn.Module):
    def __init__(self, config, num_layers=1):
        super(SimpleMultiWavelengthModel, self).__init__()
        self.config = config
        self.num_layers = num_layers
        self.field_size = config.field_size
        self.layer_size = config.layer_size
        self.pixel_size = config.pixel_size
        self.wavelengths = config.wavelengths
        self.device = config.device
        
        # 初始化相位掩膜参数 - 使用更好的初始化方法
        self.phase_masks = nn.ParameterList([
            nn.Parameter(torch.rand(config.layer_size, config.layer_size) * 2 * np.pi - np.pi)
            for _ in range(num_layers)
        ])
        
        # 设置传播距离 - 确保距离合理
        self.propagation_distances = [50e-6] * num_layers  # 例如，50微米
        
    def forward(self, input_fields):
        """
        处理多个波长的输入场
        
        Args:
            input_fields: 列表，包含每个波长的输入场
            
        Returns:
            output_fields: 列表，包含每个波长的输出场
        """
        output_fields = []
        
        # 对每个波长分别处理
        for w_idx, input_field in enumerate(input_fields):
            wavelength = self.wavelengths[w_idx]
            output_field = self._process_single_wavelength(input_field, wavelength)
            output_fields.append(output_field)
            
        return output_fields
    
    def _process_single_wavelength(self, input_field, wavelength):
        """处理单个波长的场"""
        current_field = input_field
        
        # 通过每个相位掩膜和传播
        for i in range(self.num_layers):
            # 应用相位掩膜
            current_field = current_field * torch.exp(1j * self.phase_masks[i])
            
            # 传播到下一层
            if i < self.num_layers - 1:
                current_field = angular_spectrum_propagation(
                    current_field, 
                    self.propagation_distances[i], 
                    wavelength,
                    self.config.pixel_size
                )
        
        # 最后一次传播到检测平面
        output_field = angular_spectrum_propagation(
            current_field,
            self.propagation_distances[-1],
            wavelength,
            self.config.pixel_size
        )
        
        return output_field
    
    # 修改损失函数计算
    def compute_loss(self, output_fields):
        """计算损失函数 - 鼓励波长分离"""
        total_loss = 0
        num_wavelengths = len(output_fields)
        
        # 对每个波长计算损失
        for w_idx, field in enumerate(output_fields):
            # 计算强度
            intensity = torch.abs(field)**2
            
            # 获取当前波长的目标区域
            offset_x, offset_y = self.config.offsets[w_idx]
            center_x = self.field_size // 2 + offset_x
            center_y = self.field_size // 2 + offset_y
            
            half_size = self.config.detect_size // 2
            x_start = center_x - half_size
            x_end = center_x + half_size
            y_start = center_y - half_size
            y_end = center_y + half_size
            
            # 计算目标区域内的能量
            target_energy = torch.sum(intensity[y_start:y_end, x_start:x_end])
            
            # 计算总能量
            total_energy = torch.sum(intensity)
            
            # 计算当前波长的损失 - 目标是最大化目标区域内的能量比例
            wavelength_loss = -torch.log(target_energy / (total_energy + 1e-10))
            
            # 添加其他波长目标区域的惩罚项
            for other_w_idx in range(num_wavelengths):
                if other_w_idx != w_idx:
                    other_offset_x, other_offset_y = self.config.offsets[other_w_idx]
                    other_center_x = self.field_size // 2 + other_offset_x
                    other_center_y = self.field_size // 2 + other_offset_y
                    
                    other_x_start = other_center_x - half_size
                    other_x_end = other_center_x + half_size
                    other_y_start = other_center_y - half_size
                    other_y_end = other_center_y + half_size
                    
                    # 计算其他波长目标区域的能量
                    other_target_energy = torch.sum(intensity[other_y_start:other_y_end, other_x_start:other_x_end])
                    
                    # 添加惩罚项 - 希望在其他波长的目标区域能量最小
                    wavelength_loss += torch.log(other_target_energy / (total_energy + 1e-10) + 1e-10)
            
            # 累加到总损失
            total_loss += wavelength_loss
        
        return total_loss

    def get_all_fields(self, input_fields):
        """
        计算并返回每一层的场分布
        
        Args:
            input_fields: 输入场，形状为 [num_wavelengths, field_height, field_width]
        
        Returns:
            all_fields: 每一层的场分布，形状为 [num_wavelengths, num_layers, field_height, field_width]
        """
        num_wavelengths = len(input_fields)
        all_fields = []
        
        # 对每个波长分别计算
        for w_idx in range(num_wavelengths):
            wavelength = self.wavelengths[w_idx]
            field = input_fields[w_idx]
            
            # 存储每一层的场分布
            fields_per_wavelength = []
            
            # 通过每一层传播
            for l_idx in range(len(self.phase_masks)):
                # 应用相位掩膜
                phase_mask = self.phase_masks[l_idx]
                field = field * torch.exp(1j * phase_mask)
                
                # 传播到下一层
                if l_idx < len(self.phase_masks) - 1:
                    field = angular_spectrum_propagation(
                        field, 
                        self.propagation_distances[l_idx], 
                        wavelength,
                        self.config.pixel_size
                    )
                
                # 保存当前层的场分布
                fields_per_wavelength.append(field)
            
            all_fields.append(fields_per_wavelength)
        
        return all_fields
