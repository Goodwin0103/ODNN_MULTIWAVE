import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import math

class MultiModeMultiWavelengthModel(nn.Module):
    """
    多模式多波长衍射神经网络模型
    支持多个模式和多个波长的同时处理
    """
    
    def __init__(self, config, data_generator, evaluation_regions=None):
        super(MultiModeMultiWavelengthModel, self).__init__()
        self.config = config
        self.data_generator = data_generator
        self.evaluation_regions = evaluation_regions or []
        
        # 基本参数
        self.num_modes = getattr(config, 'num_modes', 3)
        self.wavelengths = config.wavelengths
        self.num_wavelengths = len(self.wavelengths)
        self.layer_size = config.layer_size
        self.pixel_size = config.pixel_size
        
        print(f"🚀 初始化多模式多波长模型:")
        print(f"   模式数: {self.num_modes}")
        print(f"   波长数: {self.num_wavelengths}")
        print(f"   波长: {[wl*1e9 for wl in self.wavelengths]} nm")
        print(f"   层大小: {self.layer_size}×{self.layer_size}")
        print(f"   评估区域数: {len(self.evaluation_regions)}")
        
        # 创建衍射层
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer = DiffractionLayer(
                layer_size=self.layer_size,
                wavelengths=self.wavelengths,
                pixel_size=self.pixel_size,
                z_distance=config.z_layers,
                layer_index=i
            )
            self.layers.append(layer)
        
        # 传播距离参数
        self.z_prop = config.z_prop
        self.z_step = config.z_step
        
        print(f"✅ 模型创建完成，共 {len(self.layers)} 层")
    
    def forward(self, x):
        """
        前向传播
        输入: x [batch_size, num_modes, num_wavelengths, height, width]
        输出: [batch_size, num_modes, num_wavelengths, height, width]
        """
        batch_size, num_modes, num_wavelengths, height, width = x.shape
        
        # 处理每个模式和波长的组合
        outputs = []
        
        for mode_idx in range(num_modes):
            mode_outputs = []
            
            for wl_idx in range(num_wavelengths):
                # 获取当前模式和波长的输入
                current_field = x[:, mode_idx, wl_idx]  # [batch_size, height, width]
                
                # 通过所有衍射层
                for layer in self.layers:
                    current_field = layer(current_field, wl_idx)
                
                # 最终传播到检测平面
                current_field = self._propagate_to_detector(current_field, wl_idx)
                
                mode_outputs.append(current_field)
            
            outputs.append(torch.stack(mode_outputs, dim=1))  # [batch_size, num_wavelengths, height, width]
        
        # 重新组织输出格式
        output = torch.stack(outputs, dim=1)  # [batch_size, num_modes, num_wavelengths, height, width]
        
        return output
    
    def _propagate_to_detector(self, field, wavelength_idx):
        """传播到检测器平面"""
        wavelength = self.wavelengths[wavelength_idx]
        
        # 使用角谱传播
        field_ft = torch.fft.fft2(field)
        
        # 创建传播核
        k = 2 * np.pi / wavelength
        kx, ky = self._get_k_vectors()
        kz = torch.sqrt(k**2 - kx**2 - ky**2 + 0j)
        
        # 传播相位
        propagation_phase = torch.exp(1j * kz * self.z_prop)
        
        # 应用传播
        field_ft_prop = field_ft * propagation_phase
        field_prop = torch.fft.ifft2(field_ft_prop)
        
        return field_prop
    
    def _get_k_vectors(self):
        """获取k空间向量"""
        # 创建频率网格
        fx = torch.fft.fftfreq(self.layer_size, self.pixel_size)
        fy = torch.fft.fftfreq(self.layer_size, self.pixel_size)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        
        # 转换为k向量
        kx = 2 * np.pi * FX
        ky = 2 * np.pi * FY
        
        return kx.to(next(self.parameters()).device), ky.to(next(self.parameters()).device)
    
    def get_effective_phase_masks_for_layer(self, layer):
        """获取层的有效相位掩膜"""
        return layer.get_phase_masks()
    
    def analyze_layer_statistics(self, input_field):
        """分析层统计信息"""
        stats = {}
        
        with torch.no_grad():
            current_field = input_field
            
            for i, layer in enumerate(self.layers):
                # 计算层前的统计信息
                energy_before = torch.mean(torch.abs(current_field)**2)
                
                # 通过层
                if len(current_field.shape) == 5:  # [B, modes, wavelengths, H, W]
                    layer_output = torch.zeros_like(current_field)
                    for b in range(current_field.shape[0]):
                        for m in range(current_field.shape[1]):
                            for w in range(current_field.shape[2]):
                                layer_output[b, m, w] = layer(current_field[b, m, w], w)
                    current_field = layer_output
                else:
                    current_field = layer(current_field, 0)  # 默认使用第一个波长
                
                # 计算层后的统计信息
                energy_after = torch.mean(torch.abs(current_field)**2)
                
                stats[f'layer_{i}'] = {
                    'energy_before': energy_before.item(),
                    'energy_after': energy_after.item(),
                    'energy_ratio': (energy_after / energy_before).item() if energy_before > 0 else 0,
                    'phase_range': layer.get_phase_range()
                }
        
        return stats

class DiffractionLayer(nn.Module):
    """单个衍射层"""
    
    def __init__(self, layer_size, wavelengths, pixel_size, z_distance, layer_index=0):
        super(DiffractionLayer, self).__init__()
        self.layer_size = layer_size
        self.wavelengths = wavelengths
        self.num_wavelengths = len(wavelengths)
        self.pixel_size = pixel_size
        self.z_distance = z_distance
        self.layer_index = layer_index
        
        # 为每个波长创建独立的相位掩膜
        self.phase_masks = nn.ParameterList([
            nn.Parameter(torch.randn(layer_size, layer_size) * 0.1)
            for _ in range(self.num_wavelengths)
        ])
        
        print(f"   创建衍射层 {layer_index}: {self.num_wavelengths} 个波长相位掩膜")
    
    def forward(self, field, wavelength_idx):
        """
        前向传播
        field: [batch_size, height, width] 复数场
        wavelength_idx: 波长索引
        """
        # 应用相位调制
        phase_mask = self.phase_masks[wavelength_idx]
        modulated_field = field * torch.exp(1j * phase_mask)
        
        # 传播到下一层
        propagated_field = self._propagate(modulated_field, wavelength_idx)
        
        return propagated_field
    
    def _propagate(self, field, wavelength_idx):
        """使用角谱法传播"""
        wavelength = self.wavelengths[wavelength_idx]
        
        # FFT
        field_ft = torch.fft.fft2(field)
        
        # 创建传播核
        k = 2 * np.pi / wavelength
        fx = torch.fft.fftfreq(self.layer_size, self.pixel_size)
        fy = torch.fft.fftfreq(self.layer_size, self.pixel_size)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        
        kx = 2 * np.pi * FX.to(field.device)
        ky = 2 * np.pi * FY.to(field.device)
        kz = torch.sqrt(k**2 - kx**2 - ky**2 + 0j)
        
        # 传播相位
        propagation_kernel = torch.exp(1j * kz * self.z_distance)
        
        # 应用传播并逆FFT
        field_ft_prop = field_ft * propagation_kernel
        field_prop = torch.fft.ifft2(field_ft_prop)
        
        return field_prop
    
    def get_phase_masks(self):
        """获取所有波长的相位掩膜"""
        return [mask.detach().cpu().numpy() for mask in self.phase_masks]
    
    def get_phase_range(self):
        """获取相位范围"""
        ranges = []
        for mask in self.phase_masks:
            ranges.append((mask.min().item(), mask.max().item()))
        return ranges

class PhysicsConstrainedLoss(nn.Module):
    """物理约束损失函数"""
    
    def __init__(self, energy_weight=1.0, smoothness_weight=0.1):
        super(PhysicsConstrainedLoss, self).__init__()
        self.energy_weight = energy_weight
        self.smoothness_weight = smoothness_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets, input_energy=None):
        """
        计算总损失
        predictions: 模型预测 [B, modes, wavelengths, H, W]
        targets: 目标值 [B, modes, wavelengths, H, W] 或 None
        input_energy: 输入能量用于能量守恒约束
        """
        total_loss = 0.0
        
        # 主要损失（如果有目标）
        if targets is not None:
            main_loss = self.mse_loss(predictions, targets)
            total_loss += main_loss
        
        # 能量守恒约束
        if input_energy is not None:
            output_energy = torch.sum(torch.abs(predictions)**2, dim=(-2, -1))
            energy_loss = self.mse_loss(output_energy, input_energy)
            total_loss += self.energy_weight * energy_loss
        
        # 平滑性约束（相位掩膜的平滑性）
        smoothness_loss = self._compute_smoothness_loss(predictions)
        total_loss += self.smoothness_weight * smoothness_loss
        
        return total_loss
    
    def _compute_smoothness_loss(self, predictions):
        """计算平滑性损失"""
        # 计算梯度的L2范数作为平滑性度量
        grad_x = predictions[:, :, :, 1:, :] - predictions[:, :, :, :-1, :]
        grad_y = predictions[:, :, :, :, 1:] - predictions[:, :, :, :, :-1]
        
        smoothness = torch.mean(grad_x**2) + torch.mean(grad_y**2)
        return smoothness