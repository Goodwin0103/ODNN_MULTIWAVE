import torch
import torch.nn as nn
import numpy as np
from core.propagation import propagation_multi

class WavelengthDependentDiffractionLayer(nn.Module):
    """波长相关的衍射层"""
    
    def __init__(self, units: int, dx: float, wavelengths: np.ndarray, z: float, layer_idx: int = 0):
        super().__init__()
        self.units = units
        self.dx = dx
        self.wavelengths = wavelengths
        self.z = z
        self.layer_idx = layer_idx
        
        # 注册波长参数
        self.register_buffer("wavelength_tensor", torch.tensor(wavelengths, dtype=torch.float32))
        
        # 相位掩膜初始化
        self.phase = nn.Parameter(torch.rand(units, units) * np.pi)
        
        # 设置参考波长索引（默认为中间波长）
        self.reference_wl_idx = len(wavelengths) // 2
        
        # 构建波长相关的系数
        self._build_wavelength_coefficients()
    
    def _build_wavelength_coefficients(self):
        """计算波长相关的系数"""
        ref_wl = self.wavelengths[self.reference_wl_idx]
        self.wl_coefficients = torch.tensor([ref_wl/wl for wl in self.wavelengths], dtype=torch.float32)
        self.register_buffer("wavelength_coefficients", self.wl_coefficients)
    
    def forward(self, x, wavelength_idx=None):
        """
        前向传播
        
        参数:
            x: 输入光场 [batch_size, channels, H, W]
            wavelength_idx: 波长索引，若为None则处理所有波长
            
        返回:
            波长调制后的光场
        """
        if wavelength_idx is not None:
            # 单波长处理
            phase_scaled = self.phase * self.wavelength_coefficients[wavelength_idx]
            phase_term = torch.exp(1j * phase_scaled)
            modulated = x * phase_term
        else:
            # 多波长处理
            batch_size, num_channels, H, W = x.shape
            outputs = []
            
            for wl_idx in range(len(self.wavelengths)):
                phase_scaled = self.phase * self.wavelength_coefficients[wl_idx]
                phase_term = torch.exp(1j * phase_scaled)
                
                if num_channels == len(self.wavelengths):
                    # 每个通道对应一个波长
                    modulated = x[:, wl_idx:wl_idx+1] * phase_term
                else:
                    # 单一通道应用于所有波长
                    modulated = x * phase_term
                
                outputs.append(modulated)
            
            if len(outputs) > 1:
                modulated = torch.cat(outputs, dim=1)
            else:
                modulated = outputs[0]
        
        return modulated


class DiffractionNetwork(nn.Module):
    """衍射网络"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 创建多层衍射网络
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            self.layers.append(
                WavelengthDependentDiffractionLayer(
                    units=config.layer_size,
                    dx=config.pixel_size,
                    wavelengths=config.wavelengths,
                    z=config.z_layers,
                    layer_idx=i
                )
            )
        
        # 传播参数
        self.dx = config.pixel_size
        self.wavelengths = config.wavelengths
        self.z_prop = config.z_prop
    
    def forward(self, x, mode_idx=None, wavelength_idx=None):
        """
        前向传播
        
        参数:
            x: 输入光场 [batch_size, channels, H, W]
            mode_idx: 模式索引（可选）
            wavelength_idx: 波长索引（可选）
            
        返回:
            传播后的光场
        """
        # 输入处理
        if mode_idx is not None and x.shape[1] > 1:
            x = x[:, mode_idx:mode_idx+1]
            
        # 通过每层衍射网络
        field = x
        for layer in self.layers:
            # 应用相位调制
            field = layer(field, wavelength_idx)
            
            # 光场传播
            field = propagation_multi(field, self.z_prop, self.wavelengths, self.dx, 
                                     wavelength_idx=wavelength_idx)
        
        return field
