import torch
import torch.nn as nn
import numpy as np
from propagation.optical_propagation import propagation_multi

class WavelengthDependentDiffractionLayer(nn.Module):
    """波长相关的衍射层"""
    
    def __init__(self, units, dx, wavelengths, z):
        super().__init__()
        self.units = units
        self.Nx = units
        self.dx = dx
        self.wavelengths = wavelengths
        self.num_wavelengths = len(wavelengths)
        self.z = z
        self.phase = nn.Parameter(torch.randn(units, units, dtype=torch.float32) * 2 * np.pi)

    def forward(self, x):
        """前向传播"""
        # 将波长转为张量
        lam = torch.as_tensor(self.wavelengths, dtype=x.dtype, device=x.device)
        
        # 计算波长缩放系数
        lam_ratio = lam[0] / lam
        phase_scaled = (self.phase.to(dtype=x.dtype, device=x.device) * 
                       lam_ratio[:, None, None])
        
        # 相位调制
        x = x * torch.exp(1j * phase_scaled)
        
        # 传播
        return propagation_multi(x, z=self.z, wavelengths=lam, 
                               pixel_size=self.dx, device=x.device)


class WavelengthDependentPropagation(nn.Module):
    """波长相关的传播层"""
    
    def __init__(self, units, dx, wavelengths, z):
        super().__init__()
        self.dx = dx
        self.wavelengths = wavelengths
        self.z = z

    def forward(self, x):
        """前向传播"""
        return propagation_multi(x, z=self.z, wavelengths=self.wavelengths,
                               pixel_size=self.dx, device=x.device)
