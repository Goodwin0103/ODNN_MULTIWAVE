import torch
import torch.nn as nn
from model.layers import WavelengthDependentDiffractionLayer, WavelengthDependentPropagation
from model.detector import RegressionDetector
import numpy as np
class WavelengthDependentODNNModel(nn.Module):
    """多波长ODNN模型"""
    
    def __init__(self, num_layers, layer_size, z_layers, z_prop, pixel_size, wavelengths):
        super().__init__()
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.wavelengths = wavelengths
        
        # 衍射层
        self.layers = nn.ModuleList([
            WavelengthDependentDiffractionLayer(layer_size, pixel_size, wavelengths, z_layers) 
            for _ in range(num_layers)
        ])
        
        # 传播层
        self.propagation = WavelengthDependentPropagation(layer_size, pixel_size, wavelengths, z_prop)
        
        # 检测器
        self.regression = RegressionDetector()
        
        self.num_wavelengths = len(wavelengths)

    def forward(self, x):
        """前向传播"""
        # 通过各衍射层
        for layer in self.layers:
            x = layer(x)
        
        # 最终传播
        x = self.propagation(x)
        
        # 检测
        return self.regression(x)
    
    def get_phase_masks(self):
        """获取相位掩膜"""
        phase_masks = []
        for layer in self.layers:
            phase = layer.phase.detach().cpu().numpy()
            phase = np.remainder(phase, 2 * np.pi)
            phase_masks.append(phase)
        return phase_masks
