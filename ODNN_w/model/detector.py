import torch
import torch.nn as nn

class RegressionDetector(nn.Module):
    """回归检测器"""
    
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        """计算场的强度"""
        return torch.square(torch.abs(inputs))
