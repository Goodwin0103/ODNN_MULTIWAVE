import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class Visualizer:
    """可视化工具"""
    
    def __init__(self, config):
        self.config = config
    
    def plot_field_amplitude(self, field, title=None, figsize=(10, 8), save_path=None):
        """
        绘制光场振幅
        
        参数:
            field: 光场
            title: 标题
            figsize: 图像大小
            save_path: 保存路径
        """
        if torch.is_tensor(field):
            field = field.detach().cpu().numpy()
            
        if np.iscomplexobj(field):
            field = np.abs(field)
        
