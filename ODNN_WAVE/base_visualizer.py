from matplotlib import pyplot as plt
import numpy as np
import os
import re
import seaborn as sns

class BaseVisualizer:
    """基础可视化器 - 包含通用配置和辅助方法"""
    
    def __init__(self, config):
        self.config = config
        
        # 设置英文字体和样式
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        
        # 设置颜色主题
        self.colors = {
            'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'gradient': plt.cm.viridis,
            'heatmap': 'RdYlBu_r',
            'intensity': 'hot'
        }

    def _extract_file_info(self, filename):
        """解析文件名提取配置信息"""
        print(f"🔍 解析文件名: {filename}")
        
        # 提取模式 - 支持1-based索引
        mode_patterns = [
            r'mode(\d+)', r'Mode(\d+)', r'MODE(\d+)',
            r'm(\d+)', r'_(\d+)mode'
        ]
        
        mode_idx = None
        for pattern in mode_patterns:
            mode_match = re.search(pattern, filename, re.IGNORECASE)
            if mode_match:
                mode_idx = int(mode_match.group(1)) - 1
                break
        
        if mode_idx is None:
            return None
        
        if mode_idx < 0 or mode_idx >= 3:
            print(f"  ⚠ 转换后模式索引超出范围: {mode_idx}")
        
        # 提取波长
        wl_match = re.search(r'(\d+)nm', filename)
        if not wl_match:
            return None
        wl_nm = int(wl_match.group(1))
        
        # 提取层数
        layer_patterns = [
            r'(\d+)layers', r'(\d+)layer',
            r'L(\d+)', r'_(\d+)L'
        ]
        
        layers = None
        for pattern in layer_patterns:
            layer_match = re.search(pattern, filename, re.IGNORECASE)
            if layer_match:
                layers = int(layer_match.group(1))
                break
        
        if layers is None:
            return None
        
        return {
            'mode': mode_idx,
            'wavelength': wl_nm,
            'layers': layers
        }

    def _match_config_unified(self, key, layers, mode_idx, wavelength=None):
        """统一的配置匹配函数"""
        layer_patterns = [f'L{layers}_', f'layers{layers}', f'{layers}layers', f'{layers}L']
        layer_match = any(pattern in key for pattern in layer_patterns)
        
        mode_patterns = [f'_M{mode_idx+1}_', f'_mode{mode_idx+1}', f'mode{mode_idx+1}', f'M{mode_idx+1}']
        mode_match = any(pattern in key for pattern in mode_patterns)
        
        if wavelength is not None:
            wavelength_patterns = [f'_{wavelength}nm', f'{wavelength}nm']
            wavelength_match = any(pattern in key for pattern in wavelength_patterns)
            return layer_match and mode_match and wavelength_match
        else:
            return layer_match and mode_match

    def _extract_matching_values(self, cross_matrix_data, layers, mode_idx, wavelength=None):
        """提取匹配的数值"""
        matching_values = []
        
        for key, data in cross_matrix_data.items():
            if self._match_config_unified(key, layers, mode_idx, wavelength):
                if 'focus_concentration' in data:
                    matching_values.append(data['focus_concentration'])
        
        return np.mean(matching_values) if matching_values else 0

    def _extract_wavelengths_improved(self, cross_matrix_data):
        """改进的波长提取方法"""
        wavelengths = set()
        
        for key in cross_matrix_data.keys():
            patterns = [r'(\d+)nm', r'_(\d+)nm', r'nm(\d+)', r'wl(\d+)', r'(\d+)_nm']
            
            for pattern in patterns:
                match = re.search(pattern, key, re.IGNORECASE)
                if match:
                    wavelengths.add(int(match.group(1)))
                    break
        
        return sorted(list(wavelengths))
