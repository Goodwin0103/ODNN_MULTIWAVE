from matplotlib import pyplot as plt
import numpy as np
import os
import csv

class Visualizer:
    def __init__(self, config):
        self.config = config
    
    def organize_visibility_by_mode(self, results, config, num_layer_options):
        """
        按模式组织可见度数据：每个模式在不同波长和层数下的表现
        
        根据调试信息，数据结构为：
        results['visibility'][layer_idx] = [mode0_vis, mode1_vis, mode2_vis]
        每个值对应该层数下该模式的可见度（仅450nm波长）
        """
        print("按模式组织可见度数据...")
        
        num_modes = config.num_modes
        num_wavelengths = len(config.wavelengths)
        vis_data = results['visibility']
        
        print(f"配置: {num_modes}模式, {num_wavelengths}波长, {len(num_layer_options)}层数")
        print(f"原始数据结构: {[len(layer_data) for layer_data in vis_data]}")
        
        # 检查数据结构
        if len(vis_data[0]) == num_modes:
            print("检测到: 单波长数据（450nm），其他波长设为0")
            return self._organize_single_wavelength_data(vis_data, num_modes, num_wavelengths, num_layer_options)
        elif len(vis_data[0]) == num_modes * num_wavelengths:
            print("检测到: 完整多波长数据")
            return self._organize_multi_wavelength_data(vis_data, num_modes, num_wavelengths, num_layer_options)
        else:
            print("数据结构不匹配，从weights_pred重新计算")
            return self._recalculate_from_weights(results, config, num_layer_options)
    
    def _organize_single_wavelength_data(self, vis_data, num_modes, num_wavelengths, num_layer_options):
        """处理单波长数据（当前情况）"""
        visibility_by_mode = []
        
        for mode_idx in range(num_modes):
            mode_data = []
            for layer_idx, num_layers in enumerate(num_layer_options):
                wavelength_data = []
                for wave_idx in range(num_wavelengths):
                    if wave_idx == 0:  # 450nm有数据
                        vis_value = float(vis_data[layer_idx][mode_idx])
                    else:  # 550nm, 650nm设为0
                        vis_value = 0.0
                    wavelength_data.append(vis_value)
                mode_data.append(wavelength_data)
            visibility_by_mode.append(mode_data)
        
        return visibility_by_mode
    
    def _organize_multi_wavelength_data(self, vis_data, num_modes, num_wavelengths, num_layer_options):
        """处理完整多波长数据"""
        visibility_by_mode = []
        
        for mode_idx in range(num_modes):
            mode_data = []
            for layer_idx, num_layers in enumerate(num_layer_options):
                wavelength_data = []
                for wave_idx in range(num_wavelengths):
                    vis_idx = mode_idx * num_wavelengths + wave_idx
                    vis_value = float(vis_data[layer_idx][vis_idx])
                    wavelength_data.append(vis_value)
                mode_data.append(wavelength_data)
            visibility_by_mode.append(mode_data)
        
        return visibility_by_mode
    
    def _recalculate_from_weights(self, results, config, num_layer_options):
        """从权重数据重新计算可见度"""
        if 'weights_pred' not in results:
            print("错误: 无weights_pred数据")
            return self._create_zero_data(config, num_layer_options)
        
        weights_data = results['weights_pred']
        num_modes = config.num_modes
        num_wavelengths = len(config.wavelengths)
        visibility_by_mode = []
        
        for mode_idx in range(num_modes):
            mode_data = []
            for layer_idx, num_layers in enumerate(num_layer_options):
                wavelength_data = []
                layer_weights = weights_data[layer_idx]  # (3, 3, 9)
                
                for wave_idx in range(num_wavelengths):
                    # 提取该模式在该波长的权重
                    mode_start = mode_idx * 3
                    mode_end = mode_start + 3
                    mode_weights = layer_weights[:, wave_idx, mode_start:mode_end]
                    avg_weights = np.mean(mode_weights, axis=0)
                    
                    # 计算可见度
                    visibility = self._calculate_visibility(avg_weights)
                    wavelength_data.append(visibility)
                
                mode_data.append(wavelength_data)
            visibility_by_mode.append(mode_data)
        
        return visibility_by_mode
    
    def _calculate_visibility(self, weights):
        """计算可见度 = (max - min) / (max + min)"""
        weights = np.array(weights)
        if len(weights) <= 1:
            return 0.0
        
        max_val, min_val = np.max(weights), np.min(weights)
        if max_val + min_val == 0:
            return 0.0
        return (max_val - min_val) / (max_val + min_val)
    
    def _create_zero_data(self, config, num_layer_options):
        """创建零数据结构"""
        num_modes = config.num_modes
        num_wavelengths = len(config.wavelengths)
        return [[[0.0 for _ in range(num_wavelengths)] 
                 for _ in num_layer_options] 
                for _ in range(num_modes)]
    
    def plot_visibility_by_mode(self, visibility_by_mode, num_layer_options, save_path=None):
        """绘制按模式分组的可见度比较图"""
        num_modes = len(visibility_by_mode)
        wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        fig, axes = plt.subplots(1, num_modes, figsize=(5 * num_modes, 6))
        if num_modes == 1:
            axes = [axes]
        
        for mode_idx, ax in enumerate(axes):
            mode_data = np.array(visibility_by_mode[mode_idx])
            x = np.arange(len(num_layer_options))
            width = 0.25
            
            for wave_idx, (color, label) in enumerate(zip(colors, wavelength_labels)):
                values = mode_data[:, wave_idx]
                bars = ax.bar(x + wave_idx * width, values, width, 
                             label=label, color=color, alpha=0.8)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:  # 只显示非零值
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel('Visibility')
            ax.set_title(f'Mode {mode_idx + 1}')
            ax.set_xticks(x + width)
            ax.set_xticklabels([f'{layers}' for layers in num_layer_options])
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_ylim(0, 1.05)
        
        plt.suptitle('Visibility Comparison by Mode', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
        return fig
    
    def plot_visibility_comparison_by_mode_wavelength(self, visibility_by_mode, num_layer_options, save_path=None):
        """绘制模式-波长矩阵可见度图"""
        num_modes = len(visibility_by_mode)
        num_wavelengths = len(visibility_by_mode[0][0])
        wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
        
        fig, axes = plt.subplots(num_modes, num_wavelengths, 
                                figsize=(4*num_wavelengths, 3*num_modes))
        
        # 处理单个子图的情况
        if num_modes == 1 and num_wavelengths == 1:
            axes = np.array([[axes]])
        elif num_modes == 1:
            axes = axes.reshape(1, -1)
        elif num_wavelengths == 1:
            axes = axes.reshape(-1, 1)
        
        x = np.arange(len(num_layer_options))
        
        for mode_idx in range(num_modes):
            for wl_idx in range(num_wavelengths):
                ax = axes[mode_idx, wl_idx]
                
                # 获取数据
                vis_data = [visibility_by_mode[mode_idx][layer_idx][wl_idx] 
                           for layer_idx in range(len(num_layer_options))]
                
                # 绘制条形图
                bars = ax.bar(x, vis_data, width=0.6, color=f'C{mode_idx}', alpha=0.8)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                
                # 设置标题和标签
                if mode_idx == 0:
                    ax.set_title(f'{wavelength_labels[wl_idx]}', fontsize=12)
                if wl_idx == 0:
                    ax.set_ylabel(f'Mode {mode_idx+1}\nVisibility', fontsize=12)
                if mode_idx == num_modes-1:
                    ax.set_xlabel('Layers', fontsize=10)
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'{layers}' for layers in num_layer_options])
                
                ax.set_ylim(0, 1.05)
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.suptitle('Visibility Comparison by Mode and Wavelength', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
        return fig
    
    def print_visibility_summary(self, visibility_by_mode, num_layer_options):
        """打印可见度摘要"""
        print("\n=== 可见度数据摘要 ===")
        wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
        
        for mode_idx, mode_data in enumerate(visibility_by_mode):
            print(f"\n模式 {mode_idx + 1}:")
            print("层数\t" + "\t".join(wavelength_labels))
            print("-" * 32)
            
            for layer_idx, wavelength_data in enumerate(mode_data):
                values_str = "\t".join([f"{val:.3f}" for val in wavelength_data])
                print(f"{num_layer_options[layer_idx]}\t{values_str}")
        
        # 平均可见度
        print(f"\n=== 平均可见度（跨所有层数和波长）===")
        for mode_idx, mode_data in enumerate(visibility_by_mode):
            avg_vis = np.mean(np.array(mode_data))
            print(f"模式 {mode_idx + 1}: {avg_vis:.3f}")
    
    def save_visibility_data(self, visibility_by_mode, num_layer_options, save_path):
        """保存可见度数据到CSV"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
        
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Mode', 'Layers'] + wavelength_labels)
            
            for mode_idx, mode_data in enumerate(visibility_by_mode):
                for layer_idx, wavelength_data in enumerate(mode_data):
                    row = [f'Mode_{mode_idx+1}', num_layer_options[layer_idx]] + wavelength_data
                    writer.writerow(row)
        
        print(f"可见度数据已保存至: {save_path}")
