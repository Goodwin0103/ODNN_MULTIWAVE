from matplotlib import pyplot as plt
import numpy as np
import os
import csv
import json
import glob
import re
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class Visualizer:
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
    
    def calculate_visibility_from_simulation_results(self, save_dir, config, num_layer_options):
        """
        从传播仿真结果计算真实的 visibility
        
        参数:
            save_dir: 仿真结果保存目录
            config: 配置对象
            num_layer_options: 层数选项列表
        
        返回:
            dict: 按模式组织的 visibility 数据
        """
        print("🔍 从传播仿真结果计算真实 visibility...")
        
        # 查找所有仿真结果文件
        result_files = glob.glob(os.path.join(save_dir, "MC_single_*.npy"))
        
        if not result_files:
            print("❌ 未找到仿真结果文件")
            return None
        
        print(f"找到 {len(result_files)} 个仿真结果文件")
        
        # 组织数据结构
        visibility_data = {}
        
        for file_path in result_files:
            filename = os.path.basename(file_path)
            
            # 提取文件信息
            file_info = self._extract_file_info(filename)
            if not file_info:
                print(f"⚠ 无法解析文件名: {filename}")
                continue
            
            mode_idx, wl_nm, layers = file_info['mode'], file_info['wavelength'], file_info['layers']
            
            try:
                # 加载仿真数据
                data = np.load(file_path, allow_pickle=True)
                
                # 计算真实的 visibility (聚焦效率)
                visibility = self._calculate_focus_efficiency(data)
                
                # 存储数据
                key = (layers, mode_idx, wl_nm)
                visibility_data[key] = visibility
                
                print(f"  {layers}层, 模式{mode_idx}, {wl_nm}nm: visibility = {visibility:.4f}")
                
            except Exception as e:
                print(f"❌ 处理文件 {filename} 时出错: {e}")
                continue
        
        print(f"成功处理 {len(visibility_data)} 个数据点")
        
        # 按模式重新组织数据
        organized_data = self._reorganize_visibility_by_mode(visibility_data, config, num_layer_options)
        
        return organized_data

    def _extract_file_info(self, filename):
        """修复版文件名解析 - 处理1-based模式索引"""
        print(f"🔍 解析文件名: {filename}")
        
        # 提取模式 - 支持1-based索引
        mode_patterns = [
            r'mode(\d+)',
            r'Mode(\d+)', 
            r'MODE(\d+)',
            r'm(\d+)',
            r'_(\d+)mode'
        ]
        
        mode_idx = None
        for pattern in mode_patterns:
            mode_match = re.search(pattern, filename, re.IGNORECASE)
            if mode_match:
                # 关键修复：将1-based转换为0-based
                mode_idx = int(mode_match.group(1)) - 1  # 减1转换为0-based
                print(f"  模式匹配: {pattern} -> 原值={int(mode_match.group(1))}, 转换后={mode_idx}")
                break
        
        if mode_idx is None:
            print(f"  ❌ 无法提取模式信息")
            return None
        
        # 检查转换后的模式索引范围
        if mode_idx < 0 or mode_idx >= 3:  # 0-based: 0,1,2
            print(f"  ⚠ 转换后模式索引超出范围: {mode_idx}")
        
        # 提取波长
        wl_match = re.search(r'(\d+)nm', filename)
        if not wl_match:
            print(f"  ❌ 无法提取波长信息")
            return None
        
        wl_nm = int(wl_match.group(1))
        print(f"  波长匹配: {wl_nm}nm")
        
        # 提取层数
        layer_patterns = [
            r'(\d+)layers',
            r'(\d+)layer',
            r'L(\d+)',
            r'_(\d+)L'
        ]
        
        layers = None
        for pattern in layer_patterns:
            layer_match = re.search(pattern, filename, re.IGNORECASE)
            if layer_match:
                layers = int(layer_match.group(1))
                print(f"  层数匹配: {pattern} -> {layers}")
                break
        
        if layers is None:
            print(f"  ❌ 无法提取层数信息")
            return None
        
        result = {
            'mode': mode_idx,      # 现在是0-based
            'wavelength': wl_nm,
            'layers': layers
        }
        
        print(f"  ✅ 解析结果: {result}")
        return result

    def _calculate_focus_efficiency(self, field_data):
        """
        从场数据计算聚焦效率
        
        参数:
            field_data: 复数场数据或强度数据
        
        返回:
            float: 聚焦效率 (0-1)
        """
        # 计算强度
        if np.iscomplexobj(field_data):
            intensity = np.abs(field_data)**2
        else:
            intensity = np.abs(field_data)**2
        
        # 处理多维数据
        if intensity.ndim > 2:
            # 对前面的维度求和，保留最后两个空间维度
            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
        
        # 确保是2D数组
        if intensity.ndim != 2:
            print(f"⚠ 数据维度异常: {intensity.shape}")
            return 0.0
        
        # 归一化
        max_intensity = np.max(intensity)
        if max_intensity > 0:
            intensity = intensity / max_intensity
        else:
            return 0.0
        
        # 计算聚焦效率（中心区域能量占比）
        H, W = intensity.shape
        center_y, center_x = H // 2, W // 2
        
        # 定义聚焦区域（可调整）
        focus_radius = min(H, W) // 8  # 聚焦区域半径
        
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        focus_mask = ((y_grid - center_y)**2 + (x_grid - center_x)**2) <= focus_radius**2
        
        total_intensity = np.sum(intensity)
        focus_intensity = np.sum(intensity[focus_mask])
        
        focus_efficiency = focus_intensity / total_intensity if total_intensity > 0 else 0
        
        return focus_efficiency

    def _reorganize_visibility_by_mode(self, visibility_data, config, num_layer_options):
        """重新按模式组织 visibility 数据 - 修复版"""
        
        organized_data = []
        
        print(f"\n🔄 重新组织数据 (修复版):")
        print(f"  配置: {config.num_modes} 个模式, {len(num_layer_options)} 个层数选项")
        print(f"  可见性数据键值数量: {len(visibility_data)}")
        
        # 显示所有可用的键值
        print(f"  可用键值 (layers, mode, wavelength):")
        for key in sorted(visibility_data.keys()):
            print(f"    {key}: {visibility_data[key]:.4f}")
        
        missing_keys = []
        found_keys = []
        
        for mode_idx in range(config.num_modes):  # 0, 1, 2
            mode_data = []
            print(f"\n  处理模式 {mode_idx} (0-based):")
            
            for layers in num_layer_options:
                wavelength_data = []
                
                for wl in config.wavelengths:
                    wl_nm = int(wl * 1e9)
                    key = (layers, mode_idx+1, wl_nm)  # 现在使用0-based模式索引
                    
                    if key in visibility_data:
                        visibility = visibility_data[key]
                        found_keys.append(key)
                        print(f"    ✅ {key}: {visibility:.4f}")
                    else:
                        visibility = 0.0
                        missing_keys.append(key)
                        print(f"    ❌ {key}: 缺失")
                    
                    wavelength_data.append(visibility)
                
                mode_data.append(wavelength_data)
            
            organized_data.append(mode_data)
        
        print(f"\n📈 数据统计:")
        print(f"  找到的键值: {len(found_keys)}")
        print(f"  缺失的键值: {len(missing_keys)}")
        
        if missing_keys:
            print(f"  前10个缺失键值:")
            for key in missing_keys[:10]:
                print(f"    {key}")
        
        return organized_data

    def create_visibility_comparison(self, original_visibility, real_visibility, config, num_layer_options, save_path):
        """
        创建原始 visibility 和真实 visibility 的对比分析
        """
        print("创建 Visibility 对比分析图...")
        
        fig, axes = plt.subplots(2, config.num_modes, figsize=(5*config.num_modes, 10))
        
        if config.num_modes == 1:
            axes = axes.reshape(-1, 1)
        
        wavelength_labels = [f'{int(wl*1e9)}nm' for wl in config.wavelengths]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for mode_idx in range(config.num_modes):
            # 上排：原始 visibility（基于权重预测）
            ax1 = axes[0, mode_idx]
            
            # 处理原始数据
            if isinstance(original_visibility, list) and len(original_visibility) > mode_idx:
                original_data = np.array(original_visibility[mode_idx])
                if original_data.ndim == 1:
                    # 如果是1D数组，假设是单波长数据
                    original_data = original_data.reshape(-1, 1)
            else:
                # 如果没有原始数据，创建零数组
                original_data = np.zeros((len(num_layer_options), len(wavelength_labels)))
            
            for wl_idx, wl_label in enumerate(wavelength_labels):
                if wl_idx < original_data.shape[1]:
                    color = colors[wl_idx % len(colors)]
                    ax1.plot(num_layer_options, original_data[:, wl_idx], 
                            'o-', label=wl_label, linewidth=2, markersize=6, color=color)
                    
                    # 标注数值
                    for layer_idx, layer_num in enumerate(num_layer_options):
                        if layer_idx < len(original_data):
                            value = original_data[layer_idx, wl_idx]
                            ax1.annotate(f'{value:.3f}', 
                                       (layer_num, value), 
                                       textcoords="offset points", 
                                       xytext=(0,10), ha='center', fontsize=8)
            
            ax1.set_title(f'模式 {mode_idx+1} - 原始 Visibility (权重预测)', fontweight='bold', fontsize=12)
            ax1.set_xlabel('层数', fontsize=10)
            ax1.set_ylabel('Visibility', fontsize=10)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)
            ax1.set_xticks(num_layer_options)
            
            # 下排：真实 visibility（基于仿真结果）
            ax2 = axes[1, mode_idx]
            
            if real_visibility and len(real_visibility) > mode_idx:
                real_data = np.array(real_visibility[mode_idx])
                if real_data.ndim == 1:
                    real_data = real_data.reshape(-1, 1)
            else:
                real_data = np.zeros((len(num_layer_options), len(wavelength_labels)))
            
            for wl_idx, wl_label in enumerate(wavelength_labels):
                if wl_idx < real_data.shape[1]:
                    color = colors[wl_idx % len(colors)]
                    ax2.plot(num_layer_options, real_data[:, wl_idx], 
                            's-', label=wl_label, linewidth=2, markersize=6, color=color)
                    
                    # 标注数值
                    for layer_idx, layer_num in enumerate(num_layer_options):
                        if layer_idx < len(real_data):
                            value = real_data[layer_idx, wl_idx]
                            ax2.annotate(f'{value:.3f}', 
                                       (layer_num, value), 
                                       textcoords="offset points", 
                                       xytext=(0,10), ha='center', fontsize=8)
            
            ax2.set_title(f'模式 {mode_idx+1} - 真实 Visibility (仿真结果)', fontweight='bold', fontsize=12)
            ax2.set_xlabel('层数', fontsize=10)
            ax2.set_ylabel('Visibility', fontsize=10)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
            ax2.set_xticks(num_layer_options)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Visibility 对比分析图已保存: {save_path}")

    def organize_visibility_by_mode(self, results, config, num_layer_options):
        """
        Organize visibility data by mode: each mode's performance across different wavelengths and layer numbers
        
        Based on debug info, data structure is:
        results['visibility'][layer_idx] = [mode0_vis, mode1_vis, mode2_vis]
        Each value corresponds to the visibility of that mode at that layer number (450nm wavelength only)
        """
        print("📊 Organizing visibility data by mode...")
        
        num_modes = config.num_modes
        num_wavelengths = len(config.wavelengths)
        vis_data = results['visibility']
        
        print(f"Configuration: {num_modes} modes, {num_wavelengths} wavelengths, {len(num_layer_options)} layer options")
        print(f"Raw data structure: {[len(layer_data) for layer_data in vis_data]}")
        
        # Check data structure
        if len(vis_data[0]) == num_modes:
            print("✅ Detected: Single wavelength data (450nm), others set to 0")
            return self._organize_single_wavelength_data(vis_data, num_modes, num_wavelengths, num_layer_options)
        elif len(vis_data[0]) == num_modes * num_wavelengths:
            print("✅ Detected: Complete multi-wavelength data")
            return self._organize_multi_wavelength_data(vis_data, num_modes, num_wavelengths, num_layer_options)
        else:
            print("⚠ Data structure mismatch, recalculating from weights_pred")
            return self._recalculate_from_weights(results, config, num_layer_options)
    
    def _organize_single_wavelength_data(self, vis_data, num_modes, num_wavelengths, num_layer_options):
        """Handle single wavelength data (current situation)"""
        visibility_by_mode = []
        
        for mode_idx in range(num_modes):
            mode_visibility = []
            
            for layer_idx in range(len(num_layer_options)):
                wavelength_vis = []
                
                # First wavelength (450nm) - actual data
                if layer_idx < len(vis_data) and mode_idx < len(vis_data[layer_idx]):
                    actual_vis = vis_data[layer_idx][mode_idx]
                else:
                    actual_vis = 0.0
                wavelength_vis.append(actual_vis)
                
                # Other wavelengths - set to 0 or estimate
                for wl_idx in range(1, num_wavelengths):
                    wavelength_vis.append(0.0)  # Set to 0 for missing wavelengths
                
                mode_visibility.append(wavelength_vis)
            
            visibility_by_mode.append(mode_visibility)
        
        return visibility_by_mode
    
    def _organize_multi_wavelength_data(self, vis_data, num_modes, num_wavelengths, num_layer_options):
        """Handle complete multi-wavelength data"""
        visibility_by_mode = []
        
        for mode_idx in range(num_modes):
            mode_visibility = []
            
            for layer_idx in range(len(num_layer_options)):
                wavelength_vis = []
                
                for wl_idx in range(num_wavelengths):
                    data_idx = mode_idx * num_wavelengths + wl_idx
                    
                    if layer_idx < len(vis_data) and data_idx < len(vis_data[layer_idx]):
                        vis_value = vis_data[layer_idx][data_idx]
                    else:
                        vis_value = 0.0
                    
                    wavelength_vis.append(vis_value)
                
                mode_visibility.append(wavelength_vis)
            
            visibility_by_mode.append(mode_visibility)
        
        return visibility_by_mode
    
    def _recalculate_from_weights(self, results, config, num_layer_options):
        """Recalculate visibility from weights_pred if available"""
        if 'weights_pred' not in results:
            print("❌ No weights_pred data available for recalculation")
            return self._create_empty_visibility_data(config.num_modes, len(config.wavelengths), len(num_layer_options))
        
        print("🔄 Recalculating visibility from weights_pred...")
        # This would need to be implemented based on your specific calculation method
        # For now, return empty data
        return self._create_empty_visibility_data(config.num_modes, len(config.wavelengths), len(num_layer_options))
    
    def _create_empty_visibility_data(self, num_modes, num_wavelengths, num_layers):
        """Create empty visibility data structure"""
        visibility_by_mode = []
        for mode_idx in range(num_modes):
            mode_visibility = []
            for layer_idx in range(num_layers):
                wavelength_vis = [0.0] * num_wavelengths
                mode_visibility.append(wavelength_vis)
            visibility_by_mode.append(mode_visibility)
        return visibility_by_mode

    def create_detailed_visibility_analysis(self, visibility_by_mode, config, num_layer_options, save_path=None, title_suffix=""):
        """
        Create detailed visibility analysis with both bar charts and heatmaps for each mode
        
        Args:
            visibility_by_mode: List of visibility data organized by mode
            config: Configuration object
            num_layer_options: List of layer numbers
            save_path: Path to save the figure
            title_suffix: Additional suffix for the title
        """
        print("🎨 Creating detailed visibility analysis...")
        
        num_modes = len(visibility_by_mode)
        wavelength_labels = [f'{int(wl*1e9)}nm' for wl in config.wavelengths]
        
        # Create figure with subplots: 2 rows per mode (bar chart + heatmap)
        fig = plt.figure(figsize=(15, 6 * num_modes))
        
        # Define colors for different wavelengths
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for mode_idx in range(num_modes):
            mode_data = np.array(visibility_by_mode[mode_idx])  # Shape: (num_layers, num_wavelengths)
            
            # Find best performance for this mode
            best_vis = np.max(mode_data)
            best_pos = np.unravel_index(np.argmax(mode_data), mode_data.shape)
            best_layer = num_layer_options[best_pos[0]]
            best_wl = wavelength_labels[best_pos[1]]
            
            # Bar chart subplot
            ax1 = plt.subplot(2 * num_modes, 2, 2 * mode_idx + 1)
            
            x = np.arange(len(num_layer_options))
            width = 0.8 / len(wavelength_labels)
            
            for wl_idx, wl_label in enumerate(wavelength_labels):
                offset = (wl_idx - len(wavelength_labels)/2 + 0.5) * width
                values = mode_data[:, wl_idx]
                color = colors[wl_idx % len(colors)]
                
                bars = ax1.bar(x + offset, values, width, label=wl_label, 
                              color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if value > 0.01:  # Only show labels for non-zero values
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax1.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Visibility', fontsize=12, fontweight='bold')
            ax1.set_title(f'Mode {mode_idx + 1} - Visibility Comparison\nBest: {best_layer}L@{best_wl} ({best_vis:.3f})', 
                         fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(num_layer_options)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, 1.0)
            
            # Add best performance highlight
            if best_vis > 0:
                ax1.axhline(y=best_vis, color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax1.text(0.02, 0.98, f'Best: {best_vis:.3f}', transform=ax1.transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                        fontsize=10, fontweight='bold', va='top')
            
            # Heatmap subplot
            ax2 = plt.subplot(2 * num_modes, 2, 2 * mode_idx + 2)
            
            # Transpose for better visualization (wavelengths as rows, layers as columns)
            heatmap_data = mode_data.T
            
            im = ax2.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            
            # Add text annotations
            for i in range(len(wavelength_labels)):
                for j in range(len(num_layer_options)):
                    value = heatmap_data[i, j]
                    color = 'white' if value < 0.5 else 'black'
                    ax2.text(j, i, f'{value:.3f}', ha='center', va='center', 
                            color=color, fontweight='bold', fontsize=10)
            
            ax2.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Wavelength', fontsize=12, fontweight='bold')
            ax2.set_title(f'Mode {mode_idx + 1} - Heatmap', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(num_layer_options)))
            ax2.set_xticklabels(num_layer_options)
            ax2.set_yticks(range(len(wavelength_labels)))
            ax2.set_yticklabels(wavelength_labels)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Visibility', fontsize=10, fontweight='bold')
        
        # Overall title
        fig.suptitle(f'Detailed Visibility Analysis by Mode {title_suffix}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Detailed visibility analysis saved: {save_path}")
        
        plt.show()

    def create_mode_wavelength_matrix_analysis(self, visibility_by_mode, config, num_layer_options, save_path=None):
        """
        Create mode-wavelength visibility matrix analysis
        """
        print("🎨 Creating mode-wavelength matrix analysis...")
        
        num_modes = len(visibility_by_mode)
        num_wavelengths = len(config.wavelengths)
        wavelength_labels = [f'{int(wl*1e9)}nm' for wl in config.wavelengths]
        
        # Create figure
        fig, axes = plt.subplots(num_modes, num_wavelengths, figsize=(5*num_wavelengths, 4*num_modes))
        
        # Handle single mode or single wavelength cases
        if num_modes == 1 and num_wavelengths == 1:
            axes = np.array([[axes]])
        elif num_modes == 1:
            axes = axes.reshape(1, -1)
        elif num_wavelengths == 1:
            axes = axes.reshape(-1, 1)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Calculate performance summary
        performance_summary = []
        
        for mode_idx in range(num_modes):
            mode_data = np.array(visibility_by_mode[mode_idx])
            mode_avg = np.mean(mode_data)
            mode_max = np.max(mode_data)
            performance_summary.append(f"Mode{mode_idx+1}: Avg={mode_avg:.3f}, Max={mode_max:.3f}")
            
            for wl_idx in range(num_wavelengths):
                ax = axes[mode_idx, wl_idx]
                
                # Get data for this mode and wavelength
                wl_data = mode_data[:, wl_idx]
                color = colors[wl_idx % len(colors)]
                
                # Create bar plot
                bars = ax.bar(num_layer_options, wl_data, color=color, alpha=0.8, 
                             edgecolor='white', linewidth=1)
                
                # Add value labels
                for bar, value in zip(bars, wl_data):
                    if value > 0.01:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Highlight best performance
                max_val = np.max(wl_data)
                if max_val > 0:
                    max_idx = np.argmax(wl_data)
                    best_layer = num_layer_options[max_idx]
                    
                    # Add red border to best bar
                    bars[max_idx].set_edgecolor('red')
                    bars[max_idx].set_linewidth(3)
                
                ax.set_title(f'{wavelength_labels[wl_idx]}', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 1.0)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks(num_layer_options)
                
                # Add mode label on the left
                if wl_idx == 0:
                    ax.set_ylabel(f'Mode {mode_idx + 1}\nVisibility', fontsize=11, fontweight='bold')
                
                # Add x-label on bottom row
                if mode_idx == num_modes - 1:
                    ax.set_xlabel('Layers', fontsize=10, fontweight='bold')
        
        # Add performance summary at the bottom
        summary_text = " | ".join(performance_summary)
        fig.text(0.5, 0.02, f"Performance Summary: {summary_text}", 
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        fig.suptitle('Mode-Wavelength Visibility Matrix Analysis', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Mode-wavelength matrix analysis saved: {save_path}")
        
        plt.show()

    def create_comprehensive_visibility_report(self, visibility_by_mode, config, num_layer_options, save_dir):
        """
        Create comprehensive visibility analysis report
        """
        print("📋 Creating comprehensive visibility report...")
        
        # 1. Detailed analysis by mode
        detailed_path = os.path.join(save_dir, 'detailed_visibility_analysis.png')
        self.create_detailed_visibility_analysis(visibility_by_mode, config, num_layer_options, detailed_path)
        
        # 2. Mode-wavelength matrix
        matrix_path = os.path.join(save_dir, 'mode_wavelength_matrix.png')
        self.create_mode_wavelength_matrix_analysis(visibility_by_mode, config, num_layer_options, matrix_path)
        
        # 3. Performance summary statistics
        self._create_performance_statistics(visibility_by_mode, config, num_layer_options, save_dir)
        
        # 4. Export data to CSV
        self._export_visibility_data(visibility_by_mode, config, num_layer_options, save_dir)
        
        print("✅ Comprehensive visibility report created")

    def _create_performance_statistics(self, visibility_by_mode, config, num_layer_options, save_dir):
        """Create performance statistics summary"""
        print("📊 Creating performance statistics...")
        
        wavelength_labels = [f'{int(wl*1e9)}nm' for wl in config.wavelengths]
        
        # Calculate statistics
        stats = {
            'overall': {
                'max_visibility': 0,
                'avg_visibility': 0,
                'best_config': None
            },
            'by_mode': [],
            'by_wavelength': [],
            'by_layers': []
        }
        
        all_values = []
        
        # Mode statistics
        for mode_idx, mode_data in enumerate(visibility_by_mode):
            mode_array = np.array(mode_data)
            all_values.extend(mode_array.flatten())
            
            mode_stats = {
                'mode': mode_idx + 1,
                'max_vis': np.max(mode_array),
                'min_vis': np.min(mode_array),
                'avg_vis': np.mean(mode_array),
                'std_vis': np.std(mode_array),
                'best_layer': None,
                'best_wavelength': None
            }
            
            # Find best configuration for this mode
            best_pos = np.unravel_index(np.argmax(mode_array), mode_array.shape)
            mode_stats['best_layer'] = num_layer_options[best_pos[0]]
            mode_stats['best_wavelength'] = wavelength_labels[best_pos[1]]
            
            stats['by_mode'].append(mode_stats)
        
        # Wavelength statistics
        for wl_idx, wl_label in enumerate(wavelength_labels):
            wl_values = []
            for mode_data in visibility_by_mode:
                mode_array = np.array(mode_data)
                if wl_idx < mode_array.shape[1]:
                    wl_values.extend(mode_array[:, wl_idx])
            
            if wl_values:
                wl_stats = {
                    'wavelength': wl_label,
                    'max_vis': np.max(wl_values),
                    'min_vis': np.min(wl_values),
                    'avg_vis': np.mean(wl_values),
                    'std_vis': np.std(wl_values)
                }
                stats['by_wavelength'].append(wl_stats)
        
        # Layer statistics
        for layer_idx, layer_num in enumerate(num_layer_options):
            layer_values = []
            for mode_data in visibility_by_mode:
                mode_array = np.array(mode_data)
                if layer_idx < mode_array.shape[0]:
                    layer_values.extend(mode_array[layer_idx, :])
            
            if layer_values:
                layer_stats = {
                    'layers': layer_num,
                    'max_vis': np.max(layer_values),
                    'min_vis': np.min(layer_values),
                    'avg_vis': np.mean(layer_values),
                    'std_vis': np.std(layer_values)
                }
                stats['by_layers'].append(layer_stats)
        
        # Overall statistics
        if all_values:
            stats['overall']['max_visibility'] = np.max(all_values)
            stats['overall']['avg_visibility'] = np.mean(all_values)
            
            # Find best overall configuration
            best_overall = 0
            best_config = None
            for mode_idx, mode_data in enumerate(visibility_by_mode):
                mode_array = np.array(mode_data)
                mode_max = np.max(mode_array)
                if mode_max > best_overall:
                    best_overall = mode_max
                    best_pos = np.unravel_index(np.argmax(mode_array), mode_array.shape)
                    best_config = {
                        'mode': mode_idx + 1,
                        'layers': num_layer_options[best_pos[0]],
                        'wavelength': wavelength_labels[best_pos[1]],
                        'visibility': mode_max
                    }
            
            stats['overall']['best_config'] = best_config
        
        # Save statistics to JSON
        stats_path = os.path.join(save_dir, 'visibility_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Create visualization
        self._visualize_statistics(stats, save_dir)
        
        print(f"✅ Performance statistics saved: {stats_path}")

    def _visualize_statistics(self, stats, save_dir):
        """Create statistics visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Mode comparison
        ax1 = axes[0, 0]
        mode_data = stats['by_mode']
        modes = [f"Mode {m['mode']}" for m in mode_data]
        max_vis = [m['max_vis'] for m in mode_data]
        avg_vis = [m['avg_vis'] for m in mode_data]
        
        x = np.arange(len(modes))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, max_vis, width, label='Max Visibility', color='#ff7f0e', alpha=0.8)
        bars2 = ax1.bar(x + width/2, avg_vis, width, label='Avg Visibility', color='#1f77b4', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Mode', fontweight='bold')
        ax1.set_ylabel('Visibility', fontweight='bold')
        ax1.set_title('Performance by Mode', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modes)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.0)
        
        # 2. Wavelength comparison
        ax2 = axes[0, 1]
        if stats['by_wavelength']:
            wl_data = stats['by_wavelength']
            wavelengths = [w['wavelength'] for w in wl_data]
            wl_max_vis = [w['max_vis'] for w in wl_data]
            wl_avg_vis = [w['avg_vis'] for w in wl_data]
            
            x = np.arange(len(wavelengths))
            bars1 = ax2.bar(x - width/2, wl_max_vis, width, label='Max Visibility', color='#2ca02c', alpha=0.8)
            bars2 = ax2.bar(x + width/2, wl_avg_vis, width, label='Avg Visibility', color='#d62728', alpha=0.8)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax2.set_xlabel('Wavelength', fontweight='bold')
            ax2.set_ylabel('Visibility', fontweight='bold')
            ax2.set_title('Performance by Wavelength', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(wavelengths)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, 1.0)
        else:
            ax2.text(0.5, 0.5, 'No wavelength data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Performance by Wavelength', fontweight='bold')
        
        # 3. Layer comparison
        ax3 = axes[1, 0]
        if stats['by_layers']:
            layer_data = stats['by_layers']
            layers = [str(l['layers']) for l in layer_data]
            layer_max_vis = [l['max_vis'] for l in layer_data]
            layer_avg_vis = [l['avg_vis'] for l in layer_data]
            
            x = np.arange(len(layers))
            bars1 = ax3.bar(x - width/2, layer_max_vis, width, label='Max Visibility', color='#9467bd', alpha=0.8)
            bars2 = ax3.bar(x + width/2, layer_avg_vis, width, label='Avg Visibility', color='#8c564b', alpha=0.8)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax3.set_xlabel('Number of Layers', fontweight='bold')
            ax3.set_ylabel('Visibility', fontweight='bold')
            ax3.set_title('Performance by Layer Count', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(layers)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_ylim(0, 1.0)
        else:
            ax3.text(0.5, 0.5, 'No layer data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Performance by Layer Count', fontweight='bold')
        
        # 4. Best configuration summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if stats['overall']['best_config']:
            best = stats['overall']['best_config']
            summary_text = f"""
Best Overall Configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mode: {best['mode']}
Layers: {best['layers']}
Wavelength: {best['wavelength']}
Visibility: {best['visibility']:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Statistics:
Max Visibility: {stats['overall']['max_visibility']:.4f}
Avg Visibility: {stats['overall']['avg_visibility']:.4f}

Total Configurations: {len(stats['by_mode']) * len(stats['by_wavelength']) * len(stats['by_layers'])}
            """
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No configuration data', ha='center', va='center', transform=ax4.transAxes)
        
        ax4.set_title('Best Configuration Summary', fontweight='bold')
        
        plt.tight_layout()
        
        # Save statistics visualization
        stats_viz_path = os.path.join(save_dir, 'visibility_statistics_visualization.png')
        plt.savefig(stats_viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Statistics visualization saved: {stats_viz_path}")

    def _export_visibility_data(self, visibility_by_mode, config, num_layer_options, save_dir):
        """Export visibility data to CSV"""
        print("📄 Exporting visibility data to CSV...")
        
        wavelength_labels = [f'{int(wl*1e9)}nm' for wl in config.wavelengths]
        
        # Prepare data for CSV
        csv_data = []
        headers = ['Mode', 'Layers'] + wavelength_labels + ['Max_Visibility', 'Avg_Visibility']
        
        for mode_idx, mode_data in enumerate(visibility_by_mode):
            mode_array = np.array(mode_data)
            
            for layer_idx, layer_num in enumerate(num_layer_options):
                if layer_idx < mode_array.shape[0]:
                    row = [f'Mode_{mode_idx+1}', layer_num]
                    
                    # Add wavelength data
                    layer_data = mode_array[layer_idx, :]
                    row.extend([f'{val:.6f}' for val in layer_data])
                    
                    # Add statistics
                    row.append(f'{np.max(layer_data):.6f}')
                    row.append(f'{np.mean(layer_data):.6f}')
                    
                    csv_data.append(row)
        
        # Save to CSV
        csv_path = os.path.join(save_dir, 'visibility_data.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(csv_data)
        
        print(f"✅ Visibility data exported: {csv_path}")

    def create_training_progress_visualization(self, results, save_path=None):
        """Create training progress visualization"""
        print("📈 Creating training progress visualization...")
        
        if 'train_losses' not in results or 'val_losses' not in results:
            print("⚠ No training loss data available")
            return
        
        train_losses = results['train_losses']
        val_losses = results['val_losses']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Loss curves
        ax1 = axes[0, 0]
        epochs = range(1, len(train_losses) + 1)
        
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Loss difference
        ax2 = axes[0, 1]
        loss_diff = np.array(val_losses) - np.array(train_losses)
        ax2.plot(epochs, loss_diff, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Validation - Training Loss', fontweight='bold')
        ax2.set_title('Overfitting Monitor', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning rate (if available)
        ax3 = axes[1, 0]
        if 'learning_rates' in results:
            learning_rates = results['learning_rates']
            ax3.plot(epochs, learning_rates, 'm-', linewidth=2)
            ax3.set_xlabel('Epoch', fontweight='bold')
            ax3.set_ylabel('Learning Rate', fontweight='bold')
            ax3.set_title('Learning Rate Schedule', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        else:
            ax3.text(0.5, 0.5, 'Learning rate data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Learning Rate Schedule', fontweight='bold')
        
        # 4. Training summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        final_train_loss = train_losses[-1] if train_losses else 0
        final_val_loss = val_losses[-1] if val_losses else 0
        min_val_loss = min(val_losses) if val_losses else 0
        min_val_epoch = val_losses.index(min_val_loss) + 1 if val_losses else 0
        
        summary_text = f"""
Training Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Epochs: {len(train_losses)}
Final Training Loss: {final_train_loss:.6f}
Final Validation Loss: {final_val_loss:.6f}

Best Validation Loss: {min_val_loss:.6f}
Best Epoch: {min_val_epoch}

Overfitting Check:
Final Gap: {final_val_loss - final_train_loss:.6f}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        ax4.set_title('Training Summary', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Training progress visualization saved: {save_path}")
        
        plt.show()

    def create_weights_analysis(self, results, config, save_path=None):
        """Create weights analysis visualization"""
        print("🔍 Creating weights analysis...")
        
        if 'weights_pred' not in results:
            print("⚠ No weights prediction data available")
            return
        
        weights_pred = results['weights_pred']
        
        # Analyze weights structure
        print(f"Weights prediction shape: {[w.shape if hasattr(w, 'shape') else len(w) for w in weights_pred]}")
        
        # Create visualization based on data structure
        num_layers = len(weights_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Weights distribution by layer
        ax1 = axes[0, 0]
        
        layer_means = []
        layer_stds = []
        layer_labels = []
        
        for layer_idx, layer_weights in enumerate(weights_pred):
            if hasattr(layer_weights, 'flatten'):
                flat_weights = layer_weights.flatten()
            elif isinstance(layer_weights, (list, tuple)):
                flat_weights = np.array(layer_weights).flatten()
            else:
                flat_weights = np.array([layer_weights]).flatten()
            
            layer_means.append(np.mean(flat_weights))
            layer_stds.append(np.std(flat_weights))
            layer_labels.append(f'Layer {layer_idx+1}')
        
        x = np.arange(len(layer_labels))
        bars = ax1.bar(x, layer_means, yerr=layer_stds, capsize=5, 
                      color='skyblue', alpha=0.8, edgecolor='navy')
        
        # Add value labels
        for bar, mean, std in zip(bars, layer_means, layer_stds):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Layer', fontweight='bold')
        ax1.set_ylabel('Weight Value', fontweight='bold')
        ax1.set_title('Weight Distribution by Layer', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_labels)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Weights heatmap (if 2D structure available)
        ax2 = axes[0, 1]
        
        try:
            # Try to create a heatmap from weights
            if len(weights_pred) > 1 and hasattr(weights_pred[0], '__len__'):
                # Create weight matrix
                max_len = max(len(w) if hasattr(w, '__len__') else 1 for w in weights_pred)
                weight_matrix = np.zeros((len(weights_pred), max_len))
                
                for i, layer_weights in enumerate(weights_pred):
                    if hasattr(layer_weights, '__len__'):
                        layer_array = np.array(layer_weights).flatten()
                        weight_matrix[i, :len(layer_array)] = layer_array[:max_len]
                    else:
                        weight_matrix[i, 0] = layer_weights
                
                im = ax2.imshow(weight_matrix, cmap='RdBu_r', aspect='auto')
                ax2.set_xlabel('Weight Index', fontweight='bold')
                ax2.set_ylabel('Layer', fontweight='bold')
                ax2.set_title('Weights Heatmap', fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Weight Value', fontweight='bold')
                
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for heatmap', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Weights Heatmap', fontweight='bold')
                
        except Exception as e:
            ax2.text(0.5, 0.5, f'Heatmap error: {str(e)}', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Weights Heatmap', fontweight='bold')
        
        # 3. Weight evolution (if multiple predictions available)
        ax3 = axes[1, 0]
        
        # This would need training history data
        ax3.text(0.5, 0.5, 'Weight evolution requires training history', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Weight Evolution', fontweight='bold')
        
        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate overall statistics
        all_weights = []
        for layer_weights in weights_pred:
            if hasattr(layer_weights, 'flatten'):
                all_weights.extend(layer_weights.flatten())
            elif isinstance(layer_weights, (list, tuple)):
                all_weights.extend(np.array(layer_weights).flatten())
            else:
                all_weights.append(layer_weights)
        
        all_weights = np.array(all_weights)
        
        stats_text = f"""
Weights Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Parameters: {len(all_weights)}
Mean: {np.mean(all_weights):.6f}
Std: {np.std(all_weights):.6f}
Min: {np.min(all_weights):.6f}
Max: {np.max(all_weights):.6f}

Layers: {len(weights_pred)}
Non-zero params: {np.count_nonzero(all_weights)}
Sparsity: {(1 - np.count_nonzero(all_weights)/len(all_weights))*100:.2f}%
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
        
        ax4.set_title('Weights Statistics', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Weights analysis saved: {save_path}")
        
        plt.show()

    def create_complete_analysis_report(self, results, config, num_layer_options, save_dir):
        """Create complete analysis report with all visualizations"""
        print("📊 Creating complete analysis report...")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Organize visibility data
        if 'visibility' in results:
            visibility_by_mode = self.organize_visibility_by_mode(results, config, num_layer_options)
            
            # Create comprehensive visibility report
            self.create_comprehensive_visibility_report(visibility_by_mode, config, num_layer_options, save_dir)
        
        # 2. Training progress visualization
        if 'train_losses' in results:
            training_path = os.path.join(save_dir, 'training_progress.png')
            self.create_training_progress_visualization(results, training_path)
        
        # 3. Weights analysis
        if 'weights_pred' in results:
            weights_path = os.path.join(save_dir, 'weights_analysis.png')
            self.create_weights_analysis(results, config, weights_path)
        
        # 4. Create summary report
        self._create_summary_report(results, config, num_layer_options, save_dir)
        
        print("✅ Complete analysis report created")

    def _create_summary_report(self, results, config, num_layer_options, save_dir):
        """Create summary report"""
        print("📋 Creating summary report...")
        
        # Create summary text
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'num_modes': config.num_modes,
                'wavelengths': [f'{int(wl*1e9)}nm' for wl in config.wavelengths],
                'layer_options': num_layer_options,
                'device': str(config.device),
                'save_dir': config.save_dir
            },
            'results_summary': {},
            'files_created': []
        }
        
        # Add results summary
        if 'visibility' in results:
            vis_data = results['visibility']
            if vis_data:
                all_vis = []
                for layer_vis in vis_data:
                    if isinstance(layer_vis, (list, tuple)):
                        all_vis.extend(layer_vis)
                    else:
                        all_vis.append(layer_vis)
                
                summary['results_summary']['visibility'] = {
                    'max': float(np.max(all_vis)) if all_vis else 0,
                    'mean': float(np.mean(all_vis)) if all_vis else 0,
                    'min': float(np.min(all_vis)) if all_vis else 0
                }
        
        if 'train_losses' in results:
            summary['results_summary']['training'] = {
                'epochs': len(results['train_losses']),
                'final_train_loss': float(results['train_losses'][-1]) if results['train_losses'] else 0,
                'final_val_loss': float(results['val_losses'][-1]) if results['val_losses'] else 0
            }
        
        # List created files
        for file in os.listdir(save_dir):
            if file.endswith(('.png', '.csv', '.json')):
                summary['files_created'].append(file)
        
        # Save summary
        summary_path = os.path.join(save_dir, 'analysis_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Summary report saved: {summary_path}")
        print(f"📁 Total files created: {len(summary['files_created'])}")

        
