from matplotlib import pyplot as plt
import numpy as np
import os
import csv
import json
import glob
import re
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
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

    # ==================== 缺失的辅助方法 ====================
    
    def _extract_file_info(self, filename):
        """解析文件名提取配置信息"""
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
                # 将1-based转换为0-based
                mode_idx = int(mode_match.group(1)) - 1
                break
        
        if mode_idx is None:
            return None
        
        # 检查转换后的模式索引范围
        if mode_idx < 0 or mode_idx >= 3:
            print(f"  ⚠ 转换后模式索引超出范围: {mode_idx}")
        
        # 提取波长
        wl_match = re.search(r'(\d+)nm', filename)
        if not wl_match:
            return None
        
        wl_nm = int(wl_match.group(1))
        
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
                break
        
        if layers is None:
            return None
        
        result = {
            'mode': mode_idx,      # 0-based
            'wavelength': wl_nm,
            'layers': layers
        }
        
        return result

    def _reorganize_visibility_by_mode(self, visibility_data, config, num_layer_options):
        """重新按模式组织 visibility 数据"""
        
        organized_data = []
        
        print(f"\n🔄 重新组织数据:")
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
                    key = (layers, mode_idx, wl_nm)  # 使用0-based模式索引
                    
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

    # ==================== 改进的按波长分离的Cross Matrix方法 ====================
    
    def _extract_wavelengths_improved(self, cross_matrix_data):
        """改进的波长提取方法 - 支持多种格式"""
        import re
        wavelengths = set()
        
        for key in cross_matrix_data.keys():
            # 支持多种波长模式
            patterns = [
                r'(\d+)nm',      # 532nm
                r'_(\d+)nm',     # _532nm  
                r'nm(\d+)',      # nm532
                r'wl(\d+)',      # wl532
                r'(\d+)_nm'      # 532_nm
            ]
            
            for pattern in patterns:
                match = re.search(pattern, key, re.IGNORECASE)
                if match:
                    wavelengths.add(int(match.group(1)))
                    break
        
        return sorted(list(wavelengths))

    def _match_config_with_wavelength_improved(self, key, layers, mode_idx, wavelength):
        """改进的配置匹配方法 - 支持多种键名格式"""
        # 层数匹配模式
        layer_patterns = [f'L{layers}_', f'layers{layers}', f'{layers}layers', f'{layers}L']
        
        # 模式匹配模式 (支持1-based和0-based)
        mode_patterns = [f'_M{mode_idx+1}_', f'_mode{mode_idx+1}', f'mode{mode_idx+1}', 
                        f'M{mode_idx+1}', f'_M{mode_idx}_', f'mode{mode_idx}']
        
        # 波长匹配模式
        wavelength_patterns = [f'_{wavelength}nm', f'{wavelength}nm', f'wl{wavelength}', f'nm{wavelength}']
        
        layer_match = any(pattern in key for pattern in layer_patterns)
        mode_match = any(pattern in key for pattern in mode_patterns)
        wavelength_match = any(pattern in key for pattern in wavelength_patterns)
        
        return layer_match and mode_match and wavelength_match

    def _create_single_wavelength_chart_improved(self, ax, cross_matrix_data, config, 
                                               num_layer_options, wavelength):
        """
        为单个波长创建改进的Cross Matrix柱状图 - 匹配您的图片样式
        """
        import numpy as np
        
        modes = [0, 1, 2]  # Mode 1, 2, 3 (0-based indexing)
        # 使用与您图片相同的颜色
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 红色、青色、蓝色
        mode_labels = ['Mode 1', 'Mode 2', 'Mode 3']
        
        x = np.arange(len(num_layer_options))
        width = 0.25  # 柱状图宽度
        
        max_value = 0  # 用于设置Y轴范围
        
        for mode_idx in modes:
            focus_concentrations = []
            
            for layers in num_layer_options:
                mode_layer_values = []
                
                # 只获取指定波长的数据
                for key, data in cross_matrix_data.items():
                    if self._match_config_with_wavelength_improved(key, layers, mode_idx, wavelength):
                        if 'focus_concentration' in data:
                            mode_layer_values.append(data['focus_concentration'])
                
                # 计算该模式和层数组合的平均值
                avg_focus = np.mean(mode_layer_values) if mode_layer_values else 0
                focus_concentrations.append(avg_focus)
                max_value = max(max_value, avg_focus)
            
            # 绘制柱状图
            positions = x + mode_idx * width
            bars = ax.bar(positions, focus_concentrations, width, 
                         label=mode_labels[mode_idx], color=colors[mode_idx], 
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 添加数值标签 - 匹配您图片的样式
            for bar, value in zip(bars, focus_concentrations):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_value * 0.02,
                           f'{value:.3f}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
        
        # 设置图表属性 - 匹配您的图片样式
        ax.set_title(f'Cross Matrix Performance - {wavelength}nm', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Focus Concentration', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(num_layer_options)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 设置合适的Y轴范围 - 匹配您的图片
        if max_value > 0:
            ax.set_ylim(0, max_value * 1.2)
        else:
            ax.set_ylim(0, 1.0)
        
        # 设置坐标轴样式
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    def create_cross_matrix_by_wavelength_improved(self, cross_matrix_data, config, num_layer_options, 
                                                 save_path=None, title_suffix=""):
        """
        创建改进的按波长分离的Cross Matrix可视化 - 匹配您的图片样式
        
        参数:
        - cross_matrix_data: 包含focus_concentration数据的字典
        - config: 配置对象，包含num_modes, wavelengths等
        - num_layer_options: 层数选项列表，如[1,2,3,4,5]
        - save_path: 保存路径（可选）
        - title_suffix: 标题后缀（可选）
        
        返回:
        - matplotlib figure对象
        """
        if not cross_matrix_data:
            print("❌ 没有Cross Matrix数据")
            return None
        
        print("🎨 创建改进的按波长分离Cross Matrix可视化...")
        
        # 1. 提取所有可用波长
        wavelengths = self._extract_wavelengths_improved(cross_matrix_data)
        
        if not wavelengths:
            print("警告：未找到波长信息")
            return None
        
        print(f"检测到波长: {wavelengths}nm")
        
        # 2. 计算子图布局 - 水平排列
        n_wavelengths = len(wavelengths)
        if n_wavelengths == 1:
            rows, cols = 1, 1
            figsize = (10, 8)
        elif n_wavelengths == 2:
            rows, cols = 1, 2
            figsize = (20, 8)
        elif n_wavelengths == 3:
            rows, cols = 1, 3
            figsize = (24, 10)  
        else:
            rows = (n_wavelengths + 2) // 3
            cols = 3
            figsize = (24, 8 * rows)
        
        # 3. 创建图表
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # 处理单个子图的情况
        if n_wavelengths == 1:
            axes = [axes]
        elif rows == 1 and cols > 1:
            axes = list(axes)
        elif rows > 1 and cols == 1:
            axes = list(axes)
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # 4. 为每个波长生成图表
        for idx, wavelength in enumerate(wavelengths):
            ax = axes[idx]
            self._create_single_wavelength_chart_improved(ax, cross_matrix_data, config, 
                                                        num_layer_options, wavelength)
        
        # 5. 隐藏多余的子图
        for idx in range(n_wavelengths, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # 为总标题留出空间
        
        # 7. 保存图片

        if save_path is None:
            save_path = os.path.join(config.save_dir if hasattr(config, 'save_dir') else '.', 
                                   f'cross_matrix_by_wavelength{title_suffix}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            
            print(f"✅ 按波长分离的Cross Matrix图表已保存: {save_path}")
            return fig

    # ==================== 保留原有的旧方法（向后兼容）====================
    
    def _extract_wavelengths(self, cross_matrix_data):
        """从数据键中提取波长信息 - 旧版本，保持兼容性"""
        return self._extract_wavelengths_improved(cross_matrix_data)

    def _match_config_with_wavelength(self, key, layers, mode_idx, wavelength):
        """匹配特定波长、层数和模式的配置 - 旧版本，保持兼容性"""
        return self._match_config_with_wavelength_improved(key, layers, mode_idx, wavelength)

    def _create_single_wavelength_chart(self, ax, cross_matrix_data, config, 
                                      num_layer_options, wavelength):
        """为单个波长创建Cross Matrix柱状图 - 旧版本，保持兼容性"""
        return self._create_single_wavelength_chart_improved(ax, cross_matrix_data, config, 
                                                           num_layer_options, wavelength)

    def _create_cross_matrix_charts_by_wavelength(self, cross_matrix_data, config, num_layer_options):
        """为每个波长创建独立的Cross Matrix图表 - 旧版本，保持兼容性"""
        return self.create_cross_matrix_by_wavelength_improved(cross_matrix_data, config, num_layer_options)

    # ==================== 双维度可见度计算方法 ====================
    
    def calculate_cross_matrix_intensity(self, field_data, grid_size=8):
        """
        维度1：计算Cross Matrix - 每个区域内的汇聚强度
        """
        # 基础处理
        if np.iscomplexobj(field_data):
            intensity = np.abs(field_data)**2
        else:
            intensity = np.abs(field_data)**2
        
        # 处理多维数据
        if intensity.ndim > 2:
            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
        
        if intensity.ndim != 2:
            return {'cross_matrix': np.zeros((grid_size, grid_size)), 'max_intensity': 0, 'focus_region': (0, 0)}
        
        H, W = intensity.shape
        
        # 归一化强度
        max_intensity = np.max(intensity)
        if max_intensity <= 0:
            return {'cross_matrix': np.zeros((grid_size, grid_size)), 'max_intensity': 0, 'focus_region': (0, 0)}
        
        intensity_norm = intensity / max_intensity
        
        # 创建网格
        cross_matrix = np.zeros((grid_size, grid_size))
        
        # 计算每个网格区域的尺寸
        region_h = H // grid_size
        region_w = W // grid_size
        
        # 遍历每个网格区域
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算区域边界
                start_h = i * region_h
                end_h = min((i + 1) * region_h, H)
                start_w = j * region_w
                end_w = min((j + 1) * region_w, W)
                
                # 提取区域
                region = intensity_norm[start_h:end_h, start_w:end_w]
                
                # 计算区域内的汇聚强度
                region_total = np.sum(region)
                region_max = np.max(region)
                region_mean = np.mean(region)
                
                # 汇聚强度 = 总强度 × 峰值强度 × 集中度
                concentration_factor = region_max / (region_mean + 1e-10)
                cross_matrix[i, j] = region_total * region_max * min(concentration_factor / 5.0, 1.0)
        
        # 找到最强汇聚区域
        max_region_idx = np.unravel_index(np.argmax(cross_matrix), cross_matrix.shape)
        
        # 计算整体汇聚强度指标
        total_cross_intensity = np.sum(cross_matrix)
        max_cross_intensity = np.max(cross_matrix)
        
        # 汇聚集中度：最强区域占总强度的比例
        focus_concentration = max_cross_intensity / (total_cross_intensity + 1e-10)
        
        return {
            'cross_matrix': cross_matrix,
            'max_intensity': max_cross_intensity,
            'total_intensity': total_cross_intensity,
            'focus_concentration': focus_concentration,
            'focus_region': max_region_idx,
            'grid_size': grid_size
        }

    def calculate_signal_noise_ratio(self, field_data, target_region_ratio=0.25):
        """
        维度2：计算目标区域和背景区域的信噪比
        """
        # 基础处理
        if np.iscomplexobj(field_data):
            intensity = np.abs(field_data)**2
        else:
            intensity = np.abs(field_data)**2
        
        if intensity.ndim > 2:
            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
        
        if intensity.ndim != 2:
            return {'snr_db': 0, 'signal_power': 0, 'noise_power': 0, 'contrast_ratio': 1}
        
        H, W = intensity.shape
        
        # 归一化
        max_intensity = np.max(intensity)
        if max_intensity <= 0:
            return {'snr_db': 0, 'signal_power': 0, 'noise_power': 0, 'contrast_ratio': 1}
        
        intensity_norm = intensity / max_intensity
        
        # 基于峰值位置的目标区域定义
        peak_pos = np.unravel_index(np.argmax(intensity), intensity.shape)
        peak_y, peak_x = peak_pos
        
        # 计算目标区域半径
        target_area = H * W * target_region_ratio
        target_radius = int(np.sqrt(target_area / np.pi))
        target_radius = max(target_radius, min(H, W) // 8)
        
        # 创建目标区域掩码
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        target_mask = ((y_grid - peak_y)**2 + (x_grid - peak_x)**2) <= target_radius**2
        
        # 基于强度阈值的自适应目标区域
        threshold = np.percentile(intensity_norm.flatten(), 90)  # 前10%的强度作为信号
        adaptive_target_mask = intensity_norm >= threshold
        
        # 选择更合适的目标区域
        if np.sum(adaptive_target_mask) > 0.05 * H * W:
            final_target_mask = adaptive_target_mask
        else:
            final_target_mask = target_mask
        
        # 背景区域 = 非目标区域
        background_mask = ~final_target_mask
        
        # 计算信号和噪声功率
        signal_region = intensity_norm[final_target_mask]
        noise_region = intensity_norm[background_mask]
        
        if len(signal_region) == 0 or len(noise_region) == 0:
            return {'snr_db': 0, 'signal_power': 0, 'noise_power': 0, 'contrast_ratio': 1}
        
        # 信号功率：目标区域的平均强度
        signal_power = np.mean(signal_region)
        
        # 噪声功率：背景区域的平均强度
        noise_power = np.mean(noise_region)
        
        # 信噪比计算
        snr_linear = signal_power / (noise_power + 1e-10)
        snr_db = 10 * np.log10(snr_linear + 1e-10)
        
        # 对比度比率
        contrast_ratio = signal_power / (noise_power + 1e-10)
        
        # 峰值信噪比
        peak_signal = np.max(signal_region)
        peak_snr_linear = peak_signal / (noise_power + 1e-10)
        peak_snr_db = 10 * np.log10(peak_snr_linear + 1e-10)
        
        return {
            'snr_db': snr_db,
            'peak_snr_db': peak_snr_db,
            'signal_power': signal_power,
            'noise_power': noise_power,
            'contrast_ratio': contrast_ratio,
            'signal_region_size': len(signal_region),
            'background_region_size': len(noise_region),
            'target_mask': final_target_mask,
            'background_mask': background_mask
        }

    def calculate_dual_dimension_visibility(self, field_data, grid_size=8, target_region_ratio=0.25):
        """
        计算双维度可见度：Cross Matrix + SNR
        """
        print("🔍 计算双维度可见度...")
        
        # 维度1: Cross Matrix 汇聚强度
        cross_matrix_result = self.calculate_cross_matrix_intensity(field_data, grid_size)
        
        # 维度2: 信噪比
        snr_result = self.calculate_signal_noise_ratio(field_data, target_region_ratio)
        
        # 综合评分
        # 维度1评分：基于汇聚集中度
        cross_score = min(cross_matrix_result['focus_concentration'], 1.0)
        
        # 维度2评分：基于信噪比（dB转换为0-1分数）
        snr_db = snr_result['snr_db']
        snr_score = min(max(snr_db / 20.0, 0), 1.0)  # 20dB对应满分
        
        # 综合可见度 = 两个维度的加权平均
        comprehensive_visibility = cross_score * 0.5 + snr_score * 0.5
        
        return {
            'cross_matrix': cross_matrix_result,
            'snr': snr_result,
            'scores': {
                'cross_score': cross_score,
                'snr_score': snr_score,
                'comprehensive': comprehensive_visibility
            },
            'summary': {
                'focus_concentration': cross_matrix_result['focus_concentration'],
                'snr_db': snr_result['snr_db'],
                'contrast_ratio': snr_result['contrast_ratio'],
                'comprehensive_visibility': comprehensive_visibility
            }
        }


class SeparatedDimensionVisualizer(Visualizer):
    """分离的双维度可视化器"""
    
    def __init__(self, config):
        super().__init__(config)
    
    # ==================== Cross Matrix 独立可视化 ====================
    
    def create_cross_matrix_visualization(self, cross_matrix_data, config, num_layer_options, 
                                                save_path=None, title_suffix=""):
        """
        创建分离的Cross Matrix可视化 - 每个图表单独保存
        """
        if not cross_matrix_data:
            print("❌ 没有Cross Matrix数据")
            return None
        
        print("🎨 创建分离的Cross Matrix可视化...")
        
        # 确定保存路径的基础目录
        if save_path is None:
            base_dir = config.save_dir
            base_name = f'cross_matrix_analysis{title_suffix}'
        else:
            base_dir = os.path.dirname(save_path)
            base_name = os.path.splitext(os.path.basename(save_path))[0]
        
        saved_files = []
        
        # 1. 创建按波长分离的Cross Matrix性能柱状图
        print("   📊 创建按波长分离的Cross Matrix性能柱状图...")
        fig1 = self._create_cross_matrix_charts_by_wavelength(cross_matrix_data, config, num_layer_options)
        
        if fig1:
            wavelength_chart_path = os.path.join(base_dir, f'{base_name}_by_wavelength.png')
            fig1.savefig(wavelength_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            saved_files.append(wavelength_chart_path)
            print(f"   ✅ 按波长分离的柱状图已保存: {wavelength_chart_path}")
        
        # 2. 创建聚焦集中度热图
        print("   🔥 创建聚焦集中度热图...")
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
        fig2.suptitle(f'Focus Concentration Heatmap{title_suffix}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        self._create_focus_concentration_heatmap(ax2, cross_matrix_data, config, num_layer_options)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        heatmap_path = os.path.join(base_dir, f'{base_name}_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        saved_files.append(heatmap_path)
        print(f"   ✅ 热图已保存: {heatmap_path}")
        
        print(f"✅ Cross Matrix分离分析完成，共保存 {len(saved_files)} 个文件")
        return saved_files
    
    def _create_cross_matrix_bar_chart(self, ax, cross_matrix_data, config, num_layer_options):
        """Cross Matrix性能柱状图"""
        # 按模式和层数组织数据
        modes = list(range(config.num_modes))
        bar_width = 0.25
        x_positions = np.arange(len(num_layer_options))
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for mode_idx in modes:
            focus_concentrations = []
            for layers in num_layer_options:
                # 计算该层数和模式下的平均聚焦集中度
                mode_layer_values = []
                for key, data in cross_matrix_data.items():
                    if self._match_config(key, layers, mode_idx):
                        if 'focus_concentration' in data:
                            mode_layer_values.append(data['focus_concentration'])
                
                avg_focus = np.mean(mode_layer_values) if mode_layer_values else 0
                focus_concentrations.append(avg_focus)
            
            # 创建柱状图
            bars = ax.bar(x_positions + mode_idx * bar_width, focus_concentrations,
                        bar_width, label=f'Mode {mode_idx+1}',
                        color=colors[mode_idx], alpha=0.8, edgecolor='black')
            
            # 添加数值标注
            for bar, value in zip(bars, focus_concentrations):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Focus Concentration', fontsize=12, fontweight='bold')
        ax.set_title('Cross Matrix - Focus Concentration by Layers', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels([f'{layers}L' for layers in num_layer_options])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
    
    def _create_focus_concentration_heatmap(self, ax, cross_matrix_data, config, num_layer_options):
        """聚焦集中度热图"""
        # 构建热图数据矩阵
        wavelengths = [int(wl * 1e9) for wl in config.wavelengths]
        
        # 为每个模式创建子热图
        num_modes = config.num_modes
        fig_data = np.zeros((num_modes * len(wavelengths), len(num_layer_options)))
        
        row_labels = []
        for mode_idx in range(num_modes):
            for wl_idx, wl in enumerate(wavelengths):
                row_idx = mode_idx * len(wavelengths) + wl_idx
                row_labels.append(f'M{mode_idx+1}-{wl}nm')
                
                for col_idx, layers in enumerate(num_layer_options):
                    # 查找匹配的数据
                    for key, data in cross_matrix_data.items():
                        if self._match_config_full(key, layers, mode_idx, wl):
                            if 'focus_concentration' in data:
                                fig_data[row_idx, col_idx] = data['focus_concentration']
                            break
        
        # 绘制热图
        im = ax.imshow(fig_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # 添加数值标注
        for i in range(fig_data.shape[0]):
            for j in range(fig_data.shape[1]):
                value = fig_data[i, j]
                color = 'white' if value < 0.5 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                    color=color, fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mode-Wavelength', fontsize=12, fontweight='bold')
        ax.set_title('Focus Concentration Heatmap', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(num_layer_options)))
        ax.set_xticklabels([f'{layers}L' for layers in num_layer_options])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Focus Concentration', fontweight='bold')

    # ==================== SNR 独立可视化 ====================
    def create_snr_only_visualization(self, snr_data, config, num_layer_options, 
                                save_path=None, title_suffix=""):

        if not snr_data:
            print("❌ 没有SNR数据")
            return None

        print("🎨 创建SNR独立可视化...")

        # 创建2x2布局
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'SNR Analysis Dashboard{title_suffix}', 
                    fontsize=18, fontweight='bold', y=0.95)

        # 1. SNR性能柱状图 (左上)
        self._create_snr_performance_bar_chart(axes[0, 0], snr_data, config, num_layer_options)

        # 2. SNR热图 (右上)
        self._create_snr_heatmap(axes[0, 1], snr_data, config, num_layer_options)

        # 3. 信噪比分布直方图 (左下)
        self._create_snr_distribution_histogram(axes[1, 0], snr_data, config)

        # 4. 对比度分析 (右下)
        self._create_contrast_analysis(axes[1, 1], snr_data, config, num_layer_options)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        if save_path is None:
            save_path = os.path.join(config.save_dir, f'snr_analysis{title_suffix}.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"✅ SNR分析已保存: {save_path}")
        return fig

    def _create_snr_performance_bar_chart(self, ax, snr_data, config, num_layer_options):
        """SNR性能柱状图"""
        # 按模式和层数组织数据
        modes = list(range(config.num_modes))
        bar_width = 0.25
        x_positions = np.arange(len(num_layer_options))
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for mode_idx in modes:
            snr_values = []
            for layers in num_layer_options:
                # 计算该层数和模式下的平均SNR
                mode_layer_values = []
                for key, data in snr_data.items():
                    if self._match_config(key, layers, mode_idx):
                        if 'snr_db' in data:
                            # 将dB转换为0-1分数
                            snr_score = max(0, min(1, data['snr_db'] / 20.0))
                            mode_layer_values.append(snr_score)
                
                avg_snr = np.mean(mode_layer_values) if mode_layer_values else 0
                snr_values.append(avg_snr)
            
            # 创建柱状图
            bars = ax.bar(x_positions + mode_idx * bar_width, snr_values,
                        bar_width, label=f'Mode {mode_idx+1}',
                        color=colors[mode_idx], alpha=0.8, edgecolor='black')
            
            # 添加数值标注
            for bar, value in zip(bars, snr_values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel('SNR Score (0-1)', fontsize=12, fontweight='bold')
        ax.set_title('SNR Performance by Layers and Modes', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels([f'{layers}L' for layers in num_layer_options])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
    
    def _create_snr_heatmap(self, ax, snr_data, config, num_layer_options):
        """SNR热图"""
        wavelengths = [int(wl * 1e9) for wl in config.wavelengths]
        num_modes = config.num_modes
        
        # 构建热图数据矩阵
        fig_data = np.zeros((num_modes * len(wavelengths), len(num_layer_options)))
        
        row_labels = []
        for mode_idx in range(num_modes):
            for wl_idx, wl in enumerate(wavelengths):
                row_idx = mode_idx * len(wavelengths) + wl_idx
                row_labels.append(f'M{mode_idx+1}-{wl}nm')
                
                for col_idx, layers in enumerate(num_layer_options):
                    # 查找匹配的数据
                    for key, data in snr_data.items():
                        if self._match_config_full(key, layers, mode_idx, wl):
                            if 'snr_db' in data:
                                # 将dB转换为0-1分数
                                snr_score = max(0, min(1, data['snr_db'] / 20.0))
                                fig_data[row_idx, col_idx] = snr_score
                            break
        
        # 绘制热图
        im = ax.imshow(fig_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # 添加数值标注
        for i in range(fig_data.shape[0]):
            for j in range(fig_data.shape[1]):
                value = fig_data[i, j]
                color = 'white' if value < 0.5 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                       color=color, fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mode-Wavelength', fontsize=12, fontweight='bold')
        ax.set_title('SNR Score Heatmap', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(num_layer_options)))
        ax.set_xticklabels([f'{layers}L' for layers in num_layer_options])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SNR Score', fontweight='bold')
    
    def _create_snr_distribution_histogram(self, ax, snr_data, config):
        """SNR分布直方图"""
        # 收集所有SNR值
        snr_db_values = []
        snr_score_values = []
        
        for key, data in snr_data.items():
            if 'snr_db' in data:
                snr_db_values.append(data['snr_db'])
                snr_score_values.append(max(0, min(1, data['snr_db'] / 20.0)))
        
        if not snr_db_values:
            ax.text(0.5, 0.5, 'No SNR Data Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return
        
        # 创建双轴直方图
        ax2 = ax.twinx()
        
        # SNR dB分布
        n1, bins1, patches1 = ax.hist(snr_db_values, bins=20, alpha=0.7, color='skyblue', 
                                     label='SNR (dB)', edgecolor='black')
        
        # SNR Score分布
        n2, bins2, patches2 = ax2.hist(snr_score_values, bins=20, alpha=0.7, color='lightcoral', 
                                      label='SNR Score (0-1)', edgecolor='black')
        
        # 统计信息
        mean_db = np.mean(snr_db_values)
        std_db = np.std(snr_db_values)
        mean_score = np.mean(snr_score_values)
        
        ax.axvline(mean_db, color='blue', linestyle='--', linewidth=2, 
                  label=f'Mean dB: {mean_db:.2f}')
        ax2.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean Score: {mean_score:.3f}')
        
        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (dB)', fontsize=12, fontweight='bold', color='blue')
        ax2.set_ylabel('Frequency (Score)', fontsize=12, fontweight='bold', color='red')
        ax.set_title(f'SNR Distribution\nMean: {mean_db:.2f}±{std_db:.2f} dB', 
                    fontsize=14, fontweight='bold')
        
        # 图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    def _create_contrast_analysis(self, ax, snr_data, config, num_layer_options):
        """对比度分析"""
        # 收集对比度数据
        contrast_ratios = []
        signal_powers = []
        noise_powers = []
        config_labels = []
        
        for key, data in snr_data.items():
            if all(k in data for k in ['contrast_ratio', 'signal_power', 'noise_power']):
                contrast_ratios.append(data['contrast_ratio'])
                signal_powers.append(data['signal_power'])
                noise_powers.append(data['noise_power'])
                config_labels.append(key)
        
        if not contrast_ratios:
            ax.text(0.5, 0.5, 'No Contrast Data Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return
        
        # 创建散点图：信号功率 vs 对比度比率
        scatter = ax.scatter(signal_powers, contrast_ratios, 
                           c=noise_powers, cmap='viridis', 
                           s=100, alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('Signal Power', fontsize=12, fontweight='bold')
        ax.set_ylabel('Contrast Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Signal Power vs Contrast Ratio', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Noise Power', fontweight='bold')
        
        # 标记最佳点
        if contrast_ratios:
            best_idx = np.argmax(contrast_ratios)
            ax.scatter(signal_powers[best_idx], contrast_ratios[best_idx], 
                      s=200, c='red', marker='*', edgecolors='black', linewidth=2,
                      label='Best Contrast')
            ax.legend()
    
    # ==================== 辅助方法 ====================
    
    def _match_config(self, key, layers, mode_idx):
        """匹配配置（层数和模式）"""
        return (f'layers{layers}' in key or f'L{layers}_' in key) and \
               (f'mode{mode_idx+1}' in key or f'_M{mode_idx+1}_' in key)
    
    def _match_config_full(self, key, layers, mode_idx, wavelength):
        """完全匹配配置（层数、模式、波长）"""
        return self._match_config(key, layers, mode_idx) and \
               f'{wavelength}nm' in key
    
    def _contains_wavelength(self, key, wavelength):
        """检查键是否包含指定波长"""
        return f'{wavelength}nm' in key
    
    # ==================== 批量处理方法 ====================
    
    def calculate_separated_dimensions_from_simulation(self, save_dir, config, num_layer_options):
        """
        从仿真结果分别计算Cross Matrix和SNR数据
        """
        print("🔍 分别计算Cross Matrix和SNR数据...")
        
        # 查找所有仿真结果文件
        result_files = glob.glob(os.path.join(save_dir, "MC_single_*.npy"))
        
        if not result_files:
            print("❌ 未找到仿真结果文件")
            return None, None
        
        print(f"找到 {len(result_files)} 个仿真结果文件")
        
        cross_matrix_data = {}
        snr_data = {}
        
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
                
                # 分别计算两个维度
                cross_result = self.calculate_cross_matrix_intensity(data)
                snr_result = self.calculate_signal_noise_ratio(data)
                
                # 创建友好的键名
                key_str = f"L{layers}_M{mode_idx+1}_{wl_nm}nm"
                
                cross_matrix_data[key_str] = cross_result
                snr_data[key_str] = snr_result
                
                print(f"  {key_str}: Cross={cross_result['focus_concentration']:.3f}, "
                      f"SNR={snr_result['snr_db']:.2f}dB")
                
            except Exception as e:
                print(f"❌ 处理文件 {filename} 时出错: {e}")
                continue
        
        print(f"成功处理 {len(cross_matrix_data)} 个Cross Matrix数据点")
        print(f"成功处理 {len(snr_data)} 个SNR数据点")
        
        return cross_matrix_data, snr_data
    
    def create_separated_analysis_report(self, save_dir, config, num_layer_options):
        """
        创建分离的分析报告
        """
        print("📊 创建分离的双维度分析报告...")
        
        # 计算分离的数据
        cross_matrix_data, snr_data = self.calculate_separated_dimensions_from_simulation(
            save_dir, config, num_layer_options)
        
        if not cross_matrix_data or not snr_data:
            print("❌ 数据计算失败")
            return
        
        # 创建Cross Matrix可视化
        cross_path = os.path.join(save_dir, 'cross_matrix_analysis_separated.png')
        self.create_cross_matrix_visualization(cross_matrix_data, config, num_layer_options, 
                                             cross_path, "_separated")
        
        # 创建SNR可视化
        snr_path = os.path.join(save_dir, 'snr_analysis_separated.png')
        self.create_snr_only_visualization(snr_data, config, num_layer_options, 
                                         snr_path, "_separated")
        
        # 创建综合对比分析
        self.create_comprehensive_comparison(cross_matrix_data, snr_data, config, 
                                           num_layer_options, save_dir, "_separated")
        
        # 保存数据
        self._save_separated_data(cross_matrix_data, snr_data, save_dir)
        
        print("✅ 分离的双维度分析报告创建完成")
        
        return cross_matrix_data, snr_data
    
    def create_comprehensive_comparison(self, cross_matrix_data, snr_data, config, 
                                      num_layer_options, save_dir, title_suffix=""):
        """创建综合对比分析"""
        print("🔍 创建综合对比分析...")
        
        # 创建2x2布局的综合分析
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Comprehensive Cross Matrix vs SNR Analysis{title_suffix}', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. Cross Matrix vs SNR 散点图 (左上)
        self._create_cross_vs_snr_scatter(axes[0, 0], cross_matrix_data, snr_data, config)
        
        # 2. 相关性分析 (右上)
        self._create_correlation_analysis(axes[0, 1], cross_matrix_data, snr_data, config)
        
        # 3. 性能排名对比 (左下)
        self._create_performance_ranking_comparison(axes[1, 0], cross_matrix_data, snr_data, config)
        
        # 4. 最佳配置识别 (右下)
        self._create_optimal_config_identification(axes[1, 1], cross_matrix_data, snr_data, config)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        comparison_path = os.path.join(save_dir, f'comprehensive_comparison{title_suffix}.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ 综合对比分析已保存: {comparison_path}")
        return fig
    
    def _create_cross_vs_snr_scatter(self, ax, cross_matrix_data, snr_data, config):
        """Cross Matrix vs SNR 散点图"""
        # 匹配数据
        cross_values = []
        snr_values = []
        labels = []
        colors = []
        
        # 颜色映射
        cmap = plt.get_cmap('viridis')
        color_indices = np.linspace(0, 1, config.num_modes)
        
        for key in cross_matrix_data.keys():
            if key in snr_data:
                cross_val = cross_matrix_data[key].get('focus_concentration', 0)
                snr_val = max(0, min(1, snr_data[key].get('snr_db', 0) / 20.0))
                
                cross_values.append(cross_val)
                snr_values.append(snr_val)
                labels.append(key)
                
                # 根据模式索引选择颜色
                for mode_idx in range(config.num_modes):
                    if f'M{mode_idx+1}' in key:
                        colors.append(cmap(color_indices[mode_idx]))
                        break
        
        if not cross_values:
            ax.text(0.5, 0.5, 'No Matching Data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
            return
        
        # 创建散点图
        scatter = ax.scatter(cross_values, snr_values, c=colors, s=100, 
                        alpha=0.7, edgecolors='black', linewidth=1)
        
        # 添加对角线
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='Equal Performance')
        
        # 标注最佳点
        if cross_values and snr_values:
            # 综合最佳（欧几里得距离到(1,1)最近）
            distances = [(1-c)**2 + (1-s)**2 for c, s in zip(cross_values, snr_values)]
            best_idx = np.argmin(distances)
            
            ax.scatter(cross_values[best_idx], snr_values[best_idx], 
                    s=300, c='red', marker='*', edgecolors='black', linewidth=2,
                    label=f'Best Overall: {labels[best_idx]}')
        
        ax.set_xlabel('Cross Matrix Focus Concentration', fontsize=12, fontweight='bold')
        ax.set_ylabel('SNR Score (0-1)', fontsize=12, fontweight='bold')
        ax.set_title('Cross Matrix vs SNR Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 添加象限标注
        ax.text(0.8, 0.8, 'High Cross\nHigh SNR', ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        ax.text(0.2, 0.2, 'Low Cross\nLow SNR', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

    def _create_correlation_analysis(self, ax, cross_matrix_data, snr_data, config):
        """相关性分析"""
        # 收集匹配的数据
        cross_values = []
        snr_values = []
        
        for key in cross_matrix_data.keys():
            if key in snr_data:
                cross_val = cross_matrix_data[key].get('focus_concentration', 0)
                snr_val = snr_data[key].get('snr_db', 0)
                
                cross_values.append(cross_val)
                snr_values.append(snr_val)
        
        if len(cross_values) < 2:
            ax.text(0.5, 0.5, 'Insufficient Data for Correlation', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return
        
        # 计算相关系数
        correlation = np.corrcoef(cross_values, snr_values)[0, 1]
        
        # 创建相关性散点图
        ax.scatter(cross_values, snr_values, alpha=0.6, s=80, edgecolors='black')
        
        # 添加趋势线
        z = np.polyfit(cross_values, snr_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(cross_values), max(cross_values), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
               label=f'Trend Line (R={correlation:.3f})')
        
        # 统计信息
        stats_text = f"""
Correlation Analysis:
• Correlation Coefficient: {correlation:.3f}
• Cross Matrix Mean: {np.mean(cross_values):.3f}
• SNR Mean: {np.mean(snr_values):.2f} dB
• Data Points: {len(cross_values)}
        """
        
        ax.text(0.05, 0.95, stats_text.strip(), transform=ax.transAxes, 
               verticalalignment='top', fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        ax.set_xlabel('Cross Matrix Focus Concentration', fontsize=12, fontweight='bold')
        ax.set_ylabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_title('Cross Matrix vs SNR Correlation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _create_performance_ranking_comparison(self, ax, cross_matrix_data, snr_data, config):
        """性能排名对比"""
        # 收集数据并排名
        configs = []
        cross_ranks = []
        snr_ranks = []
        
        # 获取所有配置的性能数据
        performance_data = []
        for key in cross_matrix_data.keys():
            if key in snr_data:
                cross_val = cross_matrix_data[key].get('focus_concentration', 0)
                snr_val = max(0, min(1, snr_data[key].get('snr_db', 0) / 20.0))
                performance_data.append((key, cross_val, snr_val))
        
        if not performance_data:
            ax.text(0.5, 0.5, 'No Data for Ranking', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            return
        
        # 按Cross Matrix排序
        cross_sorted = sorted(performance_data, key=lambda x: x[1], reverse=True)
        # 按SNR排序
        snr_sorted = sorted(performance_data, key=lambda x: x[2], reverse=True)
        
        # 计算排名
        cross_ranking = {item[0]: idx+1 for idx, item in enumerate(cross_sorted)}
        snr_ranking = {item[0]: idx+1 for idx, item in enumerate(snr_sorted)}
        
        # 准备绘图数据
        for key, cross_val, snr_val in performance_data:
            configs.append(key.replace('_', '\n'))  # 换行显示
            cross_ranks.append(cross_ranking[key])
            snr_ranks.append(snr_ranking[key])
        
        # 只显示前10个配置
        if len(configs) > 10:
            configs = configs[:10]
            cross_ranks = cross_ranks[:10]
            snr_ranks = snr_ranks[:10]
        
        x = np.arange(len(configs))
        width = 0.35
        
        # 创建双柱状图
        bars1 = ax.bar(x - width/2, cross_ranks, width, label='Cross Matrix Rank', 
                      color='skyblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, snr_ranks, width, label='SNR Rank', 
                      color='lightcoral', alpha=0.8, edgecolor='black')
        
        # 添加数值标注
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rank (1=Best)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Ranking Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.invert_yaxis()  # 1在顶部
    
    def _create_optimal_config_identification(self, ax, cross_matrix_data, snr_data, config):
        """最佳配置识别"""
        ax.axis('off')
        
        # 找到各种"最佳"配置
        best_configs = {}
        
        # 1. Cross Matrix最佳
        if cross_matrix_data:
            best_cross_key = max(cross_matrix_data.keys(), 
                               key=lambda k: cross_matrix_data[k].get('focus_concentration', 0))
            best_configs['Cross Matrix'] = {
                'config': best_cross_key,
                'value': cross_matrix_data[best_cross_key].get('focus_concentration', 0)
            }
        
        # 2. SNR最佳
        if snr_data:
            best_snr_key = max(snr_data.keys(), 
                             key=lambda k: snr_data[k].get('snr_db', 0))
            best_configs['SNR'] = {
                'config': best_snr_key,
                'value': snr_data[best_snr_key].get('snr_db', 0)
            }
        
        # 3. 综合最佳（加权平均）
        if cross_matrix_data and snr_data:
            combined_scores = {}
            for key in cross_matrix_data.keys():
                if key in snr_data:
                    cross_score = cross_matrix_data[key].get('focus_concentration', 0)
                    snr_score = max(0, min(1, snr_data[key].get('snr_db', 0) / 20.0))
                    combined_scores[key] = 0.5 * cross_score + 0.5 * snr_score
            
            if combined_scores:
                best_combined_key = max(combined_scores.keys(), key=lambda k: combined_scores[k])
                best_configs['Combined'] = {
                    'config': best_combined_key,
                    'value': combined_scores[best_combined_key]
                }
        
        # 创建最佳配置表格
        report_text = "🏆 OPTIMAL CONFIGURATION ANALYSIS\\n"
        report_text += "=" * 50 + "\\n\\n"
        
        for metric, data in best_configs.items():
            config_name = data['config']
            value = data['value']
            
            report_text += f"📊 Best {metric}:\\n"
            report_text += f"   Configuration: {config_name}\\n"
            
            if metric == 'Cross Matrix':
                report_text += f"   Focus Concentration: {value:.4f}\\n"
            elif metric == 'SNR':
                report_text += f"   SNR: {value:.2f} dB\\n"
            elif metric == 'Combined':
                report_text += f"   Combined Score: {value:.4f}\\n"
            
            report_text += "\\n"
        
        # 添加推荐
        if 'Combined' in best_configs:
            report_text += "🎯 RECOMMENDATION:\\n"
            report_text += f"   {best_configs['Combined']['config']}\\n"
            report_text += "   (Best overall performance)\\n\\n"
        
        # 添加性能统计
        if cross_matrix_data and snr_data:
            all_cross = [data.get('focus_concentration', 0) for data in cross_matrix_data.values()]
            all_snr = [data.get('snr_db', 0) for data in snr_data.values()]
            
            report_text += "📈 PERFORMANCE STATISTICS:\\n"
            report_text += f"   Cross Matrix Range: {min(all_cross):.3f} - {max(all_cross):.3f}\\n"
            report_text += f"   SNR Range: {min(all_snr):.1f} - {max(all_snr):.1f} dB\\n"
            report_text += f"   Total Configurations: {len(cross_matrix_data)}\\n"
        
        # 显示报告
        ax.text(0.05, 0.95, report_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11, fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.8", facecolor='lightyellow', alpha=0.9))
        
        ax.set_title('Optimal Configuration Identification', fontsize=16, fontweight='bold', pad=20)
    
    def _save_separated_data(self, cross_matrix_data, snr_data, save_dir):
        """保存分离的数据到文件"""
        print("💾 保存分离的数据...")
        
        # 保存Cross Matrix数据
        cross_file = os.path.join(save_dir, 'cross_matrix_data.json')
        with open(cross_file, 'w') as f:
            # 转换numpy数组为列表以便JSON序列化
            cross_serializable = {}
            for key, data in cross_matrix_data.items():
                cross_serializable[key] = {
                    'focus_concentration': float(data.get('focus_concentration', 0)),
                    'max_intensity': float(data.get('max_intensity', 0)),
                    'total_intensity': float(data.get('total_intensity', 0)),
                    'focus_region': [int(x) for x in data.get('focus_region', [0, 0])],
                    'grid_size': int(data.get('grid_size', 8))
                }
            json.dump(cross_serializable, f, indent=2)
        
        # 保存SNR数据
        snr_file = os.path.join(save_dir, 'snr_data.json')
        with open(snr_file, 'w') as f:
            # 转换numpy数组为列表
            snr_serializable = {}
            for key, data in snr_data.items():
                snr_serializable[key] = {
                    'snr_db': float(data.get('snr_db', 0)),
                    'peak_snr_db': float(data.get('peak_snr_db', 0)),
                    'signal_power': float(data.get('signal_power', 0)),
                    'noise_power': float(data.get('noise_power', 0)),
                    'contrast_ratio': float(data.get('contrast_ratio', 1)),
                    'signal_region_size': int(data.get('signal_region_size', 0)),
                    'background_region_size': int(data.get('background_region_size', 0))
                }
            json.dump(snr_serializable, f, indent=2)
        
        # 创建综合报告CSV
        csv_file = os.path.join(save_dir, 'separated_analysis_summary.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Configuration', 'Focus_Concentration', 'SNR_dB', 
                           'Signal_Power', 'Noise_Power', 'Contrast_Ratio', 'Combined_Score'])
            
            for key in cross_matrix_data.keys():
                if key in snr_data:
                    cross_val = cross_matrix_data[key].get('focus_concentration', 0)
                    snr_val = snr_data[key].get('snr_db', 0)
                    signal_power = snr_data[key].get('signal_power', 0)
                    noise_power = snr_data[key].get('noise_power', 0)
                    contrast_ratio = snr_data[key].get('contrast_ratio', 1)
                    
                    # 计算综合分数
                    cross_score = min(cross_val, 1.0)
                    snr_score = min(max(snr_val / 20.0, 0), 1.0)
                    combined_score = 0.5 * cross_score + 0.5 * snr_score
                    
                    writer.writerow([key, cross_val, snr_val, signal_power, 
                                   noise_power, contrast_ratio, combined_score])
        
        print(f"✅ 数据已保存:")
        print(f"   Cross Matrix: {cross_file}")
        print(f"   SNR: {snr_file}")
        print(f"   Summary CSV: {csv_file}")

    # ==================== 完成SNR计算方法 ====================
    
    def calculate_signal_noise_ratio(self, field_data, target_region_ratio=0.25):
        """
        维度2：计算目标区域和背景区域的信噪比
        """
        # 基础处理
        if np.iscomplexobj(field_data):
            intensity = np.abs(field_data)**2
        else:
            intensity = np.abs(field_data)**2
        
        if intensity.ndim > 2:
            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
        
        if intensity.ndim != 2:
            return {'snr_db': 0, 'signal_power': 0, 'noise_power': 0, 'contrast_ratio': 1}
        
        H, W = intensity.shape
        
        # 归一化
        max_intensity = np.max(intensity)
        if max_intensity <= 0:
            return {'snr_db': 0, 'signal_power': 0, 'noise_power': 0, 'contrast_ratio': 1}
        
        intensity_norm = intensity / max_intensity
        
        # 基于峰值位置的目标区域定义
        peak_pos = np.unravel_index(np.argmax(intensity), intensity.shape)
        peak_y, peak_x = peak_pos
        
        # 计算目标区域半径
        target_area = H * W * target_region_ratio
        target_radius = int(np.sqrt(target_area / np.pi))
        target_radius = max(target_radius, min(H, W) // 8)
        
        # 创建目标区域掩码
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        target_mask = ((y_grid - peak_y)**2 + (x_grid - peak_x)**2) <= target_radius**2
        
        # 基于强度阈值的自适应区域定义
        intensity_threshold = 0.5 * max_intensity
        high_intensity_mask = intensity >= intensity_threshold
        
        # 结合两种掩码
        signal_mask = target_mask & high_intensity_mask
        
        # 背景区域：远离峰值且强度较低的区域
        background_radius = max(target_radius * 2, min(H, W) // 4)
        background_mask = ((y_grid - peak_y)**2 + (x_grid - peak_x)**2) >= background_radius**2
        background_mask = background_mask & (intensity < intensity_threshold * 0.5)
        
        # 计算信号和噪声功率
        if np.sum(signal_mask) > 0:
            signal_power = np.mean(intensity[signal_mask])
        else:
            signal_power = max_intensity
        
        if np.sum(background_mask) > 0:
            noise_power = np.mean(intensity[background_mask])
        else:
            noise_power = np.mean(intensity) * 0.1
        
        # 避免除零
        noise_power = max(noise_power, max_intensity * 1e-6)
        
        # 计算SNR
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        # 峰值SNR
        peak_snr_linear = max_intensity / noise_power
        peak_snr_db = 10 * np.log10(peak_snr_linear)
        
        # 对比度比率
        contrast_ratio = signal_power / noise_power
        
        return {
            'snr_db': snr_db,
            'peak_snr_db': peak_snr_db,
            'signal_power': signal_power,
            'noise_power': noise_power,
            'contrast_ratio': contrast_ratio,
            'signal_region_size': int(np.sum(signal_mask)),
            'background_region_size': int(np.sum(background_mask))
        }

