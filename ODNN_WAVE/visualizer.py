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

    def calculate_dual_visibility_from_simulation_results(self, save_dir, config, num_layer_options):
        """
        从传播仿真结果计算双维度可见度
        """
        print("🔍 从传播仿真结果计算双维度可见度...")
        
        # 查找所有仿真结果文件
        result_files = glob.glob(os.path.join(save_dir, "MC_single_*.npy"))
        
        if not result_files:
            print("❌ 未找到仿真结果文件")
            return None
        
        print(f"找到 {len(result_files)} 个仿真结果文件")
        
        # 组织数据结构
        dual_visibility_data = {}
        
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
                
                # 计算双维度可见度
                dual_result = self.calculate_dual_dimension_visibility(data)
                
                # 存储数据
                key = (layers, mode_idx, wl_nm)
                dual_visibility_data[key] = dual_result
                
                print(f"  {layers}层, 模式{mode_idx}, {wl_nm}nm: "
                      f"Cross={dual_result['scores']['cross_score']:.3f}, "
                      f"SNR={dual_result['scores']['snr_score']:.3f}, "
                      f"综合={dual_result['scores']['comprehensive']:.3f}")
                
            except Exception as e:
                print(f"❌ 处理文件 {filename} 时出错: {e}")
                continue
        
        print(f"成功处理 {len(dual_visibility_data)} 个数据点")
        
        return dual_visibility_data

    def calculate_visibility_from_simulation_results(self, save_dir, config, num_layer_options):
        """
        修改版：使用双维度可见度计算
        """
        print("🔍 从传播仿真结果计算双维度可见度...")
        
        # 首先计算双维度数据
        dual_data = self.calculate_dual_visibility_from_simulation_results(save_dir, config, num_layer_options)
        
        if not dual_data:
            return None
        
        # 提取综合可见度用于兼容现有接口
        visibility_data = {}
        for key, result in dual_data.items():
            visibility_data[key] = result['scores']['comprehensive']
        
        # 按模式重新组织数据
        organized_data = self._reorganize_visibility_by_mode(visibility_data, config, num_layer_options)
        
        # 同时保存双维度数据供详细分析使用
        self._save_dual_dimension_data(dual_data, save_dir)
        
        return organized_data

    def _save_dual_dimension_data(self, dual_data, save_dir):
        """
        保存双维度可见度数据到JSON文件
        直接转换numpy类型，无需自定义编码器
        """
        print("💾 保存双维度可见度数据...")
        
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        # 转换数据为JSON可序列化格式
        dual_data_serializable = {}
        
        for key, result in dual_data.items():
            layers, mode_idx, wl_nm = key
            key_str = f"{int(layers)}L_mode{int(mode_idx)+1}_{int(wl_nm)}nm"
            
            # 使用递归函数转换整个结果字典
            dual_data_serializable[key_str] = convert_numpy_types(result)
        
        # 保存到文件
        data_path = os.path.join(save_dir, 'dual_dimension_visibility_data.json')
        try:
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(dual_data_serializable, f, indent=2, ensure_ascii=False)
            print(f"✅ 双维度数据已保存: {data_path}")
        except Exception as e:
            print(f"❌ 保存双维度数据失败: {e}")

    # ==================== 其他必要的方法 ====================
    
    def organize_visibility_by_mode(self, results, config, num_layer_options):
        """
        组织可见度数据按模式
        """
        print("📊 组织可见度数据按模式...")
        
        num_modes = config.num_modes
        num_wavelengths = len(config.wavelengths)
        vis_data = results['visibility']
        
        print(f"配置: {num_modes} 模式, {num_wavelengths} 波长, {len(num_layer_options)} 层选项")
        print(f"原始数据结构: {[len(layer_data) for layer_data in vis_data]}")
        
        # 检查数据结构
        if len(vis_data[0]) == num_modes:
            print("✅ 检测到: 单波长数据 (450nm), 其他设为0")
            return self._organize_single_wavelength_data(vis_data, num_modes, num_wavelengths, num_layer_options)
        elif len(vis_data[0]) == num_modes * num_wavelengths:
            print("✅ 检测到: 完整多波长数据")
            return self._organize_multi_wavelength_data(vis_data, num_modes, num_wavelengths, num_layer_options)
        else:
            print("⚠ 数据结构不匹配，从weights_pred重新计算")
            return self._recalculate_from_weights(results, config, num_layer_options)
    
    def _organize_single_wavelength_data(self, vis_data, num_modes, num_wavelengths, num_layer_options):
        """处理单波长数据"""
        visibility_by_mode = []
        
        for mode_idx in range(num_modes):
            mode_visibility = []
            
            for layer_idx in range(len(num_layer_options)):
                wavelength_vis = []
                
                # 第一个波长 (450nm) - 实际数据
                if layer_idx < len(vis_data) and mode_idx < len(vis_data[layer_idx]):
                    actual_vis = vis_data[layer_idx][mode_idx]
                else:
                    actual_vis = 0.0
                wavelength_vis.append(actual_vis)
                
                # 其他波长 - 设为0或估计
                for wl_idx in range(1, num_wavelengths):
                    wavelength_vis.append(0.0)
                
                mode_visibility.append(wavelength_vis)
            
            visibility_by_mode.append(mode_visibility)
        
        return visibility_by_mode
    
    def _organize_multi_wavelength_data(self, vis_data, num_modes, num_wavelengths, num_layer_options):
        """处理完整多波长数据"""
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
        """从weights_pred重新计算可见度"""
        if 'weights_pred' not in results:
            print("❌ 没有weights_pred数据可用于重新计算")
            return self._create_empty_visibility_data(config.num_modes, len(config.wavelengths), len(num_layer_options))
        
        print("🔄 从weights_pred重新计算可见度...")
        return self._create_empty_visibility_data(config.num_modes, len(config.wavelengths), len(num_layer_options))
    
    def _create_empty_visibility_data(self, num_modes, num_wavelengths, num_layers):
        """创建空的可见度数据结构"""
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
        创建详细的可见度分析图表
        """
        print("🎨 创建详细可见度分析...")
        
        num_modes = len(visibility_by_mode)
        wavelength_labels = [f'{int(wl*1e9)}nm' for wl in config.wavelengths]
        
        # 创建图表
        fig = plt.figure(figsize=(15, 6 * num_modes))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for mode_idx in range(num_modes):
            mode_data = np.array(visibility_by_mode[mode_idx])
            
            # 找到最佳性能
            best_vis = np.max(mode_data)
            best_pos = np.unravel_index(np.argmax(mode_data), mode_data.shape)
            best_layer = num_layer_options[best_pos[0]]
            best_wl = wavelength_labels[best_pos[1]]
            
            # 柱状图子图
            ax1 = plt.subplot(2 * num_modes, 2, 2 * mode_idx + 1)
            
            x = np.arange(len(num_layer_options))
            width = 0.8 / len(wavelength_labels)
            
            for wl_idx, wl_label in enumerate(wavelength_labels):
                offset = (wl_idx - len(wavelength_labels)/2 + 0.5) * width
                values = mode_data[:, wl_idx]
                color = colors[wl_idx % len(colors)]
                
                bars = ax1.bar(x + offset, values, width, label=wl_label, 
                              color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # 添加数值标签
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if value > 0.01:
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax1.set_xlabel('层数', fontsize=12, fontweight='bold')
            ax1.set_ylabel('可见度', fontsize=12, fontweight='bold')
            # 修改标题为英语
            ax1.set_title(f'Mode {mode_idx + 1} - Visibility Comparison\nBest: {best_layer} Layers @ {best_wl} ({best_vis:.3f})', 
                          fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(num_layer_options)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, 1.0)
            
            # 热图子图
            ax2 = plt.subplot(2 * num_modes, 2, 2 * mode_idx + 2)
            
            heatmap_data = mode_data.T
            # 热图子图 (继续)
            ax2 = plt.subplot(2 * num_modes, 2, 2 * mode_idx + 2)
            
            heatmap_data = mode_data.T
            
            im = ax2.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            
            # 添加数值标注
            for i in range(len(wavelength_labels)):
                for j in range(len(num_layer_options)):
                    value = heatmap_data[i, j]
                    color = 'white' if value < 0.5 else 'black'
                    ax2.text(j, i, f'{value:.3f}', ha='center', va='center', 
                            color=color, fontsize=10, fontweight='bold')
            
            ax2.set_xlabel('层数', fontsize=12, fontweight='bold')
            ax2.set_ylabel('波长', fontsize=12, fontweight='bold')
            # 修改标题为英语
            ax2.set_title(f'Mode {mode_idx + 1} - Visibility Heatmap{title_suffix}', 
                          fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(num_layer_options)))
            ax2.set_xticklabels(num_layer_options)
            ax2.set_yticks(range(len(wavelength_labels)))
            ax2.set_yticklabels(wavelength_labels)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('可见度', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 详细可见度分析图已保存: {save_path}")
        
        plt.show()
        return fig

    def create_mode_wavelength_matrix_analysis(self, visibility_by_mode, config, num_layer_options, save_path):
        """
        创建模式-波长矩阵分析
        """
        print("🔍 创建模式-波长矩阵分析...")
        
        wavelength_labels = [f'{int(wl*1e9)}nm' for wl in config.wavelengths]
        num_modes = len(visibility_by_mode)
        
        fig, axes = plt.subplots(1, len(num_layer_options), figsize=(4*len(num_layer_options), 6))
        if len(num_layer_options) == 1:
            axes = [axes]
        
        for layer_idx, layers in enumerate(num_layer_options):
            ax = axes[layer_idx]
            
            # 构建矩阵数据
            matrix_data = np.zeros((num_modes, len(wavelength_labels)))
            for mode_idx in range(num_modes):
                for wl_idx in range(len(wavelength_labels)):
                    if layer_idx < len(visibility_by_mode[mode_idx]):
                        matrix_data[mode_idx, wl_idx] = visibility_by_mode[mode_idx][layer_idx][wl_idx]
            
            # 绘制热图
            im = ax.imshow(matrix_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            
            # 添加数值标注
            for i in range(num_modes):
                for j in range(len(wavelength_labels)):
                    value = matrix_data[i, j]
                    color = 'white' if value < 0.5 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color=color, fontsize=12, fontweight='bold')
            
            ax.set_xlabel('波长', fontsize=12, fontweight='bold')
            ax.set_ylabel('模式', fontsize=12, fontweight='bold')
            # 修改标题为英语
            ax.set_title(f'{layers} Layers - Mode x Wavelength Matrix', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(wavelength_labels)))
            ax.set_xticklabels(wavelength_labels)
            ax.set_yticks(range(num_modes))
            ax.set_yticklabels([f'模式{i+1}' for i in range(num_modes)])
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('可见度', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 模式-波长矩阵分析已保存: {save_path}")
        plt.show()

    def _create_performance_statistics(self, visibility_by_mode, config, num_layer_options, save_dir):
        """
        创建性能统计
        """
        print("📊 创建性能统计...")
        
        # 收集所有数据
        all_data = []
        for mode_idx, mode_data in enumerate(visibility_by_mode):
            for layer_idx, layer_data in enumerate(mode_data):
                for wl_idx, vis_value in enumerate(layer_data):
                    all_data.append({
                        'mode': mode_idx + 1,
                        'layers': num_layer_options[layer_idx],
                        'wavelength': int(config.wavelengths[wl_idx] * 1e9),
                        'visibility': vis_value
                    })
        
        # 统计信息
        vis_values = [d['visibility'] for d in all_data]
        stats = {
            'total_configs': len(all_data),
            'mean_visibility': np.mean(vis_values),
            'std_visibility': np.std(vis_values),
            'max_visibility': np.max(vis_values),
            'min_visibility': np.min(vis_values),
            'median_visibility': np.median(vis_values)
        }
        
        # 找到最佳配置
        best_idx = np.argmax(vis_values)
        best_config = all_data[best_idx]
        
        # 保存统计信息
        stats_path = os.path.join(save_dir, 'visibility_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': {k: float(v) for k, v in stats.items()},
                'best_configuration': best_config
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 性能统计已保存: {stats_path}")

    def _export_visibility_data(self, visibility_by_mode, config, num_layer_options, save_dir):
        """
        导出可见度数据
        """
        print("💾 导出可见度数据...")
        
        # CSV格式
        csv_data = []
        headers = ['Mode', 'Layers', 'Wavelength_nm', 'Visibility']
        
        for mode_idx, mode_data in enumerate(visibility_by_mode):
            for layer_idx, layer_data in enumerate(mode_data):
                for wl_idx, vis_value in enumerate(layer_data):
                    csv_data.append([
                        mode_idx + 1,
                        num_layer_options[layer_idx],
                        int(config.wavelengths[wl_idx] * 1e9),
                        vis_value
                    ])
        
        csv_path = os.path.join(save_dir, 'visibility_data.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(csv_data)
        
        print(f"✅ 可见度数据已导出: {csv_path}")

# ==================== 双维度分析报告方法 ====================

    def create_dual_dimension_analysis_report(self, dual_visibility_data, config, num_layer_options, save_dir):
        """
        创建双维度可见度分析报告
        """
        print("📊 创建双维度可见度分析报告...")
        
        if not dual_visibility_data:
            print("❌ 没有双维度可见度数据")
            return
        
        # 组织数据
        cross_scores = {}
        snr_scores = {}
        comprehensive_scores = {}
        
        for key, result in dual_visibility_data.items():
            layers, mode_idx, wavelength = key
            cross_scores[key] = result['scores']['cross_score']
            snr_scores[key] = result['scores']['snr_score'] 
            comprehensive_scores[key] = result['scores']['comprehensive']
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Cross Matrix Score 对比
        ax1 = axes[0, 0]
        self._plot_dual_score_comparison(cross_scores, config, num_layer_options, ax1, 
                                       'Cross Matrix Score', 'Convergence Intensity')
        
        # 2. SNR Score 对比  
        ax2 = axes[0, 1]
        self._plot_dual_score_comparison(snr_scores, config, num_layer_options, ax2,
                                       'SNR Score', 'Signal-to-Noise Ratio')
        
        # 3. 综合Score对比
        ax3 = axes[1, 0]
        self._plot_dual_score_comparison(comprehensive_scores, config, num_layer_options, ax3,
                                       'Comprehensive Score', 'Overall Visibility')
        
        # 4. 散点图：Cross vs SNR
        ax4 = axes[1, 1]
        
        cross_vals = list(cross_scores.values())
        snr_vals = list(snr_scores.values())
        comp_vals = list(comprehensive_scores.values())
        
        scatter = ax4.scatter(cross_vals, snr_vals, c=comp_vals, cmap='viridis', 
                             s=100, alpha=0.7, edgecolors='black')
        
        ax4.set_xlabel('Cross Matrix Score', fontweight='bold')
        ax4.set_ylabel('SNR Score', fontweight='bold') 
        ax4.set_title('Cross Matrix vs SNR Score Distribution', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Comprehensive Score', fontweight='bold')
        
        # 添加对角线
        ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal Performance Line')
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存报告
        report_path = os.path.join(save_dir, 'dual_dimension_visibility_report.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # 创建详细的单个配置分析
        self._create_detailed_dual_analysis(dual_visibility_data, config, num_layer_options, save_dir)
        
        print(f"✅ 双维度可见度分析报告已保存: {report_path}")

    def _plot_dual_score_comparison(self, scores, config, num_layer_options, ax, title, ylabel):
        """
        绘制双维度评分对比图
        """
        # 按模式和波长组织数据
        wavelength_labels = [f'{int(wl*1e9)}nm' for wl in config.wavelengths]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for mode_idx in range(config.num_modes):
            for wl_idx, wl_nm in enumerate([int(wl*1e9) for wl in config.wavelengths]):
                # 收集该模式和波长的数据
                layer_scores = []
                for layers in num_layer_options:
                    key = (layers, mode_idx, wl_nm)
                    if key in scores:
                        layer_scores.append(scores[key])
                    else:
                        layer_scores.append(0.0)
                
                # 绘制线条
                color = colors[wl_idx % len(colors)]
                linestyle = '-' if mode_idx == 0 else '--' if mode_idx == 1 else ':'
                label = f'Mode{mode_idx+1}@{wavelength_labels[wl_idx]}'
                
                ax.plot(num_layer_options, layer_scores, 
                       color=color, linestyle=linestyle, marker='o', 
                       linewidth=2, markersize=6, label=label, alpha=0.8)
        
        ax.set_xlabel('Number of Layers', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(num_layer_options)

    def _create_detailed_dual_analysis(self, dual_visibility_data, config, num_layer_options, save_dir):
        """
        为最佳配置创建详细的双维度分析
        """
        print("🔍 创建详细双维度分析...")
        
        # 找到综合评分最高的配置
        best_key = max(dual_visibility_data.keys(), 
                      key=lambda k: dual_visibility_data[k]['scores']['comprehensive'])
        best_result = dual_visibility_data[best_key]
        
        # 重新加载对应的场数据
        layers, mode_idx, wl_nm = best_key
        file_pattern = f"MC_single_*{layers}*mode{mode_idx+1}*{wl_nm}nm*.npy"
        matching_files = glob.glob(os.path.join(save_dir, file_pattern))
        
        if not matching_files:
            print(f"❌ 找不到最佳配置的场数据文件: {file_pattern}")
            return
        
        try:
            field_data = np.load(matching_files[0], allow_pickle=True)
            
            # 创建详细可视化
            detail_path = os.path.join(save_dir, f'best_config_dual_analysis_{layers}L_mode{mode_idx+1}_{wl_nm}nm.png')
            self.visualize_dual_dimension_details(field_data, best_result, detail_path, best_key)
            
        except Exception as e:
            print(f"❌ 创建详细分析失败: {e}")

    def visualize_dual_dimension_details(self, field_data, dual_result, save_path, config_key):
        """
        可视化双维度分析的详细结果
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 处理强度数据
        if np.iscomplexobj(field_data):
            intensity = np.abs(field_data)**2
        else:
            intensity = np.abs(field_data)**2
        
        if intensity.ndim > 2:
            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
        
        intensity_norm = intensity / np.max(intensity) if np.max(intensity) > 0 else intensity
        
        # 1. 原始强度分布
        ax1 = axes[0, 0]
        im1 = ax1.imshow(intensity_norm, cmap='hot', aspect='auto')
        ax1.set_title('Original Intensity Distribution', fontweight='bold', fontsize=14)
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. Cross Matrix 网格叠加
        ax2 = axes[0, 1]
        ax2.imshow(intensity_norm, cmap='hot', aspect='auto', alpha=0.7)
        
        # 叠加网格线和数值
        cross_matrix = dual_result['cross_matrix']['cross_matrix']
        grid_size = dual_result['cross_matrix']['grid_size']
        H, W = intensity.shape
        
        # 绘制网格线
        for i in range(grid_size + 1):
            y_pos = i * H // grid_size
            ax2.axhline(y=y_pos, color='cyan', linewidth=2, alpha=0.8)
        for j in range(grid_size + 1):
            x_pos = j * W // grid_size
            ax2.axvline(x=x_pos, color='cyan', linewidth=2, alpha=0.8)
        
        # 标注每个区域的强度值
        region_h = H // grid_size
        region_w = W // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                center_y = i * region_h + region_h // 2
                center_x = j * region_w + region_w // 2
                value = cross_matrix[i, j]
                ax2.text(center_x, center_y, f'{value:.3f}', 
                        ha='center', va='center', color='white', 
                        fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        ax2.set_title(f'Cross Matrix Analysis ({grid_size}×{grid_size})', fontweight='bold', fontsize=14)
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        
        # 3. Cross Matrix 热图
        ax3 = axes[0, 2]
        im3 = ax3.imshow(cross_matrix, cmap='RdYlBu_r', aspect='auto')
        ax3.set_title('Cross Matrix Heatmap', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Grid X')
        ax3.set_ylabel('Grid Y')
        
        # 添加数值标注
        for i in range(grid_size):
            for j in range(grid_size):
                value = cross_matrix[i, j]
                color = 'white' if value < np.max(cross_matrix) * 0.5 else 'black'
                ax3.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        color=color, fontweight='bold', fontsize=10)
        
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. 信噪比区域分析
        ax4 = axes[1, 0]
        ax4.imshow(intensity_norm, cmap='hot', aspect='auto', alpha=0.7)
        
        # 叠加目标区域和背景区域
        snr_data = dual_result['snr']
        target_mask = snr_data['target_mask']
        background_mask = snr_data['background_mask']
        
        # 创建彩色掩码
        colored_mask = np.zeros((*intensity.shape, 3))
        colored_mask[target_mask] = [0, 1, 0]  # 绿色：目标区域
        colored_mask[background_mask] = [1, 0, 0]  # 红色：背景区域
        
        ax4.imshow(colored_mask, alpha=0.3)
        ax4.set_title('Signal (Green) vs Noise (Red) Regions', fontweight='bold', fontsize=14)
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.3, label='Signal Region'),
                          Patch(facecolor='red', alpha=0.3, label='Background Region')]
        ax4.legend(handles=legend_elements, loc='upper right')
        
        # 5. SNR 统计分布
        ax5 = axes[1, 1]
        
        signal_intensities = intensity_norm[target_mask]
        background_intensities = intensity_norm[background_mask]
        
        # 绘制直方图
        bins = np.linspace(0, 1, 50)
        ax5.hist(signal_intensities, bins=bins, alpha=0.7, color='green', 
                label=f'Signal (μ={np.mean(signal_intensities):.3f})', density=True)
        ax5.hist(background_intensities, bins=bins, alpha=0.7, color='red', 
                label=f'Background (μ={np.mean(background_intensities):.3f})', density=True)
        
        ax5.set_xlabel('Normalized Intensity')
        ax5.set_ylabel('Density')
        ax5.set_title('Signal vs Background Distribution', fontweight='bold', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 综合评分总结
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 修改总结文本为英语
        scores = dual_result['scores']
        summary = dual_result['summary']
        layers, mode_idx, wl_nm = config_key
        
        summary_text = f"""
Best Configuration Dual Dimension Analysis
-------------------------------------------
Configuration: {layers} Layers, Mode {mode_idx+1}, {wl_nm}nm

Dimension 1: Cross Matrix Convergence Intensity
  Focus Concentration: {summary['focus_concentration']:.4f}
  Cross Score: {scores['cross_score']:.4f}

Dimension 2: Signal-to-Noise Ratio Analysis  
  SNR (dB): {summary['snr_db']:.2f}
  Contrast Ratio: {summary['contrast_ratio']:.2f}
  SNR Score: {scores['snr_score']:.4f}

Overall Evaluation:
  Comprehensive Visibility: {summary['comprehensive_visibility']:.4f}
  
Region Information:
  Signal Pixels: {dual_result['snr']['signal_region_size']}
  Noise Pixels: {dual_result['snr']['background_region_size']}
  Signal Power: {dual_result['snr']['signal_power']:.4f}
  Noise Power: {dual_result['snr']['noise_power']:.4f}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        ax6.set_title('Comprehensive Analysis Summary', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 最佳配置双维度分析图已保存: {save_path}")
        
        plt.show()
        
        return fig
    def create_snr_analysis_visualization(self, real_visibility_data, config, num_layer_options, 
                                        save_path=None, title_suffix=""):
        """
        创建专门的SNR分析可视化
        """
        if not real_visibility_data:
            print("❌ 没有可用的双维度可见度数据")
            return
        
        # 提取SNR数据
        snr_data = {}
        cross_data = {}
        
        for key, data in real_visibility_data.items():
            if 'snr_score' in data and 'cross_score' in data:
                snr_data[key] = data['snr_score']
                cross_data[key] = data['cross_score']
        
        if not snr_data:
            print("❌ 没有找到SNR数据")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. SNR热力图 - 全局概览
        self._create_snr_heatmap(ax1, data, config, num_layer_options)
        
        # 2. SNR柱状图 - 精确对比  
        self._create_snr_bar_chart(ax2, data, config, num_layer_options)
        
        plt.tight_layout()
        return fig
            
                

    def _create_snr_bar_chart(self, ax, snr_data, config, num_layer_options):
        """创建SNR柱状图"""
        # 按模式和配置组织数据
        modes = list(range(config.num_modes))
        wavelengths = [450e-9, 550e-9, 650e-9]
        
        # 为每个模式创建子图数据
        bar_width = 0.25
        x_positions = np.arange(len(num_layer_options))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']  # 蓝、红、绿
        
        for mode_idx in modes:
            ax_mode = ax if mode_idx == 0 else plt.subplot(2, 3, mode_idx + 1)
            
            for i, wavelength in enumerate(wavelengths):
                snr_values = []
                for layers in num_layer_options:
                    key = f"mode{mode_idx+1}_layers{layers}_wl{wavelength*1e9:.0f}nm"
                    snr_values.append(snr_data.get(key, 0))
                
                ax_mode.bar(x_positions + i * bar_width, snr_values, 
                        bar_width, label=f'{wavelength*1e9:.0f}nm', 
                        color=colors[i], alpha=0.8)
            
            ax_mode.set_xlabel('Number of Layers')  # Changed from Chinese "层数"
            ax_mode.set_ylabel('SNR')
            ax_mode.set_title(f'Mode {mode_idx+1} - SNR Comparison')  # Changed title to English
            ax_mode.set_xticks(x_positions + bar_width)
            ax_mode.set_xticklabels(num_layer_options)
            ax_mode.legend()
            ax_mode.grid(True, alpha=0.3)

    def _create_snr_heatmap(self, ax, snr_data, config, num_layer_options):
        """创建SNR热力图"""
        wavelengths = [450e-9, 550e-9, 650e-9]
        
        # 创建热力图数据矩阵
        heatmap_data = np.zeros((len(wavelengths), len(num_layer_options)))
        
        for i, wavelength in enumerate(wavelengths):
            for j, layers in enumerate(num_layer_options):
                # 计算所有模式的平均SNR
                snr_sum = 0
                count = 0
                for mode_idx in range(config.num_modes):
                    key = f"mode{mode_idx+1}_layers{layers}_wl{wavelength*1e9:.0f}nm"
                    if key in snr_data:
                        snr_sum += snr_data[key]
                        count += 1
                
                if count > 0:
                    heatmap_data[i, j] = snr_sum / count
        
        # 创建热力图
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(num_layer_options)))
        ax.set_xticklabels(num_layer_options)
        ax.set_yticks(range(len(wavelengths)))
        ax.set_yticklabels([f'{wl*1e9:.0f}nm' for wl in wavelengths])
        
        # 添加数值标注
        for i in range(len(wavelengths)):
            for j in range(len(num_layer_options)):
                ax.text(j, i, f'{heatmap_data[i, j]:.3f}', 
                    ha='center', va='center', color='black' if heatmap_data[i, j] < 0.5 else 'white')
        
        ax.set_title('SNR Heatmap (Average of All Modes)')  # Changed title to English
        ax.set_xlabel('Number of Layers')  # Changed from Chinese "层数"
        ax.set_ylabel('Wavelength')       # Changed from Chinese "波长"
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)


