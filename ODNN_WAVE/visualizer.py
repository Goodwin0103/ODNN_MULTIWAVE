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
from tomlkit import datetime

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
        创建增强的SNR分析可视化 - 柱状图版本
        """
        if not real_visibility_data:
            print("❌ 没有可用的双维度可见度数据")
            return None
        
        # 提取SNR数据
        snr_data = {}
        for key, data in real_visibility_data.items():
            if 'snr_score' in data:
                snr_data[key] = data['snr_score']
        
        if not snr_data:
            print("❌ 没有找到SNR数据")
            return None
        
        # 创建2x2布局的柱状图可视化
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'SNR Analysis Dashboard - Bar Chart Format{title_suffix}', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. 主要SNR柱状图 (左上)
        self._create_snr_bar_chart(axes[0, 0], snr_data, config, num_layer_options)
        
        # 2. 按波长分组的柱状图 (右上)
        self._create_wavelength_grouped_bar_chart(axes[0, 1], snr_data, config, num_layer_options)
        
        # 3. 按层数分组的柱状图 (左下)
        self._create_layer_grouped_bar_chart(axes[1, 0], snr_data, config, num_layer_options)
        
        # 4. 综合性能对比柱状图 (右下)
        self._create_comprehensive_bar_chart(axes[1, 1], snr_data, config, num_layer_options)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.15)
        
        if save_path is None:
            save_path = os.path.join(config.save_dir, f'snr_analysis_bar_charts{title_suffix}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ SNR柱状图分析已保存: {save_path}")
        
        # 返回关键统计信息
        return {
            'optimal_snr': max(snr_data.values()) if snr_data else 0,
            'optimal_config': self._find_optimal_config_from_data(snr_data),
            'total_configs': len(snr_data),
            'average_snr': np.mean(list(snr_data.values())) if snr_data else 0
        }

    def _create_snr_bar_chart(self, ax, snr_data, config, num_layer_options):
        """创建主要SNR柱状图 - 按层数和模式分组"""
        # 组织数据：按层数分组，每个层数显示不同模式的平均SNR
        modes = list(range(config.num_modes))
        bar_width = 0.25
        x_positions = np.arange(len(num_layer_options))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']  # 蓝、红、绿
        
        for mode_idx in modes:
            snr_values = []
            for layers in num_layer_options:
                # 计算该层数和模式下所有波长的平均SNR
                mode_layer_values = []
                for key, value in snr_data.items():
                    if f'mode{mode_idx+1}' in key and f'layers{layers}' in key:
                        mode_layer_values.append(value)
                
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
        ax.set_ylabel('Average SNR', fontsize=12, fontweight='bold')
        ax.set_title('SNR Performance by Layers and Modes', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels([f'{layers}L' for layers in num_layer_options])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _create_wavelength_grouped_bar_chart(self, ax, snr_data, config, num_layer_options):
        """按波长分组的柱状图"""
        wavelengths = [450, 550, 650]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # 准备数据
        wl_data = {wl: [] for wl in wavelengths}
        
        for key, snr_value in snr_data.items():
            for wl in wavelengths:
                if f'{wl}nm' in key:
                    wl_data[wl].append(snr_value)
                    break
        
        # 计算统计数据
        wl_means = []
        wl_stds = []
        wl_labels = []
        bar_colors = []
        
        for i, wl in enumerate(wavelengths):
            if wl_data[wl]:
                wl_means.append(np.mean(wl_data[wl]))
                wl_stds.append(np.std(wl_data[wl]))
                wl_labels.append(f'{wl}nm\n({len(wl_data[wl])} configs)')
                bar_colors.append(colors[i])
            else:
                wl_means.append(0)
                wl_stds.append(0)
                wl_labels.append(f'{wl}nm\n(0 configs)')
                bar_colors.append('#cccccc')
        
        # 创建柱状图
        bars = ax.bar(range(len(wavelengths)), wl_means, 
                    color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1,
                    yerr=wl_stds, capsize=5, error_kw={'linewidth': 2})
        
        # 添加数值标注
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, wl_means, wl_stds)):
            if mean_val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        
        # 标记最佳波长
        if wl_means and max(wl_means) > 0:
            best_idx = np.argmax(wl_means)
            best_bar = bars[best_idx]
            ax.text(best_bar.get_x() + best_bar.get_width()/2, 
                best_bar.get_height() + wl_stds[best_idx] + 0.03,
                '★ BEST', ha='center', va='bottom', 
                fontsize=12, color='gold', fontweight='bold')
        
        ax.set_title('SNR Performance by Wavelength', fontsize=14, fontweight='bold')
        ax.set_xlabel('Wavelength', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average SNR ± Std', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(wavelengths)))
        ax.set_xticklabels(wl_labels)
        ax.grid(True, alpha=0.3, axis='y')

    def _create_layer_grouped_bar_chart(self, ax, snr_data, config, num_layer_options):
        """按层数分组的柱状图"""
        # 准备数据
        layer_data = {layers: [] for layers in num_layer_options}
        
        for key, snr_value in snr_data.items():
            for layers in num_layer_options:
                if f'layers{layers}' in key or f'L{layers}_' in key:
                    layer_data[layers].append(snr_value)
                    break
        
        # 计算统计数据
        layer_means = []
        layer_stds = []
        layer_labels = []
        bar_colors = []
        
        # 使用渐变色
        cmap = plt.cm.viridis
        colors = [cmap(i / max(1, len(num_layer_options) - 1)) for i in range(len(num_layer_options))]
        
        for i, layers in enumerate(num_layer_options):
            if layer_data[layers]:
                layer_means.append(np.mean(layer_data[layers]))
                layer_stds.append(np.std(layer_data[layers]))
                layer_labels.append(f'{layers} Layers\n({len(layer_data[layers])} configs)')
                bar_colors.append(colors[i])
            else:
                layer_means.append(0)
                layer_stds.append(0)
                layer_labels.append(f'{layers} Layers\n(0 configs)')
                bar_colors.append('#cccccc')
        
        # 创建柱状图
        bars = ax.bar(range(len(num_layer_options)), layer_means, 
                    color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1,
                    yerr=layer_stds, capsize=5, error_kw={'linewidth': 2})
        
        # 添加数值标注
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, layer_means, layer_stds)):
            if mean_val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
        
        # 标记最佳层数
        if layer_means and max(layer_means) > 0:
            best_idx = np.argmax(layer_means)
            best_bar = bars[best_idx]
            ax.text(best_bar.get_x() + best_bar.get_width()/2, 
                best_bar.get_height() + layer_stds[best_idx] + 0.03,
                '★ BEST', ha='center', va='bottom', 
                fontsize=12, color='gold', fontweight='bold')
        
        ax.set_title('SNR Performance by Layer Count', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average SNR ± Std', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(num_layer_options)))
        ax.set_xticklabels(layer_labels)
        ax.grid(True, alpha=0.3, axis='y')

    def _create_comprehensive_bar_chart(self, ax, snr_data, config, num_layer_options):
        """综合性能对比柱状图 - 显示所有配置"""
        # 准备所有配置的数据
        configs = []
        values = []
        colors = []
        
        # 颜色映射
        wavelength_colors = {450: '#1f77b4', 550: '#ff7f0e', 650: '#2ca02c'}
        
        for key, snr_value in snr_data.items():
            # 解析配置信息
            try:
                # 提取波长
                wavelength = None
                for wl in [450, 550, 650]:
                    if f'{wl}nm' in key:
                        wavelength = wl
                        break
                
                # 提取层数
                layers = None
                for layer_num in num_layer_options:
                    if f'layers{layer_num}' in key or f'L{layer_num}_' in key:
                        layers = layer_num
                        break
                
                # 提取模式
                mode = None
                if 'mode' in key:
                    parts = key.split('_')
                    for part in parts:
                        if part.startswith('mode'):
                            mode = int(part.replace('mode', ''))
                            break
                elif '_M' in key:
                    # 处理 L5_M2_550nm 格式
                    parts = key.split('_M')
                    if len(parts) > 1:
                        mode_part = parts[1].split('_')[0]
                        mode = int(mode_part)
                
                if wavelength and layers and mode:
                    config_label = f'{wavelength}nm\n{layers}L-M{mode}'
                    configs.append(config_label)
                    values.append(snr_value)
                    colors.append(wavelength_colors.get(wavelength, '#gray'))
                    
            except Exception as e:
                continue
        
        if not values:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=14)
            return
        
        # 按性能排序
        sorted_data = sorted(zip(configs, values, colors), key=lambda x: x[1], reverse=True)
        configs, values, colors = zip(*sorted_data)
        
        # 只显示前15个配置（避免过于拥挤）
        if len(configs) > 15:
            configs = configs[:15]
            values = values[:15]
            colors = colors[:15]
        
        # 创建柱状图
        bars = ax.bar(range(len(configs)), values, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1)
        
        # 添加数值标注
        max_val = max(values) if values else 1
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.01,
                f'{value:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, rotation=0)
            
            # 标记前3名
            if i < 3:
                rank_colors = ['gold', 'silver', '#CD7F32']  # 金银铜
                rank_labels = ['1st', '2nd', '3rd']
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.05,
                    rank_labels[i], ha='center', va='bottom', 
                    fontsize=10, color=rank_colors[i], fontweight='bold')
        
        ax.set_title('Top Performing Configurations', fontsize=14, fontweight='bold')
        ax.set_xlabel('Configuration (Wavelength-Layers-Mode)', fontsize=12, fontweight='bold')
        ax.set_ylabel('SNR Score', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加图例
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='black') 
                        for color in wavelength_colors.values()]
        ax.legend(legend_elements, [f'{wl}nm' for wl in wavelength_colors.keys()], 
                loc='upper right', title='Wavelength', fontsize=10)


    def calculate_snr_from_simulation_results(self, save_dir, config, num_layer_options):
        """
        从传播仿真结果计算SNR数据（简化版）
        """
        print("🔍 从传播仿真结果计算SNR数据...")
        
        # 查找所有仿真结果文件
        result_files = glob.glob(os.path.join(save_dir, "MC_single_*.npy"))
        
        if not result_files:
            print("❌ 未找到仿真结果文件")
            return None
        
        print(f"找到 {len(result_files)} 个仿真结果文件")
        
        # 组织数据结构
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
                
                # 计算SNR
                snr_result = self.calculate_signal_noise_ratio(data)
                
                # 创建友好的键名
                key_str = f"layers{layers}_mode{mode_idx+1}_{wl_nm}nm"
                snr_data[key_str] = snr_result['snr_db'] / 20.0  # 归一化到0-1
                
                print(f"  {layers}层, 模式{mode_idx+1}, {wl_nm}nm: SNR={snr_result['snr_db']:.2f}dB")
                
            except Exception as e:
                print(f"❌ 处理文件 {filename} 时出错: {e}")
                continue
        
        print(f"成功处理 {len(snr_data)} 个数据点")
        
        return snr_data

    def _find_optimal_config_from_data(self, snr_data):
        """从SNR数据中找到最优配置"""
        if not snr_data:
            return "N/A"
        return max(snr_data, key=snr_data.get)
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
        
        # 1. 创建Cross Matrix性能柱状图
        print("   📊 创建Cross Matrix性能柱状图...")
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        fig1.suptitle(f'Cross Matrix Performance Analysis{title_suffix}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        self._create_cross_matrix_bar_chart(ax1, cross_matrix_data, config, num_layer_options)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        bar_chart_path = os.path.join(base_dir, f'{base_name}_bar_chart.png')
        plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        saved_files.append(bar_chart_path)
        print(f"   ✅ 柱状图已保存: {bar_chart_path}")
        
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

    # 辅助方法
    def _match_config(self, key, layers, mode_idx):
        """匹配配置（层数和模式）"""
        return (f'layers{layers}' in key or f'L{layers}_' in key) and \
            (f'mode{mode_idx+1}' in key or f'_M{mode_idx+1}_' in key)

    def _match_config_full(self, key, layers, mode_idx, wavelength):
        """完全匹配配置（层数、模式、波长）"""
        return self._match_config(key, layers, mode_idx) and \
            f'{wavelength}nm' in key

    print("✅ 聚焦集中度热图方法和辅助方法已定义")
   # ==================== SNR 独立可视化 ====================
    
    def create_snr_only_visualization(self, snr_data, config, num_layer_options, 
                                    save_path=None, title_suffix=""):
        """
        独立的SNR可视化
        """
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
        
        cross_path = os.path.join(save_dir, 'cross_matrix_analysis_separated.png')
        self.create_cross_matrix_visualization(cross_matrix_data, config, num_layer_options, 
                                             cross_path, "_separated")
        
        # 创建SNR可视化
        snr_path = os.path.join(save_dir, 'snr_analysis_separated.png')
        self.create_snr_only_visualization(snr_data, config, num_layer_options, 
                                         snr_path, "_separated")
        
        # 创建对比分析
        self.create_dimension_comparison_analysis(cross_matrix_data, snr_data, config, 
                                                num_layer_options, save_dir)
        
        # 保存数据
        self._save_separated_data(cross_matrix_data, snr_data, save_dir)
        
        print("✅ 分离的双维度分析报告创建完成")
        
        return cross_matrix_data, snr_data
    
    def create_dimension_comparison_analysis(self, cross_matrix_data, snr_data, config, 
                                           num_layer_options, save_dir):
        """
        创建两个维度的对比分析
        """
        print("🔍 创建维度对比分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Cross Matrix vs SNR Comparison Analysis', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. 散点图：Cross Matrix vs SNR (左上)
        self._create_cross_vs_snr_scatter(axes[0, 0], cross_matrix_data, snr_data, config)
        
        # 2. 相关性分析 (右上)
        self._create_correlation_analysis(axes[0, 1], cross_matrix_data, snr_data, config)
        
        # 3. 性能排名对比 (左下)
        self._create_performance_ranking_comparison(axes[1, 0], cross_matrix_data, snr_data, config)
        
        # 4. 最佳配置识别 (右下)
        self._create_optimal_config_identification(axes[1, 1], cross_matrix_data, snr_data, config)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        comparison_path = os.path.join(save_dir, 'dimension_comparison_analysis.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ 维度对比分析已保存: {comparison_path}")
    
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
                    if f'mode{mode_idx+1}' in key:
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
        
        # 3. 综合最佳（两个指标的加权平均）
        if cross_matrix_data and snr_data:
            composite_scores = {}
            for key in cross_matrix_data.keys():
                if key in snr_data:
                    cross_val = cross_matrix_data[key].get('focus_concentration', 0)
                    snr_val = max(0, min(1, snr_data[key].get('snr_db', 0) / 20.0))
                    composite_scores[key] = 0.5 * cross_val + 0.5 * snr_val
            
            if composite_scores:
                best_composite_key = max(composite_scores.keys(), key=composite_scores.get)
                best_configs['Composite'] = {
                    'config': best_composite_key,
                    'value': composite_scores[best_composite_key]
                }
        
        # 创建总结文本
        summary_text = "🏆 OPTIMAL CONFIGURATION ANALYSIS\n"
        summary_text += "=" * 50 + "\n\n"
        
        for metric, info in best_configs.items():
            config_name = info['config']
            value = info['value']
            
            if metric == 'Cross Matrix':
                summary_text += f"🎯 Best {metric}:\n"
                summary_text += f"   Configuration: {config_name}\n"
                summary_text += f"   Focus Concentration: {value:.4f}\n\n"
            elif metric == 'SNR':
                summary_text += f"📡 Best {metric}:\n"
                summary_text += f"   Configuration: {config_name}\n"
                summary_text += f"   SNR: {value:.2f} dB\n\n"
            elif metric == 'Composite':
                summary_text += f"⭐ Best {metric} (50% Cross + 50% SNR):\n"
                summary_text += f"   Configuration: {config_name}\n"
                summary_text += f"   Composite Score: {value:.4f}\n\n"
        
        # 添加推荐
        if 'Composite' in best_configs:
            recommended_config = best_configs['Composite']['config']
            summary_text += f"💡 RECOMMENDATION:\n"
            summary_text += f"   Use configuration: {recommended_config}\n"
            summary_text += f"   This provides the best balance between\n"
            summary_text += f"   convergence intensity and signal quality.\n"
        
        # 显示文本
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
        
        ax.set_title('Optimal Configuration Summary', fontweight='bold', fontsize=16)
    
    def _save_separated_data(self, cross_matrix_data, snr_data, save_dir):
        """保存分离的数据"""
        print("💾 保存分离的数据...")
        
        def convert_numpy_types(obj):
            """递归转换numpy类型"""
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
        
        # 保存Cross Matrix数据
        cross_data_serializable = convert_numpy_types(cross_matrix_data)
        cross_path = os.path.join(save_dir, 'cross_matrix_data.json')
        with open(cross_path, 'w', encoding='utf-8') as f:
            json.dump(cross_data_serializable, f, indent=2, ensure_ascii=False)
        
        # 保存SNR数据
        snr_data_serializable = convert_numpy_types(snr_data)
        snr_path = os.path.join(save_dir, 'snr_data.json')
        with open(snr_path, 'w', encoding='utf-8') as f:
            json.dump(snr_data_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Cross Matrix数据已保存: {cross_path}")
        print(f"✅ SNR数据已保存: {snr_path}")

    def create_cross_matrix_by_wavelength_separated(self, cross_matrix_data, config, num_layer_options, 
                                                save_path=None, title_suffix=""):
        """
        按波长分离的Cross Matrix可视化 - 创建三个独立的子图
        """
        if not cross_matrix_data:
            print("❌ 没有Cross Matrix数据")
            return None
        
        print("🎨 创建按波长分离的Cross Matrix可视化...")
        
        # 获取波长信息
        wavelengths = [int(wl * 1e9) for wl in config.wavelengths]  # 转换为nm
        wavelength_labels = [f'{wl}nm' for wl in wavelengths]
        
        # 创建1x3布局的子图
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'Cross Matrix - Focus Concentration by Wavelength{title_suffix}', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 为每个波长创建独立的图表
        modes = list(range(config.num_modes))
        mode_colors = ['#3498db', '#e74c3c', '#2ecc71']  # 蓝、红、绿
        mode_labels = ['Mode 1', 'Mode 2', 'Mode 3']
        
        for wl_idx, (wavelength, wl_label) in enumerate(zip(wavelengths, wavelength_labels)):
            ax = axes[wl_idx]
            
            # 为当前波长收集数据
            bar_width = 0.25
            x_positions = np.arange(len(num_layer_options))
            
            # 存储当前波长的最佳值用于标注
            best_value = 0
            best_config = ""
            
            for mode_idx in modes:
                focus_concentrations = []
                
                # 收集当前模式和波长的数据
                for layers in num_layer_options:
                    mode_layer_values = []
                    for key, data in cross_matrix_data.items():
                        if (self._match_config_full(key, layers, mode_idx, wavelength) and 
                            'focus_concentration' in data):
                            mode_layer_values.append(data['focus_concentration'])
                    
                    avg_focus = np.mean(mode_layer_values) if mode_layer_values else 0
                    focus_concentrations.append(avg_focus)
                    
                    # 更新最佳值
                    if avg_focus > best_value:
                        best_value = avg_focus
                        best_config = f"{layers}L-M{mode_idx+1}"
                
                # 创建当前模式的柱状图
                bars = ax.bar(x_positions + mode_idx * bar_width, focus_concentrations,
                            bar_width, label=mode_labels[mode_idx],
                            color=mode_colors[mode_idx], alpha=0.8, 
                            edgecolor='black', linewidth=1)
                
                # 添加数值标注
                for bar, value in zip(bars, focus_concentrations):
                    if value > 0.01:  # 只显示有意义的值
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom',
                            fontweight='bold', fontsize=10)
            
            # 设置当前子图的属性
            ax.set_xlabel('Number of Layers', fontsize=14, fontweight='bold')
            ax.set_ylabel('Focus Concentration', fontsize=14, fontweight='bold')
            ax.set_title(f'{wl_label}\nBest: {best_config} ({best_value:.3f})', 
                        fontsize=16, fontweight='bold')
            ax.set_xticks(x_positions + bar_width)
            ax.set_xticklabels([f'{layers}L' for layers in num_layer_options])
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.0)
            
            # 只在第一个子图显示图例
            if wl_idx == 0:
                ax.legend(loc='upper right', fontsize=12)
            
            # 标注最佳配置
            if best_value > 0:
                # 找到最佳柱子并添加星标
                max_bar_idx = np.argmax([max(ax.patches[i::len(modes)]) 
                                    for i in range(len(modes))])
                best_bars = [bar for bar in ax.patches 
                            if bar.get_height() == best_value]
                if best_bars:
                    best_bar = best_bars[0]
                    ax.text(best_bar.get_x() + best_bar.get_width()/2, 
                        best_bar.get_height() + 0.05,
                        '★', ha='center', va='bottom', 
                        fontsize=20, color='gold', fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        if save_path is None:
            save_path = os.path.join(config.save_dir, f'cross_matrix_by_wavelength_separated{title_suffix}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ 按波长分离的Cross Matrix分析已保存: {save_path}")
        return fig

    def create_wavelength_performance_comparison(self, cross_matrix_data, config, num_layer_options,
                                            save_path=None, title_suffix=""):
        """
        创建波长性能对比图 - 显示每个波长的整体性能
        """
        wavelengths = [int(wl * 1e9) for wl in config.wavelengths]
        
        # 创建2x2布局：3个波长图 + 1个总结图
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Cross Matrix Performance by Wavelength{title_suffix}', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 前三个位置放波长图
        positions = [(0, 0), (0, 1), (1, 0)]
        
        for wl_idx, wavelength in enumerate(wavelengths):
            if wl_idx < 3:  # 最多3个波长
                row, col = positions[wl_idx]
                ax = axes[row, col]
                self._create_single_wavelength_plot(ax, cross_matrix_data, config, 
                                                num_layer_options, wavelength)
        
        # 第四个位置放总结对比
        summary_ax = axes[1, 1]
        self._create_wavelength_summary_plot(summary_ax, cross_matrix_data, config, 
                                        num_layer_options, wavelengths)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        if save_path is None:
            save_path = os.path.join(config.save_dir, f'wavelength_performance_comparison{title_suffix}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return fig

    def _create_single_wavelength_plot(self, ax, cross_matrix_data, config, num_layer_options, wavelength):
        """为单个波长创建性能图"""
        modes = list(range(config.num_modes))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # 收集数据
        layer_data = {layers: [] for layers in num_layer_options}
        
        for key, data in cross_matrix_data.items():
            if f'{wavelength}nm' in key and 'focus_concentration' in data:
                for layers in num_layer_options:
                    if self._contains_layers(key, layers):
                        layer_data[layers].append(data['focus_concentration'])
                        break
        
        # 计算每个层数的统计数据
        layer_means = []
        layer_stds = []
        layer_labels = []
        
        for layers in num_layer_options:
            if layer_data[layers]:
                layer_means.append(np.mean(layer_data[layers]))
                layer_stds.append(np.std(layer_data[layers]))
                layer_labels.append(f'{layers}L')
            else:
                layer_means.append(0)
                layer_stds.append(0)
                layer_labels.append(f'{layers}L')
        
        # 创建柱状图
        bars = ax.bar(range(len(num_layer_options)), layer_means,
                    color=f'C{wavelength//200}', alpha=0.8, edgecolor='black',
                    yerr=layer_stds, capsize=5)
        
        # 添加数值标注
        for bar, mean_val, std_val in zip(bars, layer_means, layer_stds):
            if mean_val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + std_val + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
        
        # 标记最佳
        if layer_means and max(layer_means) > 0:
            best_idx = np.argmax(layer_means)
            best_bar = bars[best_idx]
            ax.text(best_bar.get_x() + best_bar.get_width()/2,
                best_bar.get_height() + layer_stds[best_idx] + 0.03,
                '★ BEST', ha='center', va='bottom',
                fontsize=12, color='gold', fontweight='bold')
        
        ax.set_title(f'{wavelength}nm Wavelength Performance', fontweight='bold', fontsize=14)
        ax.set_xlabel('Number of Layers', fontweight='bold')
        ax.set_ylabel('Focus Concentration', fontweight='bold')
        ax.set_xticks(range(len(num_layer_options)))
        ax.set_xticklabels(layer_labels)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)

    def _create_wavelength_summary_plot(self, ax, cross_matrix_data, config, num_layer_options, wavelengths):
        """创建波长总结对比图"""
        # 计算每个波长的整体性能
        wl_performance = {}
        
        for wavelength in wavelengths:
            wl_values = []
            for key, data in cross_matrix_data.items():
                if f'{wavelength}nm' in key and 'focus_concentration' in data:
                    wl_values.append(data['focus_concentration'])
            
            if wl_values:
                wl_performance[wavelength] = {
                    'mean': np.mean(wl_values),
                    'std': np.std(wl_values),
                    'max': np.max(wl_values),
                    'count': len(wl_values)
                }
            else:
                wl_performance[wavelength] = {
                    'mean': 0, 'std': 0, 'max': 0, 'count': 0
                }
        
        # 创建对比图
        wl_labels = [f'{wl}nm' for wl in wavelengths]
        means = [wl_performance[wl]['mean'] for wl in wavelengths]
        stds = [wl_performance[wl]['std'] for wl in wavelengths]
        maxs = [wl_performance[wl]['max'] for wl in wavelengths]
        
        x = np.arange(len(wavelengths))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, means, width, label='Average Performance',
                    alpha=0.8, yerr=stds, capsize=5)
        bars2 = ax.bar(x + width/2, maxs, width, label='Peak Performance',
                    alpha=0.8)
        
        # 添加数值标注
        for bar, value in zip(bars1, means):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for bar, value in zip(bars2, maxs):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Wavelength Performance Summary', fontweight='bold', fontsize=14)
        ax.set_xlabel('Wavelength', fontweight='bold')
        ax.set_ylabel('Focus Concentration', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(wl_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)

    # 辅助方法
    def _contains_layers(self, key, layers):
        """检查键是否包含指定层数"""
        return f'L{layers}_' in key or f'layers{layers}' in key
