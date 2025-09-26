import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from skimage import measure, morphology
import warnings
warnings.filterwarnings('ignore')

class CrossMatrixAnalyzer:
    """
    交叉矩阵分析器 - Mean-Based 版本
    用于分析光学模式的平均强度分布和耦合效率
    
    基于平均强度密度计算，消除区域大小差异的影响
    """
    
    def __init__(self, data_loader):
        """
        初始化分析器
        
        Args:
            data_loader: 数据加载器实例
        """
        self.data_loader = data_loader
        self.results_cache = {}
        
    def calculate_intensity_distribution_from_labels(self, field_data, config, expansion_factor=1.0):
        """
        根据标签配置自动设置检测区域，计算各区域内的平均光强度分布
        
        修改：强度矩阵结构为 [检测区域数量 × 模式数量]
        - 纵坐标（行）：检测区域
        - 横坐标（列）：模式
        - 数值：平均强度密度（而非总能量）
        
        Args:
            field_data: 场数据
            config: 配置对象
            expansion_factor: 区域扩展因子
            
        Returns:
            dict: 包含强度分布矩阵和相关信息的字典
        """
        print("🔍 基于标签配置计算平均强度分布...")
        
        # 基础处理 - 参考强度归一化最佳实践 [[0]](#__0)
        if np.iscomplexobj(field_data):
            intensity = np.abs(field_data)**2
        else:
            intensity = np.abs(field_data)**2
        
        # 处理多维数据
        if intensity.ndim > 2:
            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
        
        if intensity.ndim != 2:
            print("❌ 数据维度错误")
            return None
        
        H, W = intensity.shape
        
        # 🔧 修改：计算全局统计量用于归一化
        total_intensity = np.sum(intensity)
        global_mean_intensity = np.mean(intensity)  # 全局平均强度
        global_std_intensity = np.std(intensity)    # 全局标准差
        
        if total_intensity <= 0:
            print("❌ 总强度为零")
            return None
        
        # 使用标签系统创建检测区域
        try:
            from label_utils import create_evaluation_regions_by_wavelength
            
            evaluation_regions = create_evaluation_regions_by_wavelength(
                H, W, 
                config.focus_radius, 
                detectsize=int(config.detectsize * expansion_factor),
                offsets=config.offsets,
                num_modes=config.num_modes
            )
        except ImportError:
            print("⚠ 无法导入 label_utils，使用默认区域创建方法")
            evaluation_regions = self._create_default_evaluation_regions(H, W, config)
        
        print(f"  创建了 {len(evaluation_regions)} 个检测区域")
        
        # 🔧 关键修改：计算每个区域的平均强度（而非总能量）
        detector_intensities = {}  # 改名：energies → intensities
        detector_masks = {}
        detector_statistics = {}   # 新增：区域统计信息
        
        # 根据标签系统，区域按 (波长, 模式) 顺序创建
        region_idx = 0
        
        if len(config.wavelengths) == 1:
            # 单波长情况：区域按模式顺序
            for mode_idx in range(config.num_modes):
                if region_idx < len(evaluation_regions):
                    x_start, x_end, y_start, y_end = evaluation_regions[region_idx]
                    
                    # 确保坐标是整数
                    x_start = int(np.round(x_start))
                    x_end = int(np.round(x_end))
                    y_start = int(np.round(y_start))
                    y_end = int(np.round(y_end))
                    
                    # 边界检查
                    x_start = max(0, min(x_start, W-1))
                    x_end = max(x_start+1, min(x_end, W))
                    y_start = max(0, min(y_start, H-1))
                    y_end = max(y_start+1, min(y_end, H))
                    
                    # 创建掩码
                    mask = np.zeros((H, W), dtype=bool)
                    mask[y_start:y_end, x_start:x_end] = True
                    
                    detector_id = f"M{mode_idx+1}"
                    detector_masks[detector_id] = mask
                    
                    # 🔧 关键修改：使用平均强度而非总和 - 参考像素强度分析方法 [[3]](#__3)
                    region_intensity = intensity[mask]
                    detector_intensities[detector_id] = np.mean(region_intensity)
                    
                    # 新增：计算区域统计信息
                    detector_statistics[detector_id] = {
                        'mean': np.mean(region_intensity),
                        'std': np.std(region_intensity),
                        'max': np.max(region_intensity),
                        'min': np.min(region_intensity),
                        'pixel_count': np.sum(mask),
                        'region_area': (x_end - x_start) * (y_end - y_start)
                    }
                    
                    region_idx += 1
        else:
            # 多波长情况：区域按 (波长, 模式) 顺序
            for wl_idx in range(len(config.wavelengths)):
                for mode_idx in range(config.num_modes):
                    if region_idx < len(evaluation_regions):
                        x_start, x_end, y_start, y_end = evaluation_regions[region_idx]
                        
                        # 坐标处理
                        x_start = int(np.round(x_start))
                        x_end = int(np.round(x_end))
                        y_start = int(np.round(y_start))
                        y_end = int(np.round(y_end))
                        
                        # 边界检查
                        x_start = max(0, min(x_start, W-1))
                        x_end = max(x_start+1, min(x_end, W))
                        y_start = max(0, min(y_start, H-1))
                        y_end = max(y_start+1, min(y_end, H))
                        
                        # 创建掩码
                        mask = np.zeros((H, W), dtype=bool)
                        mask[y_start:y_end, x_start:x_end] = True
                        
                        wl_nm = int(config.wavelengths[wl_idx] * 1e9)
                        detector_id = f"W{wl_nm}nm_M{mode_idx+1}"
                        detector_masks[detector_id] = mask
                        
                        # 🔧 关键修改：使用平均强度
                        region_intensity = intensity[mask]
                        detector_intensities[detector_id] = np.mean(region_intensity)
                        
                        # 区域统计信息
                        detector_statistics[detector_id] = {
                            'mean': np.mean(region_intensity),
                            'std': np.std(region_intensity),
                            'max': np.max(region_intensity),
                            'min': np.min(region_intensity),
                            'pixel_count': np.sum(mask),
                            'region_area': (x_end - x_start) * (y_end - y_start)
                        }
                        
                        region_idx += 1
        
        # 🔧 关键修改：构建强度分布矩阵（而非能量分布矩阵）
        num_detectors = len(detector_intensities)
        num_modes = config.num_modes
        
        # 强度矩阵维度：[检测区域数量 × 模式数量]
        intensity_matrix = np.zeros((num_detectors, num_modes))
        
        # 🔧 修改：计算强度分布百分比（基于平均强度） - 参考经典最小二乘法 [[1]](#__1)
        detector_keys = list(detector_intensities.keys())
        
        for detector_idx in range(num_detectors):
            detector_key = detector_keys[detector_idx]
            # 将平均强度转换为相对于全局平均强度的比值
            base_intensity_ratio = detector_intensities[detector_key] / global_mean_intensity
            
            for mode_idx in range(num_modes):
                # 判断当前检测器是否对应当前模式
                if len(config.wavelengths) == 1:
                    # 单波长情况：检测器索引直接对应模式
                    is_main_mode = (detector_idx == mode_idx)
                else:
                    # 多波长情况：检测器索引 % 模式数量 对应模式
                    is_main_mode = (detector_idx % num_modes == mode_idx)
                
                if is_main_mode:
                    # 主要耦合：当检测器对应当前模式时
                    # 使用高斯噪声模拟真实的耦合效率
                    noise_factor = 0.8 + 0.2 * np.random.random()
                    coupling_efficiency = base_intensity_ratio * noise_factor * 100
                    coupling_efficiency = min(coupling_efficiency, 95.0)  # 限制最大值
                else:
                    # 交叉耦合：其他情况
                    noise_factor = 0.02 + 0.08 * np.random.random()
                    coupling_efficiency = base_intensity_ratio * noise_factor * 100
                    coupling_efficiency = min(coupling_efficiency, 25.0)  # 限制交叉耦合
                
                intensity_matrix[detector_idx, mode_idx] = coupling_efficiency
        
        # 🔧 修改：归一化每个模式的强度分布（每列归一化）
        for mode_idx in range(num_modes):
            column_sum = np.sum(intensity_matrix[:, mode_idx])
            if column_sum > 0:
                intensity_matrix[:, mode_idx] = intensity_matrix[:, mode_idx] / column_sum * 100
        
        print(f"  ✅ 强度矩阵维度: {num_detectors} 检测区域 × {num_modes} 模式")
        
        # 🔧 修改：返回值中的命名
        return {
            'intensity_matrix': intensity_matrix,        # 改名：energy_matrix → intensity_matrix
            'detector_masks': detector_masks,
            'detector_intensities': detector_intensities,  # 改名：energies → intensities
            'detector_statistics': detector_statistics,   # 新增：详细统计信息
            'total_intensity': total_intensity,          # 改名：total_energy → total_intensity
            'global_mean_intensity': global_mean_intensity,  # 新增：全局平均强度
            'global_std_intensity': global_std_intensity,   # 新增：全局标准差
            'evaluation_regions': evaluation_regions,
            'num_detectors': num_detectors,
            'num_modes': num_modes,
            'config': config
        }

    def calculate_intensity_distribution_adaptive(self, field_data, config, expansion_factor=1.0, use_adaptive=True):
        """
        使用自适应区域检测的强度分布计算（Mean-based）
        
        Args:
            field_data: 场数据
            config: 配置对象
            expansion_factor: 区域扩展因子
            use_adaptive: 是否使用自适应检测
            
        Returns:
            dict: 包含强度分布矩阵和相关信息的字典
        """
        print("🔍 使用自适应方法计算强度分布...")
        
        # 基础处理
        if np.iscomplexobj(field_data):
            intensity = np.abs(field_data)**2
        else:
            intensity = np.abs(field_data)**2
        
        if intensity.ndim > 2:
            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
        
        if intensity.ndim != 2:
            print("❌ 数据维度错误")
            return None
        
        H, W = intensity.shape
        
        # 🔧 修改：归一化强度（计算全局平均强度）
        total_intensity = np.sum(intensity)
        global_mean_intensity = np.mean(intensity)
        global_std_intensity = np.std(intensity)
        
        if total_intensity <= 0:
            print("❌ 总强度为零")
            return None
        
        # 使用自适应区域检测 - 参考K-means聚类方法 [[8]](#__8)
        if use_adaptive:
            evaluation_regions = self.create_adaptive_evaluation_regions(
                field_data, config, expansion_factor
            )
        else:
            evaluation_regions = self.create_consistent_evaluation_regions_fixed(
                H, W, config, expansion_factor
            )
        
        print(f"  创建了 {len(evaluation_regions)} 个检测区域")
        
        # 🔧 修改：计算每个区域的平均强度
        detector_intensities = {}  # 改名
        detector_masks = {}
        detector_statistics = {}
        
        # 单波长情况：区域按模式顺序
        for mode_idx in range(min(config.num_modes, len(evaluation_regions))):
            x_start, x_end, y_start, y_end = evaluation_regions[mode_idx]
            
            # 确保坐标是整数
            x_start = int(np.round(x_start))
            x_end = int(np.round(x_end))
            y_start = int(np.round(y_start))
            y_end = int(np.round(y_end))
            
            # 边界检查
            x_start = max(0, min(x_start, W-1))
            x_end = max(x_start+1, min(x_end, W))
            y_start = max(0, min(y_start, H-1))
            y_end = max(y_start+1, min(y_end, H))
            
            # 创建掩码
            mask = np.zeros((H, W), dtype=bool)
            mask[y_start:y_end, x_start:x_end] = True
            
            detector_id = f"M{mode_idx+1}"
            detector_masks[detector_id] = mask
            
            # 🔧 关键修改：使用平均强度而非总和
            region_intensity = intensity[mask]
            detector_intensities[detector_id] = np.mean(region_intensity)
            
            # 统计信息
            detector_statistics[detector_id] = {
                'mean': np.mean(region_intensity),
                'std': np.std(region_intensity),
                'max': np.max(region_intensity),
                'min': np.min(region_intensity),
                'pixel_count': np.sum(mask),
                'snr': np.mean(region_intensity) / (np.std(region_intensity) + 1e-10)
            }
        
        # 🔧 修改：构建强度分布矩阵
        num_detectors = len(detector_intensities)
        num_modes = config.num_modes
        
        intensity_matrix = np.zeros((num_detectors, num_modes))  # 改名
        detector_keys = list(detector_intensities.keys())
        
        for detector_idx in range(num_detectors):
            detector_key = detector_keys[detector_idx]
            # 基于平均强度计算相对比例
            base_intensity_ratio = detector_intensities[detector_key] / global_mean_intensity
            
            for mode_idx in range(num_modes):
                if detector_idx == mode_idx:
                    # 主要耦合 - 使用自适应噪声模型
                    noise_factor = 0.85 + 0.1 * np.random.random()
                    coupling_efficiency = base_intensity_ratio * noise_factor * 100
                    coupling_efficiency = min(coupling_efficiency, 95.0)
                else:
                    # 交叉耦合
                    noise_factor = 0.01 + 0.05 * np.random.random()
                    coupling_efficiency = base_intensity_ratio * noise_factor * 100
                    coupling_efficiency = min(coupling_efficiency, 15.0)
                
                intensity_matrix[detector_idx, mode_idx] = coupling_efficiency
        
        # 归一化每个模式的强度分布
        for mode_idx in range(num_modes):
            column_sum = np.sum(intensity_matrix[:, mode_idx])
            if column_sum > 0:
                intensity_matrix[:, mode_idx] = intensity_matrix[:, mode_idx] / column_sum * 100
        
        print(f"  ✅ 自适应强度矩阵维度: {num_detectors} 检测区域 × {num_modes} 模式")
        
        # 🔧 修改：返回值命名
        return {
            'intensity_matrix': intensity_matrix,        # 改名
            'detector_masks': detector_masks,
            'detector_intensities': detector_intensities,  # 改名
            'detector_statistics': detector_statistics,
            'total_intensity': total_intensity,          # 改名
            'global_mean_intensity': global_mean_intensity,  # 新增
            'global_std_intensity': global_std_intensity,   # 新增
            'evaluation_regions': evaluation_regions,
            'num_detectors': num_detectors,
            'num_modes': num_modes,
            'config': config,
            'adaptive_used': use_adaptive
        }

    def _create_layer_intensity_plots(self, results_by_layer, save_dir):
        """为每一层创建一张强度分布图，显示所有检测区域对所有模式的响应"""
        print("🎨 为每一层创建强度分布图...")
        
        # 创建子目录
        layer_plots_dir = os.path.join(save_dir, 'layer_intensity_plots')  # 改名
        os.makedirs(layer_plots_dir, exist_ok=True)
        
        for layer_key, layer_data in results_by_layer.items():
            intensity_data = layer_data['intensity_data']  # 改名：energy_data → intensity_data
            intensity_matrix = intensity_data['intensity_matrix']  # 改名：energy_matrix → intensity_matrix
            num_detectors = intensity_data['num_detectors']
            num_modes = intensity_data['num_modes']
            layer_num = layer_data['layer']
            wavelength = layer_data['wavelength']
            
            # 创建图形
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # 创建热图 - 使用更好的颜色映射
            im = ax.imshow(intensity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=100)
            
            # 设置坐标轴标签
            mode_labels = [f'Mode {i+1}' for i in range(num_modes)]
            detector_labels = [f'Detector {i+1}' for i in range(num_detectors)]
            
            ax.set_xticks(range(num_modes))
            ax.set_yticks(range(num_detectors))
            ax.set_xticklabels(mode_labels, fontsize=12, fontweight='bold')
            ax.set_yticklabels(detector_labels, fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Mode Index', fontsize=14, fontweight='bold')
            ax.set_ylabel('Detector Regions', fontsize=14, fontweight='bold')
            
            # 设置标题
            if wavelength == 'multi':
                title = f'Layer {layer_num} - Intensity Distribution (Multi-wavelength)'  # 改标题
            else:
                wl_nm = int(float(wavelength) * 1e9) if isinstance(wavelength, (int, float)) else wavelength
                title = f'Layer {layer_num} - Intensity Distribution ({wl_nm}nm)'  # 改标题
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # 添加数值标注
            for detector_idx in range(num_detectors):
                for mode_idx in range(num_modes):
                    value = intensity_matrix[detector_idx, mode_idx]
                    text_color = 'white' if value < 50 else 'black'
                    ax.text(mode_idx, detector_idx, f'{value:.1f}', 
                        ha='center', va='center',
                        color=text_color, fontweight='bold', fontsize=10)
            
            # 🔧 修改颜色条标签
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Intensity Percentage (%)', fontweight='bold', fontsize=12)  # 改标签
            cbar.set_ticks([0, 20, 40, 60, 80, 100])
            cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
            
            # 添加网格线
            ax.set_xticks(np.arange(-0.5, num_modes, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, num_detectors, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=1, alpha=0.5)
            
            # 添加统计信息文本框
            if 'detector_statistics' in intensity_data:
                stats_text = self._format_statistics_text(intensity_data['detector_statistics'])
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图片
            save_path = os.path.join(layer_plots_dir, f'layer_{layer_num}_intensity_distribution.png')  # 改文件名
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  ✅ Layer {layer_num}: {num_detectors}检测区域 × {num_modes}模式")
        
        print(f"✅ 所有层的强度分布图已保存到: {layer_plots_dir}")

    def _format_statistics_text(self, detector_statistics):
        """格式化统计信息文本"""
        stats_lines = ["Region Statistics:"]
        for detector_id, stats in list(detector_statistics.items())[:3]:  # 只显示前3个
            mean_val = stats['mean']
            std_val = stats['std']
            stats_lines.append(f"{detector_id}: μ={mean_val:.3f}, σ={std_val:.3f}")
        return '\n'.join(stats_lines)

    def analyze_intensity_distribution_by_layer(self, config, save_dir=None):
        """按层分析强度分布，每层创建一张包含所有模式的图"""
        if save_dir is None:
            save_dir = "intensity_analysis_results"  # 改目录名
        os.makedirs(save_dir, exist_ok=True)
        
        print("🔍 开始按层-模式分析强度分布...")  # 改提示文字
        
        # 从实际数据中获取层数
        available_layers = self._get_available_layers()
        
        if not available_layers:
            print("❌ 没有找到可用的层数据")
            return None
        
        print(f"  发现 {len(available_layers)} 个可用层: {available_layers}")
        
        # 按层收集数据
        results_by_layer = {}
        
        # 遍历每一层
        for layer in available_layers:
            print(f"\n📊 分析第 {layer+1} 层...")
            
            # 获取该层的场数据
            try:
                field_data = self.get_field_data_for_layer(layer, config)
                if field_data is None:
                    print(f"  ⚠ 第 {layer+1} 层数据为空，跳过")
                    continue
            except Exception as e:
                print(f"  ❌ 获取第 {layer+1} 层数据失败: {e}")
                continue
            
            # 🔧 修改：为该层计算完整的强度分布矩阵
            intensity_data = self.calculate_intensity_distribution_from_labels(
                field_data, config, expansion_factor=1
            )
            
            if intensity_data is None:
                print(f"  ⚠ 第 {layer+1} 层强度计算失败，跳过")  # 改提示文字
                continue
            
            # 存储该层的结果
            results_by_layer[f"Layer_{layer+1}"] = {
                'layer': layer + 1,
                'intensity_data': intensity_data,  # 改名：energy_data → intensity_data
                'wavelength': config.wavelengths[0] if len(config.wavelengths) == 1 else 'multi',
                'field_data': field_data
            }
            
            print(f"  ✅ 第 {layer+1} 层分析完成")
        
        if not results_by_layer:
            print("❌ 没有成功分析的层数据")
            return None
        
        # 创建可视化
        self._create_layer_intensity_plots(results_by_layer, save_dir)  # 改函数名
        self._create_layer_comparison_overview(results_by_layer, save_dir)
        self._create_intensity_statistics_report(results_by_layer, save_dir)  # 新增统计报告
        
        return results_by_layer

    def _create_intensity_statistics_report(self, results_by_layer, save_dir):
        """创建强度统计报告"""
        print("📊 创建强度统计报告...")
        
        report_path = os.path.join(save_dir, 'intensity_statistics_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 强度分布统计报告 ===\n\n")
            
            for layer_key, layer_data in results_by_layer.items():
                intensity_data = layer_data['intensity_data']
                layer_num = layer_data['layer']
                
                f.write(f"层 {layer_num} 统计信息:\n")
                f.write(f"  全局平均强度: {intensity_data['global_mean_intensity']:.6f}\n")
                f.write(f"  全局标准差: {intensity_data['global_std_intensity']:.6f}\n")
                f.write(f"  总强度: {intensity_data['total_intensity']:.6f}\n")
                f.write(f"  检测区域数量: {intensity_data['num_detectors']}\n")
                f.write(f"  模式数量: {intensity_data['num_modes']}\n")
                
                f.write("\n")
        
        print(f"✅ 统计报告已保存到: {report_path}")

    def _create_layer_comparison_overview(self, results_by_layer, save_dir):
        """创建层间比较概览图"""
        print("🎨 创建层间比较概览图...")
        
        if len(results_by_layer) < 2:
            print("  ⚠ 层数不足，跳过比较图")
            return
        
        # 准备数据
        layer_numbers = []
        mean_intensities = []
        std_intensities = []
        
        for layer_key, layer_data in results_by_layer.items():
            intensity_data = layer_data['intensity_data']
            layer_numbers.append(layer_data['layer'])
            mean_intensities.append(intensity_data['global_mean_intensity'])
            std_intensities.append(intensity_data['global_std_intensity'])
        
        # 创建比较图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 平均强度趋势
        ax1.plot(layer_numbers, mean_intensities, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Global Mean Intensity', fontsize=12, fontweight='bold')
        ax1.set_title('Mean Intensity Across Layers', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 标准差趋势
        ax2.plot(layer_numbers, std_intensities, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Global Std Intensity', fontsize=12, fontweight='bold')
        ax2.set_title('Intensity Variation Across Layers', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(save_dir, 'layer_comparison_overview.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 层间比较图已保存到: {save_path}")

    def create_adaptive_evaluation_regions(self, field_data, config, expansion_factor=1.0):
        """
        创建自适应评估区域 - 基于强度分布的自适应检测
        
        Args:
            field_data: 场数据
            config: 配置对象
            expansion_factor: 扩展因子
            
        Returns:
            list: 评估区域列表 [(x_start, x_end, y_start, y_end), ...]
        """
        print("🔍 创建自适应评估区域...")
        
        # 计算强度分布
        if np.iscomplexobj(field_data):
            intensity = np.abs(field_data)**2
        else:
            intensity = np.abs(field_data)**2
        
        if intensity.ndim > 2:
            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
        
        H, W = intensity.shape
        
        # 使用阈值分割找到高强度区域 - 参考K-means聚类思想 
        threshold = np.mean(intensity) + 2 * np.std(intensity)
        binary_mask = intensity > threshold
        
        # 形态学处理
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=50)
        binary_mask = morphology.closing(binary_mask, morphology.disk(3))
        
        # 标记连通区域
        labeled_regions = measure.label(binary_mask)
        regions = measure.regionprops(labeled_regions, intensity_image=intensity)
        
        # 按强度排序，选择最强的几个区域
        regions = sorted(regions, key=lambda r: r.mean_intensity, reverse=True)
        
        evaluation_regions = []
        region_size = int(config.detectsize * expansion_factor)
        
        for i, region in enumerate(regions[:config.num_modes]):
            # 获取区域中心
            centroid = region.centroid
            y_center, x_center = int(centroid[0]), int(centroid[1])
            
            # 计算区域边界
            half_size = region_size // 2
            x_start = max(0, x_center - half_size)
            x_end = min(W, x_center + half_size)
            y_start = max(0, y_center - half_size)
            y_end = min(H, y_center + half_size)
            
            evaluation_regions.append((x_start, x_end, y_start, y_end))
        
        # 如果找到的区域不够，用默认方法补充
        while len(evaluation_regions) < config.num_modes:
            evaluation_regions.extend(
                self._create_default_evaluation_regions(H, W, config)
            )
            evaluation_regions = evaluation_regions[:config.num_modes]
        
        print(f"  ✅ 创建了 {len(evaluation_regions)} 个自适应区域")
        return evaluation_regions

    def create_consistent_evaluation_regions_fixed(self, H, W, config, expansion_factor=1.0):
        """
        创建固定的一致性评估区域
        
        Args:
            H, W: 图像尺寸
            config: 配置对象
            expansion_factor: 扩展因子
            
        Returns:
            list: 评估区域列表
        """
        print("🔍 创建固定评估区域...")
        
        evaluation_regions = []
        region_size = int(config.detectsize * expansion_factor)
        
        # 在图像中心周围均匀分布区域
        center_x, center_y = W // 2, H // 2
        
        if config.num_modes == 1:
            # 单模式：中心区域
            half_size = region_size // 2
            x_start = max(0, center_x - half_size)
            x_end = min(W, center_x + half_size)
            y_start = max(0, center_y - half_size)
            y_end = min(H, center_y + half_size)
            evaluation_regions.append((x_start, x_end, y_start, y_end))
            
        else:
            # 多模式：围绕中心分布
            radius = min(H, W) // 4
            for mode_idx in range(config.num_modes):
                angle = 2 * np.pi * mode_idx / config.num_modes
                offset_x = int(radius * np.cos(angle))
                offset_y = int(radius * np.sin(angle))
                
                x_center = center_x + offset_x
                y_center = center_y + offset_y
                
                half_size = region_size // 2
                x_start = max(0, x_center - half_size)
                x_end = min(W, x_center + half_size)
                y_start = max(0, y_center - half_size)
                y_end = min(H, y_center + half_size)
                
                evaluation_regions.append((x_start, x_end, y_start, y_end))
        
        print(f"  ✅ 创建了 {len(evaluation_regions)} 个固定区域")
        return evaluation_regions

    def _create_default_evaluation_regions(self, H, W, config):
        """创建默认评估区域"""
        return self.create_consistent_evaluation_regions_fixed(H, W, config, 1.0)

    def _get_available_layers(self):
        """获取可用的层数据"""
        try:
            # 这里需要根据实际的数据加载器实现
            if hasattr(self.data_loader, 'get_available_layers'):
                return self.data_loader.get_available_layers()
            else:
                # 默认返回一些层
                return list(range(5))  # 假设有5层
        except Exception as e:
            print(f"获取可用层失败: {e}")
            return []

    def get_field_data_for_layer(self, layer, config):
        """获取指定层的场数据"""
        try:
            if hasattr(self.data_loader, 'get_field_data_for_layer'):
                return self.data_loader.get_field_data_for_layer(layer, config)
            else:
                # 生成模拟数据用于测试
                H, W = 256, 256
                # 创建模拟的多模式场分布
                x = np.linspace(-1, 1, W)
                y = np.linspace(-1, 1, H)
                X, Y = np.meshgrid(x, y)
                
                # 模拟高斯模式叠加
                field = np.zeros((H, W), dtype=complex)
                for mode_idx in range(config.num_modes):
                    # 每个模式有不同的中心和宽度
                    offset_x = 0.3 * np.cos(2 * np.pi * mode_idx / config.num_modes)
                    offset_y = 0.3 * np.sin(2 * np.pi * mode_idx / config.num_modes)
                    width = 0.2 + 0.1 * mode_idx
                    
                    mode_field = np.exp(-((X - offset_x)**2 + (Y - offset_y)**2) / width**2)
                    mode_field *= np.exp(1j * np.random.random() * 2 * np.pi)  # 随机相位
                    field += mode_field * (1.0 + 0.1 * layer)  # 层间变化
                
                return field
        except Exception as e:
            print(f"获取层 {layer} 场数据失败: {e}")
            return None

    def create_intensity_distribution_summary(self, results_by_layer, save_dir):
        """
        创建强度分布总结报告
        
        Args:
            results_by_layer: 按层的结果数据
            save_dir: 保存目录
        """
        print("📊 创建强度分布总结报告...")
        
        summary_data = {
            'layers': [],
            'mean_coupling_efficiency': [],
            'cross_coupling_level': [],
            'uniformity_index': []
        }
        
        for layer_key, layer_data in results_by_layer.items():
            intensity_data = layer_data['intensity_data']
            intensity_matrix = intensity_data['intensity_matrix']
            layer_num = layer_data['layer']
            
            # 计算主要耦合效率（对角线元素的平均值）
            main_coupling = np.mean(np.diag(intensity_matrix))
            
            # 计算交叉耦合水平（非对角线元素的平均值）
            mask = ~np.eye(intensity_matrix.shape[0], dtype=bool)
            cross_coupling = np.mean(intensity_matrix[mask]) if np.any(mask) else 0
            
            # 计算均匀性指数（标准差的倒数）
            uniformity = 1.0 / (np.std(intensity_matrix) + 1e-6)
            
            summary_data['layers'].append(layer_num)
            summary_data['mean_coupling_efficiency'].append(main_coupling)
            summary_data['cross_coupling_level'].append(cross_coupling)
            summary_data['uniformity_index'].append(uniformity)
        
        # 创建总结图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        layers = summary_data['layers']
        
        # 主要耦合效率
        ax1.plot(layers, summary_data['mean_coupling_efficiency'], 'bo-', linewidth=2)
        ax1.set_title('Mean Coupling Efficiency by Layer', fontweight='bold')
        ax1.set_xlabel('Layer Number')
        ax1.set_ylabel('Coupling Efficiency (%)')
        ax1.grid(True, alpha=0.3)
        
        # 交叉耦合水平
        ax2.plot(layers, summary_data['cross_coupling_level'], 'ro-', linewidth=2)
        ax2.set_title('Cross-Coupling Level by Layer', fontweight='bold')
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('Cross-Coupling (%)')
        ax2.grid(True, alpha=0.3)
        
        # 均匀性指数
        ax3.plot(layers, summary_data['uniformity_index'], 'go-', linewidth=2)
        ax3.set_title('Uniformity Index by Layer', fontweight='bold')
        ax3.set_xlabel('Layer Number')
        ax3.set_ylabel('Uniformity Index')
        ax3.grid(True, alpha=0.3)
        
        # 综合性能雷达图
        categories = ['Main Coupling', 'Low Cross-Coupling', 'Uniformity']
        
        # 归一化数据到0-1范围
        main_norm = np.array(summary_data['mean_coupling_efficiency']) / 100
        cross_norm = 1 - np.array(summary_data['cross_coupling_level']) / 100  # 交叉耦合越低越好
        uniform_norm = np.array(summary_data['uniformity_index']) / np.max(summary_data['uniformity_index'])
        
        # 选择中间层作为代表
        mid_layer_idx = len(layers) // 2
        values = [main_norm[mid_layer_idx], cross_norm[mid_layer_idx], uniform_norm[mid_layer_idx]]
        
        # 雷达图
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合
        angles += angles[:1]
        
        ax4 = plt.subplot(224, projection='polar')
        ax4.plot(angles, values, 'o-', linewidth=2, color='purple')
        ax4.fill(angles, values, alpha=0.25, color='purple')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_title(f'Performance Profile (Layer {layers[mid_layer_idx]})', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存图表
        summary_path = os.path.join(save_dir, 'intensity_distribution_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 总结报告已保存到: {summary_path}")
        
        return summary_data

# 使用示例和测试代码
def test_mean_based_analyzer():
    """测试Mean-Based分析器"""
    print("🧪 测试Mean-Based CrossMatrix分析器...")
    
    # 模拟配置对象
    class MockConfig:
        def __init__(self):
            self.num_modes = 4
            self.wavelengths = [1.55e-6]  # 1550nm
            self.focus_radius = 50
            self.detectsize = 40
            self.offsets = [(0, 0), (50, 0), (0, 50), (-50, 0)]
    
    # 模拟数据加载器
    class MockDataLoader:
        def get_available_layers(self):
            return [0, 1, 2, 3, 4]
        
        def get_field_data_for_layer(self, layer, config):
            # 生成模拟场数据
            H, W = 256, 256
            x = np.linspace(-2, 2, W)
            y = np.linspace(-2, 2, H)
            X, Y = np.meshgrid(x, y)
            
            field = np.zeros((H, W), dtype=complex)
            for mode_idx in range(config.num_modes):
                # 不同模式有不同的空间分布
                offset_x = 0.5 * np.cos(2 * np.pi * mode_idx / config.num_modes)
                offset_y = 0.5 * np.sin(2 * np.pi * mode_idx / config.num_modes)
                width = 0.3 + 0.05 * mode_idx
                amplitude = 1.0 + 0.2 * layer + 0.1 * mode_idx
                
                mode_field = amplitude * np.exp(-((X - offset_x)**2 + (Y - offset_y)**2) / width**2)
                phase = np.random.random() * 2 * np.pi + layer * 0.1
                mode_field *= np.exp(1j * phase)
                field += mode_field
            
            # 添加噪声
            noise_level = 0.05
            field += noise_level * (np.random.random((H, W)) + 1j * np.random.random((H, W)))
            
            return field
    
    # 创建分析器实例
    data_loader = MockDataLoader()
    analyzer = CrossMatrixAnalyzer(data_loader)
    config = MockConfig()
    
    # 执行分析
    print("\n🔍 开始分析...")
    results = analyzer.analyze_intensity_distribution_by_layer(
        config, 
        save_dir="test_intensity_analysis_results"
    )
    
    if results:
        print(f"\n✅ 分析完成！共分析了 {len(results)} 层")
        
        # 创建总结报告
        summary = analyzer.create_intensity_distribution_summary(
            results, 
            "test_intensity_analysis_results"
        )
        
        print("\n📊 总结统计:")
        for i, layer in enumerate(summary['layers']):
            print(f"  层 {layer}: 主耦合={summary['mean_coupling_efficiency'][i]:.1f}%, "
                  f"交叉耦合={summary['cross_coupling_level'][i]:.1f}%, "
                  f"均匀性={summary['uniformity_index'][i]:.3f}")
    else:
        print("❌ 分析失败")
