from matplotlib import pyplot as plt
import numpy as np
import os

class Visualizer:
    def __init__(self, config):
        self.config = config
    
    def verify_detector_label_consistency(self):
        """
        验证检测区域与标签生成区域的一致性
        """
        print("\n=== 检测区域与标签一致性验证 ===")
        
        # 1. 获取标签生成时使用的区域
        label_regions = self._get_label_regions()
        
        # 2. 获取检测评估时使用的区域
        detector_regions = self._get_detector_regions()
        
        # 3. 比较两者是否一致
        if len(label_regions) != len(detector_regions):
            print(f"⚠️ 警告：标签区域数量({len(label_regions)})与检测区域数量({len(detector_regions)})不一致!")
        
        # 4. 详细比较每个区域
        print("\n区域比较:")
        print("索引 | 标签区域(x_start,x_end,y_start,y_end) | 检测区域(x_start,x_end,y_start,y_end) | 一致性")
        print("-" * 80)
        
        for i in range(min(len(label_regions), len(detector_regions))):
            label_region = label_regions[i]
            detector_region = detector_regions[i]
            
            # 检查是否完全一致
            is_identical = label_region == detector_region
            
            # 计算重叠率
            overlap = self._calculate_overlap(label_region, detector_region)
            
            # 打印比较结果
            status = "✅ 完全一致" if is_identical else f"❌ 不一致 (重叠率: {overlap:.2f}%)"
            print(f"{i+1:2d} | {label_region} | {detector_region} | {status}")
        
        # 5. 可视化比较
        self._visualize_region_comparison(label_regions, detector_regions)
        
        return label_regions, detector_regions
    
    def _get_label_regions(self):
        """获取标签生成时使用的区域"""
        # 这里需要根据您的具体代码实现来获取标签区域
        # 例如，从DataGenerator中获取
        if hasattr(self, 'data_generator'):
            # 假设data_generator有一个get_label_regions方法
            if hasattr(self.data_generator, 'get_label_regions'):
                return self.data_generator.get_label_regions()
        
        # 如果无法直接获取，则尝试重新计算
        # 注意：这里假设标签使用的是圆形区域
        H, W = self.config.field_size, self.config.field_size
        radius = self.config.focus_radius
        
        # 假设标签使用的是圆形区域，需要转换为矩形区域进行比较
        if hasattr(self.config, 'num_modes') and self.config.num_modes == 5:
            # 使用5个检测器的特殊布局
            centers = self._get_5_detector_centers(H, W, radius)
        else:
            # 使用基本的3个检测器布局
            centers = self._get_3_detector_centers(H, W, radius)
        
        # 将圆形区域转换为近似的矩形区域
        label_regions = []
        for center_x, center_y in centers:
            x_start = max(center_x - radius, 0)
            x_end = min(center_x + radius, W)
            y_start = max(center_y - radius, 0)
            y_end = min(center_y + radius, H)
            label_regions.append((x_start, x_end, y_start, y_end))
        
        return label_regions
    
    def _get_detector_regions(self):
        """获取检测评估时使用的区域"""
        H, W = self.config.field_size, self.config.field_size
        radius = self.config.focus_radius
        detectsize = self.config.detectsize
        
        if hasattr(self.config, 'num_modes') and self.config.num_modes == 5:
            # 使用5个检测器的特殊布局
            regions = self._create_evaluation_regions_4_MMF3_phase(H, W, radius, detectsize)
        else:
            # 使用基本的3个检测器布局
            regions = self._create_evaluation_regions(H, W, 3, radius, detectsize)
        
        return regions
    
    def _create_evaluation_regions(self, H, W):
        """
        创建9个检测区域，每个对应一种波长-模式组合
        
        参数:
            H, W: 图像的高度和宽度
        返回:
            regions: 包含9个区域坐标的列表
        """
        num_modes = self.config.num_modes
        num_wavelengths = len(self.config.wavelengths)
        
        # 计算区域大小
        region_width = W // num_wavelengths
        region_height = H // num_modes
        
        # 初始化区域列表
        regions = []
        
        # 生成网格布局的检测区域 - 行是模式，列是波长
        for mode_idx in range(num_modes):
            for wl_idx in range(num_wavelengths):
                # 计算该区域的坐标
                x_start = wl_idx * region_width
                x_end = (wl_idx + 1) * region_width
                y_start = mode_idx * region_height
                y_end = (mode_idx + 1) * region_height
                
                # 添加区域坐标
                regions.append((x_start, x_end, y_start, y_end))
        
        return regions

    
    def _create_evaluation_regions_4_MMF3_phase(self, H, W, radius, detectsize):
        """创建5个检测器的特殊布局"""
        # 计算水平间距
        top_spacing = (W - 6 * radius) / 4  # 上排3个圆之间的间距
    
        # 定义上排和下排圆的纵坐标
        top_row_y = H // 3  # 上排大约在图像1/3高度
        bottom_row_y = 2 * H // 3  # 下排大约在图像2/3高度
    
        # 上排3个圆的横坐标
        top_centers_x = [
            int((2 * radius + top_spacing) * i + radius + top_spacing) for i in range(3)
        ]
    
        # 下排两个圆的横坐标
        bottom_center_x_4 = (top_centers_x[0] + top_centers_x[1]) // 2  # 第4个圆（下方）
        bottom_center_x_5 = (top_centers_x[1] + top_centers_x[2]) // 2  # 第5个圆（下方）
    
        # 初始化检测区域列表
        regions = []
    
        # 计算并存储所有5个区域的检测正方形坐标
        for center_x, center_y in zip(
            top_centers_x + [bottom_center_x_4, bottom_center_x_5], 
            [top_row_y] * 3 + [bottom_row_y] * 2
        ):
            half_size = detectsize // 2
            x_start = max(center_x - half_size, 0)
            x_end = min(center_x + half_size, W)
            y_start = max(center_y - half_size, 0)
            y_end = min(center_y + half_size, H)
            
            # 存储检测区域坐标
            regions.append((x_start, x_end, y_start, y_end))
    
        return regions
    
    def _calculate_overlap(self, region1, region2):
        """计算两个区域的重叠率"""
        x_start1, x_end1, y_start1, y_end1 = region1
        x_start2, x_end2, y_start2, y_end2 = region2
        
        # 计算交集区域
        x_overlap = max(0, min(x_end1, x_end2) - max(x_start1, x_start2))
        y_overlap = max(0, min(y_end1, y_end2) - max(y_start1, y_start2))
        overlap_area = x_overlap * y_overlap
        
        # 计算两个区域的面积
        area1 = (x_end1 - x_start1) * (y_end1 - y_start1)
        area2 = (x_end2 - x_start2) * (y_end2 - y_start2)
        
        # 计算重叠率（相对于较小区域）
        min_area = min(area1, area2)
        if min_area == 0:
            return 0
        
        return (overlap_area / min_area) * 100
    
    def _visualize_region_comparison(self, label_regions, detector_regions):
        """可视化标签区域和检测区域的比较"""
        H, W = self.config.field_size, self.config.field_size
        
        # 创建图像
        image = np.zeros((H, W, 3))  # RGB图像
        
        # 绘制标签区域（红色）
        for x_start, x_end, y_start, y_end in label_regions:
            image[y_start:y_end, x_start:x_end, 0] = 0.5  # 红色通道
        
        # 绘制检测区域（蓝色）
        for x_start, x_end, y_start, y_end in detector_regions:
            image[y_start:y_end, x_start:x_end, 2] = 0.5  # 蓝色通道
        
        # 重叠区域将显示为紫色
        
        # 显示图像
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title('标签区域与检测区域比较')
        
        # 创建自定义图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(0.5,0,0), label='标签区域'),
            Patch(facecolor=(0,0,0.5), label='检测区域'),
            Patch(facecolor=(0.5,0,0.5), label='重叠区域')
        ]
        plt.legend(handles=legend_elements)
        
        plt.axis('on')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    
    def _get_3_detector_centers(self, H, W, radius):
        """获取3个检测器的中心位置"""
        # 计算网格中的行和列数量
        num_rows = 1
        num_cols = 3
        
        # 计算圆之间的间距
        row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
        col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)
        
        centers = []
        for r in range(1, num_rows + 1):
            for c in range(1, num_cols + 1):
                # 计算每个圆的中心
                center_row = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
                center_col = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
                centers.append((center_col, center_row))
        
        return centers
    
    def _get_5_detector_centers(self, H, W, radius):
        """获取5个检测器的中心位置"""
        # 计算水平间距
        top_spacing = (W - 6 * radius) / 4  # 上排3个圆之间的间距
    
        # 定义上排和下排圆的纵坐标
        top_row_y = H // 3  # 上排大约在图像1/3高度
        bottom_row_y = 2 * H // 3  # 下排大约在图像2/3高度
    
        # 上排3个圆的横坐标
        top_centers_x = [
            int((2 * radius + top_spacing) * i + radius + top_spacing) for i in range(3)
        ]
    
        # 下排两个圆的横坐标
        bottom_center_x_4 = (top_centers_x[0] + top_centers_x[1]) // 2  # 第4个圆（下方）
        bottom_center_x_5 = (top_centers_x[1] + top_centers_x[2]) // 2  # 第5个圆（下方）
    
        # 合并所有中心点
        centers = [(x, top_row_y) for x in top_centers_x] + [(bottom_center_x_4, bottom_row_y), (bottom_center_x_5, bottom_row_y)]
        
        return centers
    
    def unify_detector_and_label_regions(self):
        """
        统一检测区域与标签区域
        """
        print("\n=== 统一检测区域与标签区域 ===")
        
        # 1. 获取标签区域和检测区域
        label_regions, detector_regions = self.verify_detector_label_consistency()
        
        # 2. 决定使用哪种区域作为统一标准
        # 通常应该使用标签区域作为标准，因为标签已经生成
        unified_regions = label_regions
        
        # 3. 更新配置
        if hasattr(self.config, 'detector_regions'):
            self.config.detector_regions = unified_regions
        
        # 4. 打印统一后的区域
        print("\n统一后的检测区域:")
        for i, region in enumerate(unified_regions):
            x_start, x_end, y_start, y_end = region
            print(f"区域 {i+1}: x[{x_start}:{x_end}], y[{y_start}:{y_end}]")
        
        # 5. 可视化统一后的区域
        self._visualize_unified_regions(unified_regions)
        
        return unified_regions
    
    def _visualize_unified_regions(self, regions):
        """可视化统一后的区域"""
        H, W = self.config.field_size, self.config.field_size
        
        # 创建图像
        image = np.zeros((H, W))
        
        # 绘制区域
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for i, (x_start, x_end, y_start, y_end) in enumerate(regions):
            # 绘制区域
            rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, 
                                fill=True, color='green', alpha=0.3)
            ax.add_patch(rect)
            
            # 标记中心
            center_x = (x_start + x_end) // 2
            center_y = (y_start + y_end) // 2
            
            # 添加编号
            ax.text(center_x, center_y, str(i+1), 
                   color='white', fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))
        
        # 显示图像
        ax.imshow(image, cmap='gray', extent=[0, W, 0, H], origin='lower')
        ax.set_title('统一后的检测区域', fontsize=14)
        ax.set_xlabel('X (像素)', fontsize=12)
        ax.set_ylabel('Y (像素)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    
    def visualize_detector_regions(self, save_path=None):
        """
        可视化9个波长-模式组合的检测区域布局
        
        参数:
            save_path: 可选的保存路径
        """
        H, W = self.config.field_size, self.config.field_size
        num_modes = self.config.num_modes
        num_wavelengths = len(self.config.wavelengths)
        
        # 创建空白图像
        image = np.zeros((H, W))
        
        # 获取检测区域
        regions = self._create_evaluation_regions(H, W)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 定义标签
        wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
        mode_labels = [f'模式{i+1}' for i in range(num_modes)]
        
        # 绘制检测区域
        for i, (x_start, x_end, y_start, y_end) in enumerate(regions):
            # 计算当前区域对应的模式和波长
            mode_idx = i // num_wavelengths
            wl_idx = i % num_wavelengths
            
            # 计算区域中心点
            center_x = (x_start + x_end) // 2
            center_y = (y_start + y_end) // 2
            
            # 生成唯一的颜色
            color = plt.cm.tab10(mode_idx % 10)
            
            # 绘制矩形区域
            rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, 
                                fill=True, color=color, alpha=0.3, 
                                edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            
            # 添加区域标签
            label = f"{mode_labels[mode_idx]}\n{wavelength_labels[wl_idx]}"
            ax.text(center_x, center_y, f"{i+1}: {label}", 
                ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
        
        # 设置图像属性
        ax.imshow(image, cmap='gray', extent=[0, W, H, 0], origin='upper')
        ax.set_title('光场检测区域布局 (9点模式-波长组合)', fontsize=15)
        ax.set_xlabel('X (像素)', fontsize=12)
        ax.set_ylabel('Y (像素)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例
        legend_elements = []
        for i in range(num_modes):
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                            markerfacecolor=plt.cm.tab10(i % 10), 
                                            markersize=15, label=mode_labels[i]))
        ax.legend(handles=legend_elements, loc='upper right', title='模式')
        
        # 添加配置信息
        info_text = (
            f"检测器配置:\n"
            f"- 场大小: {H}×{W} 像素\n"
            f"- 总区域数: {len(regions)}\n"
            f"- 波长数: {num_wavelengths}\n"
            f"- 模式数: {num_modes}\n"
            f"- 像素尺寸: {self.config.pixel_size*1e6:.2f} μm"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            va='top', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # 保存图像
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"检测区域布局图已保存至: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return regions


    def plot_training_losses(self, losses, num_layer_options):
        """
        绘制不同层数模型的训练损失曲线
        
        参数:
            losses: 字典，键为层数，值为该层数模型的训练损失列表
            num_layer_options: 层数选项列表
        """
        plt.figure(figsize=(10, 6))
        
        for num_layers in num_layer_options:
            if num_layers in losses:
                plt.plot(losses[num_layers], label=f'层数: {num_layers}')
        
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title('不同层数模型的训练损失曲线')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return True
    # 修改 visualizer.py 中的 plot_visibility_comparison 方法

    def plot_visibility_comparison_by_mode(self, visibility_by_mode, num_layer_options):
        """
        绘制不同模式的可见度比较图
        
        参数:
            visibility_by_mode: 列表的列表，外层按模式索引，内层按层数索引
            num_layer_options: 层数选项列表
        """
        plt.figure(figsize=(12, 6))
        
        # 准备数据
        x = np.arange(len(num_layer_options))
        width = 0.8 / len(visibility_by_mode)  # 根据模式数量调整条形宽度
        
        # 绘制分组条形图
        for i, mode_vis in enumerate(visibility_by_mode):
            offset = (i - len(visibility_by_mode)/2 + 0.5) * width
            plt.bar(x + offset, mode_vis, width, label=f'Mode {i+1}')
        
        plt.xlabel('Layer')
        plt.ylabel('Visibility')
        plt.title('visibility Comparison by Mode')
        plt.xticks(x, [f'{layers} layer(s)' for layers in num_layer_options])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.show()
        
        return True

    def plot_wavelength_response(self, weights_pred: np.ndarray, wavelengths: np.ndarray, 
                                wavelength_coeffs=None, propagation_data=None, save_path=None):
        """
        Plot model response curves for different wavelengths, considering wavelength coefficients
        
        Parameters:
            weights_pred: Predicted weights, shape (N, C, N_det)
            wavelengths: List of wavelengths in meters
            wavelength_coeffs: List of wavelength coefficients, defaults to all 1s if None
            propagation_data: Optional, intensity data from optical field propagation simulation
            save_path: Optional, path to save the image
        """
        N, C, N_det = weights_pred.shape
        
        # Default to all 1s if wavelength coefficients not provided
        if wavelength_coeffs is None:
            wavelength_coeffs = np.ones(C)
        elif len(wavelength_coeffs) != C:
            raise ValueError(f"Length of wavelength coefficients ({len(wavelength_coeffs)}) does not match number of wavelengths ({C})")
        
        # Print wavelength coefficients for debugging
        print(f"Using wavelength coefficients: {wavelength_coeffs}")
        
        # Calculate normalized weights - detector responses sum to 1 for each wavelength
        norm = weights_pred / (weights_pred.sum(axis=2, keepdims=True) + 1e-12)
        
        # Calculate average response for each wavelength across all layers
        for c in range(C):
            this_W = norm[:, c, :]  # All layers' responses at wavelength c
            mean_map = this_W.mean(axis=0, keepdims=True)  # Average to one row
            
            # Apply wavelength coefficient
            weighted_map = mean_map * wavelength_coeffs[c]
            
            # Normalize again
            weighted_norm_map = weighted_map / (weighted_map.sum() + 1e-12)
            
            # Create figure
            fig = plt.figure(figsize=(10, 5))
            
            # Left: Original response heatmap
            ax1 = fig.add_subplot(121)
            im1 = ax1.imshow(mean_map, cmap='Oranges', vmin=0, vmax=1)
            plt.colorbar(im1, ax=ax1)
            ax1.set_title(f'Original Response: λ#{c}: {wavelengths[c]*1e9:.0f} nm')
            ax1.set_xlabel('Detector ID')
            ax1.set_yticks([])
            ax1.set_xticks(range(N_det))
            
            # Add value labels
            for j in range(N_det):
                ax1.text(j, 0, f"{mean_map[0, j]:.2f}",
                        ha='center', va='center', color='black')
            
            # Right: Weighted response heatmap
            ax2 = fig.add_subplot(122)
            im2 = ax2.imshow(weighted_norm_map, cmap='Oranges', vmin=0, vmax=1)
            plt.colorbar(im2, ax=ax2)
            ax2.set_title(f'Weighted Response (coeff={wavelength_coeffs[c]:.4f}): λ#{c}: {wavelengths[c]*1e9:.0f} nm')
            ax2.set_xlabel('Detector ID')
            ax2.set_yticks([])
            ax2.set_xticks(range(N_det))
            
            # Add value labels
            for j in range(N_det):
                ax2.text(j, 0, f"{weighted_norm_map[0, j]:.2f}",
                        ha='center', va='center', color='black')
            
            plt.tight_layout()
            
            # Save image
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(f"{save_path}_wavelength_{int(wavelengths[c]*1e9)}.png", dpi=300)
            
            plt.show()
            
            # If propagation data is available, plot comparison
            if propagation_data is not None and c < len(propagation_data):
                self._plot_propagation_comparison(
                    mean_map[0], weighted_norm_map[0], propagation_data[c], 
                    wavelengths[c], wavelength_coeffs[c], save_path
                )
        
        return norm

    def _plot_propagation_comparison(self, original_response, weighted_response, 
                                    propagation_data, wavelength, wavelength_coeff, save_path=None):
        """
        Compare model responses with optical field propagation results
        """
        N_det = len(original_response)
        
        # Create comparison figure
        fig = plt.figure(figsize=(12, 6))
        
        # Plot comparison of three responses
        plt.bar(np.arange(N_det) - 0.2, original_response, width=0.2, color='blue', 
                label='Model Original Response')
        plt.bar(np.arange(N_det), weighted_response, width=0.2, color='green', 
                label=f'Model Weighted Response (coeff={wavelength_coeff:.4f})')
        plt.bar(np.arange(N_det) + 0.2, propagation_data, width=0.2, color='red', 
                label='Optical Field Propagation Result')
        
        plt.xlabel('Detector ID')
        plt.ylabel('Normalized Response')
        plt.title(f'Response Comparison at Wavelength {wavelength*1e9:.0f} nm')
        plt.xticks(range(N_det))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add correlation coefficient information
        corr_orig = np.corrcoef(original_response, propagation_data)[0, 1]
        corr_weighted = np.corrcoef(weighted_response, propagation_data)[0, 1]
        
        plt.annotate(f'Original Response Correlation: {corr_orig:.4f}\nWeighted Response Correlation: {corr_weighted:.4f}', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Save image
        if save_path:
            plt.savefig(f"{save_path}_wavelength_{int(wavelength*1e9)}_comparison.png", dpi=300)
        
        plt.show()



    def plot_energy_distribution(self, all_weights_pred, num_layer_option):
        """
        绘制波长-模式的能量分布热图（简化版）
        
        参数:
        all_weights_pred: 列表，包含不同层数模型的预测权重
        num_layer_option: 列表，包含不同的层数选项
        """
        # 检查数据结构并打印调试信息
        print(f"模型数量: {len(all_weights_pred)}")
        for i, model_weights in enumerate(all_weights_pred):
            print(f"模型 {i} 权重形状: {model_weights.shape}")
        
        # 获取实际维度
        first_model = all_weights_pred[0]
        num_wavelengths = first_model.shape[0]  # 波长数
        num_modes = first_model.shape[1] if len(first_model.shape) > 1 else 1  # 模式数
        
        print(f"检测到的维度: 波长数={num_wavelengths}, 模式数={num_modes}")
        
        # 创建图表布局 - 行数为波长数，列数为不同层数
        fig, axes = plt.subplots(num_wavelengths, len(num_layer_option), 
                                figsize=(5 * len(num_layer_option), 5 * num_wavelengths))
        
        # 确保axes是二维数组
        if num_wavelengths == 1 and len(num_layer_option) == 1:
            axes = np.array([[axes]])
        elif num_wavelengths == 1:
            axes = axes.reshape(1, -1)
        elif len(num_layer_option) == 1:
            axes = axes.reshape(-1, 1)
        
        # 定义波长标签
        wavelength_labels = [f'{wl*1e9:.0f} nm' for wl in self.config.wavelengths]
        if len(wavelength_labels) != num_wavelengths:
            wavelength_labels = [f'Wavelength {i+1}' for i in range(num_wavelengths)]
        
        # 保存可见度值
        visibility_list = []
        
        # 双层循环：外层循环波长，内层循环层数
        for w_idx in range(num_wavelengths):  # 波长循环
            for m_idx, num_layer in enumerate(num_layer_option):  # 层数循环
                print(f"\n评估 {wavelength_labels[w_idx]} 波长下的 {num_layer} 层ODNN...\n")
                
                # 获取当前模型在当前波长下的权重预测
                current_weights_pred = all_weights_pred[m_idx][w_idx]  # [模式数]
                
                # 检查当前权重的维度
                print(f"当前权重形状: {current_weights_pred.shape}")
                
                # 如果数据仍有探测器维度，需要压缩
                if len(current_weights_pred.shape) > 1 and current_weights_pred.shape[-1] > num_modes:
                    # 可以选择多种方式压缩:
                    # 1. 使用对角线元素(假设模式i应该主要影响探测器i)
                    if current_weights_pred.shape[0] == current_weights_pred.shape[1]:
                        simplified_weights = np.diagonal(current_weights_pred)
                    # 2. 对每行取最大值(每个模式的最大响应)
                    else:
                        simplified_weights = np.max(current_weights_pred, axis=1)
                    
                    current_weights_pred = simplified_weights
                
                # 转为行向量以便绘制热图
                normalized_weights = current_weights_pred.reshape(1, -1)
                
                # 使用二维索引访问子图
                ax = axes[w_idx, m_idx]
                
                # 绘制热图
                im = ax.imshow(normalized_weights, cmap='Oranges', interpolation='nearest', vmin=0, vmax=1)
                
                # 根据位置设置标题和标签
                if w_idx == 0:
                    ax.set_title(f'{num_layer} Layers')
                if m_idx == 0:
                    ax.set_ylabel(f'{wavelength_labels[w_idx]}\nResponse')
                if w_idx == num_wavelengths - 1:
                    ax.set_xlabel('Mode Index')
                
                # 设置刻度
                ax.set_xticks(np.arange(normalized_weights.shape[1]))
                ax.set_xticklabels(np.arange(1, normalized_weights.shape[1] + 1))
                ax.set_yticks([])  # 不显示y轴刻度
                
                # 网格的百分比标注
                for i in range(normalized_weights.shape[0]):
                    for j in range(normalized_weights.shape[1]):
                        value = normalized_weights[i, j] * 100
                        # 根据背景色自动调整文本颜色
                        text_color = 'white' if normalized_weights[i, j] > 0.7 else 'black'
                        ax.text(j, i, f"{value:.1f}", ha='center', va='center', 
                                color=text_color, fontsize=10, fontweight='bold')
                
                # 计算可见度 (最大值和最小值的对比)
                if normalized_weights.size > 1:
                    visibility = (np.max(normalized_weights) - np.min(normalized_weights)) / (np.max(normalized_weights) + np.min(normalized_weights) + 1e-10)
                    visibility_list.append(visibility)
                    
                    # 在图上显示可见度
                    ax.text(0.5, -0.15, f"Visibility: {visibility:.3f}", transform=ax.transAxes,
                            ha='center', va='center', fontsize=9)
                    
                    print(f'Visibility for {wavelength_labels[w_idx]}, {num_layer} layers: {visibility:.3f}')
        
        # 添加共享颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Normalized Energy")
        
        # 设置总标题
        fig.suptitle("Energy Distribution per Layer", fontsize=16, y=0.98)
        
        # 调整布局
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
        
        return fig, visibility_list



    def plot_visibility_comparison_by_mode_wavelength(self, visibility_data, num_layer_options, wavelengths=None):
        """
        绘制不同模式在不同波长下的可见度比较图
        
        参数:
            visibility_data: 三维数组，形状为 [模式数, 波长数, 层数]
            num_layer_options: 层数选项列表
            wavelengths: 波长列表（单位：纳米），如果为None则使用默认标签
        """
        num_modes = visibility_data.shape[0]
        num_wavelengths = visibility_data.shape[1]
        
        # 如果没有提供波长，则使用默认标签
        if wavelengths is None:
            wavelength_labels = [f'λ{i+1}' for i in range(num_wavelengths)]
        else:
            wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in wavelengths]
        
        # 创建子图网格：每个模式一行，每个波长一列
        fig, axes = plt.subplots(num_modes, num_wavelengths, figsize=(4*num_wavelengths, 3*num_modes), 
                                sharex=True, sharey=True)
        
        # 确保axes是二维数组
        if num_modes == 1 and num_wavelengths == 1:
            axes = np.array([[axes]])
        elif num_modes == 1:
            axes = axes.reshape(1, -1)
        elif num_wavelengths == 1:
            axes = axes.reshape(-1, 1)
        
        # 为条形图准备x轴位置
        x = np.arange(len(num_layer_options))
        
        # 绘制每个子图
        for mode_idx in range(num_modes):
            for wl_idx in range(num_wavelengths):
                ax = axes[mode_idx, wl_idx]
                
                # 获取当前模式和波长的可见度数据
                vis_data = visibility_data[mode_idx, wl_idx]
                
                # 绘制条形图
                bars = ax.bar(x, vis_data, width=0.6, color=f'C{mode_idx}', alpha=0.8)
                
                # 添加数值标签
                for bar_idx, bar in enumerate(bars):
                    height = bar.get_height()
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
                
                # 设置y轴范围为0-1
                ax.set_ylim(0, 1.05)
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 添加总标题
        plt.suptitle('Visibility Comparison by Mode and Wavelength', fontsize=16, y=0.98)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        return fig
