import numpy as np
import matplotlib.pyplot as plt
import torch

def create_labels_mode_wavelength(H, W, radius, mode_idx, wl_idx, offsets=None):
    """
    为特定的模式和波长组合创建标签，与evaluation_regions使用相同的坐标计算
    
    参数:
        H, W: 图像高度和宽度
        radius: 圆形区域的半径
        mode_idx: 模式索引 (0, 1, 2)
        wl_idx: 波长索引 (0, 1, 2)
        offsets: 可选的偏移列表 [(row_offset, col_offset), ...]
    
    返回:
        output_image: 二值图像，圆内为1，其它区域为0
    """
    # 初始化输出图像
    output_image = np.zeros((H, W))
    
    # 使用与create_evaluation_regions_mode_wavelength完全相同的坐标计算
    grid_size = 3
    padding = radius * 2
    cell_width = (W - 2 * padding) // grid_size
    cell_height = (H - 2 * padding) // grid_size
    
    # 计算基础圆心位置（与evaluation_regions完全一致）
    center_x = padding + wl_idx * cell_width + cell_width // 2
    center_y = padding + mode_idx * cell_height + cell_height // 2
    
    # 应用偏移（如果提供）
    if offsets is not None and wl_idx < len(offsets):
        row_offset, col_offset = offsets[wl_idx]
        center_x += col_offset
        center_y += row_offset
    
    # 创建圆形区域
    Y, X = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    output_image[dist_from_center <= radius] = 1
    
    return output_image

def create_evaluation_regions_mode_wavelength(H, W, radius, detectsize, offsets=None):
    """
    为3种模式和3种波长创建9个评估区域，支持偏移
    
    参数:
        H, W: 图像高度和宽度
        radius: 圆形区域的半径
        detectsize: 检测区域的大小
        offsets: 可选的偏移列表 [(row_offset, col_offset), ...]
    
    返回:
        evaluation_regions: 列表，包含9个区域的坐标 (x_start, x_end, y_start, y_end)
    """
    output_image = np.zeros((H, W))
    evaluation_regions = []
    
    # 计算3×3网格的布局参数
    grid_size = 3
    padding = radius * 2
    cell_width = (W - 2 * padding) // grid_size
    cell_height = (H - 2 * padding) // grid_size
    
    # 为每个模式-波长组合创建评估区域
    for mode_idx in range(grid_size):
        for wl_idx in range(grid_size):
            # 计算基础圆心位置
            center_x = padding + wl_idx * cell_width + cell_width // 2
            center_y = padding + mode_idx * cell_height + cell_height // 2
            
            # 应用偏移（如果提供）
            if offsets is not None and wl_idx < len(offsets):
                row_offset, col_offset = offsets[wl_idx]
                center_x += col_offset
                center_y += row_offset
            
            # 计算检测区域坐标
            half_size = detectsize // 2
            x_start = max(center_x - half_size, 0)
            x_end = min(center_x + half_size, W)
            y_start = max(center_y - half_size, 0)
            y_end = min(center_y + half_size, H)
            
            # 保存评估区域坐标
            evaluation_regions.append((x_start, x_end, y_start, y_end))
            
            # 在可视化图像中标记区域
            output_image[y_start:y_end, x_start:x_end] = 0.5
            
            # 标记圆形区域
            Y, X = np.ogrid[:H, :W]
            dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            output_image[dist_from_center <= radius] = 1
    
    # 显示评估区域图像
    plt.figure(figsize=(8, 8))
    plt.imshow(output_image, cmap='gray')
    plt.title('Evaluation regions for 3 modes and 3 wavelengths (with offsets)')
    plt.axis('off')
    plt.show()
    
    return evaluation_regions


def evaluate_output(self, output_field):
    """
    计算输出场在9个波长-模式组合区域的能量分布
    
    参数:
        output_field: 输出光场
    返回:
        energies: 9个区域的归一化能量值
    """
    # 处理不同输入格式
    if isinstance(output_field, torch.Tensor):
        output_field = output_field.detach().cpu().numpy()
    
    # 计算场的强度
    intensity = np.abs(output_field) ** 2
    
    # 获取9个检测区域
    regions = self._create_evaluation_regions(intensity.shape[0], intensity.shape[1])
    
    # 使用辅助函数计算区域能量
    return self.evaluate_all_regions(intensity, regions)

def evaluate_all_regions(self, intensity, regions):
    """
    计算所有区域的能量分布并归一化
    
    参数:
        intensity: 场强度
        regions: 区域坐标列表
    返回:
        normalized_energies: 归一化的能量列表
    """
    # 计算每个区域的能量
    energies = []
    for x_start, x_end, y_start, y_end in regions:
        region_energy = np.sum(intensity[y_start:y_end, x_start:x_end])
        energies.append(region_energy)
    
    # 归一化
    total_energy = sum(energies)
    if total_energy > 0:
        return [e / total_energy for e in energies]
    return energies


def evaluate_all_regions(output, evaluation_regions):
    """
    评估输出在所有检测区域中的能量分布。
    
    参数:
        output: 模型输出，形状为 [H, W]
        evaluation_regions: 评估区域列表
    
    返回:
        energies: 包含所有区域能量的列表
    """
    energies = []
    for i, region in enumerate(evaluation_regions):
        x_start, x_end, y_start, y_end = region
        detection_region = output[y_start:y_end, x_start:x_end]
        energy = np.sum(np.abs(detection_region)**2)
        energies.append(energy)
    
    return energies

def visualize_labels(labels, wavelengths):
    """generate_input_fields
    可视化多模式多波长的标签图像
    
    参数:
        labels: 形状为 [num_modes, num_wavelengths, H, W] 的张量
        wavelengths: 波长列表
    """
    num_modes = labels.shape[0]
    num_wl = labels.shape[1]
    
    plt.figure(figsize=(num_wl*3, num_modes*3))
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wl):
            plt.subplot(num_modes, num_wl, mode_idx*num_wl + wl_idx + 1)
            plt.imshow(labels[mode_idx, wl_idx], cmap='viridis')
            plt.title(f'MODE {mode_idx+1}, λ={int(wavelengths[wl_idx]*1e9)}nm')
            plt.axis('off')
    plt.tight_layout()
    plt.show()