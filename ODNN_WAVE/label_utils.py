# label_utils.py - 完整修复版本

import numpy as np
import torch
import matplotlib.pyplot as plt

def create_evaluation_regions_mode_wavelength(H, W, radius, detectsize, offsets=None):
    """
    创建按模式-波长排列的评估区域（兼容旧版本）
    这是为了保持与现有代码的兼容性
    """
    print("⚠️  使用兼容性函数 create_evaluation_regions_mode_wavelength")
    print("   建议更新为 create_evaluation_regions_by_wavelength")
    
    # 调用新的按波长分列的函数
    return create_evaluation_regions_by_wavelength(H, W, radius, detectsize, offsets)

def create_evaluation_regions_by_wavelength(H, W, radius, detectsize, offsets=None):
    """
    按波长分列的评估区域创建函数
    同一波长的所有模式分布在同一列
    """
    output_image = np.zeros((H, W))
    evaluation_regions = []
    
    num_modes = 3
    num_wavelengths = 2
    
    # 按波长分列的布局参数
    padding_ratio = 0.15
    padding_x = int(W * padding_ratio)
    padding_y = int(H * padding_ratio)
    
    # 计算可用空间
    available_width = W - 2 * padding_x
    available_height = H - 2 * padding_y
    
    # 列宽度（按波长数量分）
    col_width = available_width // num_wavelengths
    # 行高度（按模式数量分）
    row_height = available_height // num_modes
    
    print(f"创建按波长分列的评估区域:")
    print(f"  图像尺寸: {H}x{W}")
    print(f"  列宽度: {col_width}, 行高度: {row_height}")
    print(f"  检测区域大小: {detectsize}x{detectsize}")
    
    # 按波长-模式顺序创建区域
    for wl_idx in range(num_wavelengths):      # 列索引（波长）
        for mode_idx in range(num_modes):      # 行索引（模式）
            # 计算网格中心位置
            grid_center_x = padding_x + wl_idx * col_width + col_width // 2
            grid_center_y = padding_y + mode_idx * row_height + row_height // 2
            
            # 应用波长特定的偏移
            if offsets is not None and wl_idx < len(offsets):
                row_offset, col_offset = offsets[wl_idx]
                grid_center_x += col_offset
                grid_center_y += row_offset
            
            # 确保位置在有效范围内
            center_x = max(detectsize//2, min(W - detectsize//2, grid_center_x))
            center_y = max(detectsize//2, min(H - detectsize//2, grid_center_y))
            
            # 计算检测区域坐标
            half_size = detectsize // 2
            x_start = max(center_x - half_size, 0)
            x_end = min(center_x + half_size, W)
            y_start = max(center_y - half_size, 0)
            y_end = min(center_y + half_size, H)
            
            # 保存评估区域坐标
            evaluation_regions.append((x_start, x_end, y_start, y_end))
            
            print(f"  波长 {wl_idx+1}, 模式 {mode_idx+1}: 中心位置 ({center_x:.0f}, {center_y:.0f})")
    
    print(f"✓ 创建了 {len(evaluation_regions)} 个评估区域")
    return evaluation_regions

def evaluate_output(output, evaluation_regions):
    """
    评估输出在指定区域的能量
    """
    if torch.is_tensor(output):
        output = output.detach().cpu().numpy()
    
    energies = []
    for region in evaluation_regions:
        x_start, x_end, y_start, y_end = region
        region_energy = np.sum(output[y_start:y_end, x_start:x_end])
        energies.append(region_energy)
    
    return np.array(energies)

def evaluate_all_regions(outputs, evaluation_regions):
    """
    评估所有输出在所有区域的能量
    outputs: [batch_size, channels, height, width] 或 [channels, height, width]
    """
    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    
    # 处理不同的输入维度
    if outputs.ndim == 3:  # [channels, height, width]
        outputs = outputs[np.newaxis, ...]  # 添加批次维度
    
    batch_size, channels, height, width = outputs.shape
    num_regions = len(evaluation_regions)
    
    # 结果数组: [batch_size, channels, num_regions]
    all_energies = np.zeros((batch_size, channels, num_regions))
    
    for b in range(batch_size):
        for c in range(channels):
            for r, region in enumerate(evaluation_regions):
                x_start, x_end, y_start, y_end = region
                region_energy = np.sum(outputs[b, c, y_start:y_end, x_start:x_end])
                all_energies[b, c, r] = region_energy
    
    return all_energies



def visualize_labels_by_wavelength(labels, wavelengths, save_path=None, show_colorbar=False):
    """
    按波长分列的标签可视化 - 修复版本
    
    参数:
        labels: torch.Tensor, shape [modes, wavelengths, H, W]
        wavelengths: list or array, 波长列表
        save_path: str, 保存路径（可选）
        show_colorbar: bool, 是否显示颜色条（默认False）
    """
    # 转换为numpy数组
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    num_modes = labels.shape[0]
    num_wl = labels.shape[1]
    
    print(f"可视化标签: {num_modes} 个模式, {num_wl} 个波长")
    print(f"标签形状: {labels.shape}")
    
    # 🔧 创建图像，增加边距以容纳标签
    fig, axes = plt.subplots(num_modes, num_wl, figsize=(num_wl*4 + 1, num_modes*4 + 1))
    
    # 处理单行或单列的情况
    if num_modes == 1 and num_wl == 1:
        axes = np.array([[axes]])
    elif num_modes == 1:
        axes = axes.reshape(1, -1)
    elif num_wl == 1:
        axes = axes.reshape(-1, 1)
    
    # 🔧 设置列标题（波长） - 确保显示
    for wl_idx in range(num_wl):
        wl_nm = int(wavelengths[wl_idx] * 1e9)
        axes[0, wl_idx].set_title(f'λ = {wl_nm}nm', 
                                 fontsize=16, 
                                 fontweight='bold', 
                                 pad=25,
                                 color='black')
    
    # 🔧 设置行标题（模式） - 确保显示
    for mode_idx in range(num_modes):
        axes[mode_idx, 0].set_ylabel(f'MODE {mode_idx+1}', 
                                    fontsize=14, 
                                    fontweight='bold',
                                    rotation=90,
                                    labelpad=20,
                                    color='black')
    
    # 绘制每个标签
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wl):
            label_data = labels[mode_idx, wl_idx]
            
            # 检查数据范围
            vmin, vmax = label_data.min(), label_data.max()
            print(f"  模式 {mode_idx+1}, 波长 {wl_idx+1}: 数据范围 [{vmin:.3f}, {vmax:.3f}]")
            
            # 绘制图像
            im = axes[mode_idx, wl_idx].imshow(label_data, 
                                             cmap='plasma', 
                                             vmin=0, vmax=1,
                                             interpolation='bilinear')
            axes[mode_idx, wl_idx].axis('off')
            
            # 🔧 在每个子图上添加模式和波长信息
            axes[mode_idx, wl_idx].text(0.02, 0.98, f'Mode {mode_idx+1}', 
                                      transform=axes[mode_idx, wl_idx].transAxes,
                                      fontsize=12, fontweight='bold',
                                      color='white', ha='left', va='top',
                                      bbox=dict(boxstyle="round,pad=0.3", 
                                              facecolor='black', alpha=0.7))
            
            # 只有在需要时才添加颜色条
            if show_colorbar:
                plt.colorbar(im, ax=axes[mode_idx, wl_idx], fraction=0.046, pad=0.04)
    
    # 🔧 调整布局，确保标签可见
    plt.tight_layout(pad=2.0)
    
    # 🔧 添加总标题
    if save_path is not None and isinstance(save_path, str):
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✓ 图像已保存到: {save_path}")
        except Exception as e:
            print(f"⚠️  保存图像失败: {e}")
    
    plt.show()

def visualize_labels(labels, wavelengths=None, save_path=None, show_colorbar=False):
    """
    通用标签可视化函数
    """
    print(f"标签可视化 - 输入形状: {labels.shape}")
    
    if wavelengths is not None and not isinstance(wavelengths, str):
        if labels.ndim == 4:  # [modes, wavelengths, H, W]
            print(f"使用4D标签可视化，波长: {[f'{wl*1e9:.0f}nm' for wl in wavelengths]}")
            visualize_labels_by_wavelength(labels, wavelengths, save_path, show_colorbar)
            return
    
    # 默认处理其他情况
    if labels.ndim == 4:  # [modes, wavelengths, H, W]
        default_wavelengths = np.array([1310e-9, 1550e-9])
        print(f"使用默认波长: {[f'{wl*1e9:.0f}nm' for wl in default_wavelengths]}")
        visualize_labels_by_wavelength(labels, default_wavelengths, save_path, show_colorbar)
    else:
        # 简单可视化其他维度
        plt.figure(figsize=(10, 8))
        if torch.is_tensor(labels):
            data = labels.numpy()
        else:
            data = labels
            
        if data.ndim > 2:
            data = np.sum(data, axis=tuple(range(data.ndim-2)))
        
        plt.imshow(data, cmap='plasma')
        plt.title('标签可视化')
        if show_colorbar:
            plt.colorbar()
        plt.axis('off')
        
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
