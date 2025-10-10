# label_utils.py - 完整修复版本

from matplotlib import patches
import numpy as np
import torch
import matplotlib.pyplot as plt

def create_single_wavelength_centered_regions(H, W, radius, detectsize, num_modes):
    """
    单波长情况：将所有模式在图像中心垂直排列
    """
    print(f"🎯 单波长模式：创建垂直居中布局，{num_modes}个模式")
    
    evaluation_regions = []
    
    # 使用图像的真实中心
    center_x = W // 2
    center_y = H // 2
    
    # 计算垂直排列的间距，确保整体居中
    if num_modes == 1:
        mode_positions = [center_y]
    else:
        # 计算合适的间距
        spacing = min(detectsize * 4, H // (num_modes + 0.5))
        total_height = (num_modes - 1) * spacing
        start_y = center_y - total_height // 2
        mode_positions = [start_y + i * spacing for i in range(num_modes)]
    
    print(f"  图像中心: ({center_x}, {center_y})")
    print(f"  模式垂直位置: {mode_positions}")
    
    for mode_idx, mode_center_y in enumerate(mode_positions):
        mode_center_x = center_x
        
        # 边界保护
        mode_center_x = max(detectsize // 2, min(W - detectsize // 2, mode_center_x))
        mode_center_y = max(detectsize // 2, min(H - detectsize // 2, mode_center_y))
        
        # 计算检测区域边界
        x_start = max(0, mode_center_x - detectsize // 2)
        x_end = min(W, mode_center_x + detectsize // 2)
        y_start = max(0, mode_center_y - detectsize // 2)
        y_end = min(H, mode_center_y + detectsize // 2)
        
        evaluation_regions.append((x_start, x_end, y_start, y_end))
        
        print(f"  模式{mode_idx+1}: 中心({mode_center_x}, {mode_center_y}), 区域({x_start}, {x_end}, {y_start}, {y_end})")
    
    return evaluation_regions

def create_evaluation_regions_by_wavelength(H, W, radius, detectsize, offsets=None, num_modes=3):
    """
    修改版本：支持单波长、单模式多波长和多模式多波长的评估区域创建
    """
    # 从offsets判断波长数量
    if offsets is None:
        num_wavelengths = 2  # 默认值
    else:
        num_wavelengths = len(offsets)
    
    print(f"📊 评估区域创建参数:")
    print(f"   图像尺寸: {H}x{W}")
    print(f"   模式数: {num_modes}, 波长数: {num_wavelengths}")
    print(f"   检测区域大小: {detectsize}x{detectsize}")
    
    # 🔧 单波长特殊处理（多模式单波长）
    if num_wavelengths == 1:
        return create_single_wavelength_centered_regions(H, W, radius, detectsize, num_modes)
    
    # 🔧 单模式多波长特殊处理
    if num_modes == 1:
        return create_single_mode_multi_wavelength_regions(H, W, radius, detectsize, num_wavelengths, offsets)
    
    # 多波长多模式情况保持原有逻辑
    output_image = np.zeros((H, W))
    evaluation_regions = []
    
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
    
    print(f"多模式多波长布局:")
    print(f"  列宽度: {col_width}, 行高度: {row_height}")
    
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

def visualize_evaluation_regions(H, W, regions, title="评估区域分布"):
    """
    可视化评估区域分布
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建空白图像
    image = np.zeros((H, W))
    
    # 为每个区域分配不同的颜色值
    for i, (x_start, x_end, y_start, y_end) in enumerate(regions):
        image[y_start:y_end, x_start:x_end] = i + 1
    
    # 显示图像
    im = ax.imshow(image, cmap='tab10', vmin=0, vmax=len(regions))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 添加区域标签
    for i, (x_start, x_end, y_start, y_end) in enumerate(regions):
        center_x = (x_start + x_end) / 2
        center_y = (y_start + y_end) / 2
        ax.text(center_x, center_y, f'{i+1}', 
                ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('区域编号', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()

def visualize_labels_by_wavelength(labels, wavelengths, save_path=None, show_colorbar=False, save_individual=False):
    """
    按波长分列的标签可视化 - 黑底纯净版本
    
    参数:
        save_individual: bool, 是否保存单个纯净图像（无标识文字）
    """
    # 转换为numpy数组
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    num_modes = labels.shape[0]
    num_wl = labels.shape[1]
    
    print(f"可视化标签: {num_modes} 个模式, {num_wl} 个波长")
    
    # 🔧 如果需要保存单个纯净图像
    if save_individual and save_path is not None:
        save_individual_clean_images(labels, wavelengths, save_path)
    
    # 🔧 创建黑底图像
    fig, axes = plt.subplots(num_modes, num_wl, figsize=(num_wl*3, num_modes*3), 
                            facecolor='black')
    
    # 处理单行或单列的情况
    if num_modes == 1 and num_wl == 1:
        axes = np.array([[axes]])
    elif num_modes == 1:
        axes = axes.reshape(1, -1)
    elif num_wl == 1:
        axes = axes.reshape(-1, 1)
    
    # 🔧 绘制每个标签，使用黑底配色
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wl):
            label_data = labels[mode_idx, wl_idx]
            
            # 🔧 设置子图黑色背景
            axes[mode_idx, wl_idx].set_facecolor('black')
            
            # 🔧 使用热图配色方案，黑色为背景
            im = axes[mode_idx, wl_idx].imshow(label_data, 
                                             cmap='hot',  # 改为hot配色，黑底红黄色
                                             vmin=0, vmax=1,
                                             interpolation='bilinear')
            
            # 移除坐标轴
            axes[mode_idx, wl_idx].axis('off')
            
            # 🔧 添加模式标识（白色文字）
            axes[mode_idx, wl_idx].text(0.02, 0.98, f'M{mode_idx+1}', 
                                      transform=axes[mode_idx, wl_idx].transAxes,
                                      color='white', fontsize=12, fontweight='bold',
                                      verticalalignment='top')
            
            # 🔧 在第一行添加波长标识
            if mode_idx == 0 and wavelengths is not None and wl_idx < len(wavelengths):
                wl_nm = int(wavelengths[wl_idx] * 1e9)
                axes[mode_idx, wl_idx].text(0.5, 0.98, f'λ = {wl_nm}nm', 
                                          transform=axes[mode_idx, wl_idx].transAxes,
                                          color='white', fontsize=10,
                                          horizontalalignment='center',
                                          verticalalignment='top')
    
    # 🔧 移除子图间的间距，设置黑色背景
    plt.subplots_adjust(wspace=0.02, hspace=0.02, 
                       left=0.01, right=0.99, 
                       top=0.99, bottom=0.01)
    
    # 🔧 保存为黑底图像
    if save_path is not None and isinstance(save_path, str):
        try:
            plt.savefig(save_path, dpi=300, 
                       bbox_inches='tight', 
                       pad_inches=0.02,
                       facecolor='black',  # 确保保存时背景为黑色
                       edgecolor='none')
            print(f"✓ 黑底标签图像已保存到: {save_path}")
        except Exception as e:
            print(f"⚠️  保存图像失败: {e}")
    
    plt.show()

def save_individual_clean_images(labels, wavelengths, base_save_path):
    """
    保存单个纯净图像（无标识文字）
    
    参数:
        labels: 标签数据 [modes, wavelengths, H, W]
        wavelengths: 波长列表
        base_save_path: 基础保存路径，会自动添加后缀
    """
    num_modes = labels.shape[0]
    num_wl = labels.shape[1]
    
    # 从基础路径提取目录和文件名
    import os
    base_dir = os.path.dirname(base_save_path) if os.path.dirname(base_save_path) else '.'
    base_name = os.path.splitext(os.path.basename(base_save_path))[0]
    
    print(f"🎯 开始保存 {num_modes}×{num_wl} 个单独纯净图像...")
    
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wl):
            label_data = labels[mode_idx, wl_idx]
            
            # 🔧 创建单个纯净图像
            fig = plt.figure(figsize=(6, 6), facecolor='black')
            ax = fig.add_subplot(111, facecolor='black')
            
            # 🔧 纯净显示，无任何装饰
            im = ax.imshow(label_data, 
                          cmap='hot',
                          vmin=0, vmax=1,
                          interpolation='bilinear')
            
            # 🔧 完全移除所有装饰
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 🔧 构建文件名
            if wavelengths is not None and wl_idx < len(wavelengths):
                wl_nm = int(wavelengths[wl_idx] * 1e9)
                filename = f"{base_name}_M{mode_idx+1}_{wl_nm}nm.png"
            else:
                filename = f"{base_name}_M{mode_idx+1}_W{wl_idx+1}.png"
            
            filepath = os.path.join(base_dir, filename)
            
            # 🔧 保存纯净图像
            try:
                plt.savefig(filepath, dpi=300,
                           bbox_inches='tight',
                           pad_inches=0,  # 完全无边距
                           facecolor='black',
                           edgecolor='none')
                print(f"  ✓ 保存: {filename}")
            except Exception as e:
                print(f"  ⚠️  保存失败 {filename}: {e}")
            
            plt.close(fig)  # 关闭图形以释放内存
    
    print(f"✅ 完成保存 {num_modes}×{num_wl} 个纯净图像")

def visualize_labels(labels, wavelengths=None, save_path=None, show_colorbar=False, save_individual=False):
    """
    通用标签可视化函数 - 修改为黑底风格
    
    参数:
        save_individual: bool, 是否同时保存单个纯净图像
    """
    print(f"标签可视化 - 输入形状: {labels.shape}")
    
    if wavelengths is not None and not isinstance(wavelengths, str):
        if labels.ndim == 4:  # [modes, wavelengths, H, W]
            print(f"使用4D标签可视化，波长: {[f'{wl*1e9:.0f}nm' for wl in wavelengths]}")
            visualize_labels_by_wavelength(labels, wavelengths, save_path, show_colorbar, save_individual)
            return
    
    # 默认处理
    if labels.ndim == 4:  # [modes, wavelengths, H, W]
        default_wavelengths = np.array([1310e-9, 1550e-9])
        print(f"使用默认波长: {[f'{wl*1e9:.0f}nm' for wl in default_wavelengths]}")
        visualize_labels_by_wavelength(labels, default_wavelengths, save_path, show_colorbar, save_individual)
        
    elif labels.ndim == 3:  # [channels, H, W]
        print("使用3D标签可视化")
        
        # 🔧 如果需要保存单个纯净图像
        if save_individual and save_path is not None:
            save_individual_3d_clean_images(labels, save_path)
        
        num_channels = labels.shape[0]
        
        # 🔧 创建黑底图像
        fig, axes = plt.subplots(1, num_channels, figsize=(num_channels*4, 4), 
                                facecolor='black')
        if num_channels == 1:
            axes = [axes]
        
        for i in range(num_channels):
            data = labels[i].numpy() if torch.is_tensor(labels) else labels[i]
            
            # 🔧 设置黑色背景和热图配色
            axes[i].set_facecolor('black')
            im = axes[i].imshow(data, cmap='hot', vmin=0, vmax=1)
            
            # 🔧 白色标题
            axes[i].text(0.02, 0.98, f'M{i+1}', 
                        transform=axes[i].transAxes,
                        color='white', fontsize=14, fontweight='bold',
                        verticalalignment='top')
            axes[i].axis('off')
            
            if show_colorbar:
                cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                cbar.ax.yaxis.set_tick_params(color='white')
                cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), color='white')
        
        plt.tight_layout()
        
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
        
        plt.show()
        
    else:  # 2D 或其他
        print("使用2D标签可视化")
        # 🔧 创建黑底图像
        fig = plt.figure(figsize=(8, 6), facecolor='black')
        ax = fig.add_subplot(111, facecolor='black')
        
        data = labels.numpy() if torch.is_tensor(labels) else labels
        
        if data.ndim > 2:
            data = np.sum(data, axis=tuple(range(data.ndim-2)))
        
        # 🔧 使用热图配色
        im = ax.imshow(data, cmap='hot')
        
        # 🔧 白色标题
        ax.text(0.5, 0.98, '标签可视化', 
               transform=ax.transAxes,
               color='white', fontsize=16, fontweight='bold',
               horizontalalignment='center', verticalalignment='top')
        ax.axis('off')
        
        if show_colorbar:
            cbar = plt.colorbar(im)
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), color='white')
        
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
        
        plt.show()

def save_individual_3d_clean_images(labels, base_save_path):
    """
    保存3D标签的单个纯净图像
    """
    num_channels = labels.shape[0]
    
    import os
    base_dir = os.path.dirname(base_save_path) if os.path.dirname(base_save_path) else '.'
    base_name = os.path.splitext(os.path.basename(base_save_path))[0]
    
    print(f"🎯 开始保存 {num_channels} 个3D纯净图像...")
    
    for i in range(num_channels):
        data = labels[i].numpy() if torch.is_tensor(labels) else labels[i]
        
        # 🔧 创建单个纯净图像
        fig = plt.figure(figsize=(6, 6), facecolor='black')
        ax = fig.add_subplot(111, facecolor='black')
        
        # 🔧 纯净显示
        im = ax.imshow(data, cmap='hot', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 🔧 保存
        filename = f"{base_name}_M{i+1}.png"
        filepath = os.path.join(base_dir, filename)
        
        try:
            plt.savefig(filepath, dpi=300,
                       bbox_inches='tight',
                       pad_inches=0,
                       facecolor='black',
                       edgecolor='none')
            print(f"  ✓ 保存: {filename}")
        except Exception as e:
            print(f"  ⚠️  保存失败 {filename}: {e}")
        
        plt.close(fig)
    
    print(f"✅ 完成保存 {num_channels} 个3D纯净图像")

def create_detection_regions_overlay(labels, evaluation_regions, wavelengths, 
                                   save_path=None):
    """
    创建标签与检测区域的叠加显示
    """
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    num_modes, num_wl = labels.shape[0], labels.shape[1]
    
    # 创建叠加图像
    fig, axes = plt.subplots(num_modes, num_wl, 
                            figsize=(num_wl*4, num_modes*4), 
                            facecolor='black')
    
    if num_modes == 1 and num_wl == 1:
        axes = np.array([[axes]])
    elif num_modes == 1:
        axes = axes.reshape(1, -1)
    elif num_wl == 1:
        axes = axes.reshape(-1, 1)
    
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wl):
            ax = axes[mode_idx, wl_idx]
            ax.set_facecolor('black')
            
            # 显示标签
            label_data = labels[mode_idx, wl_idx]
            im = ax.imshow(label_data, cmap='hot', vmin=0, vmax=1, alpha=0.7)
            
            # 叠加检测区域
            region_idx = wl_idx * num_modes + mode_idx
            if region_idx < len(evaluation_regions):
                x_start, x_end, y_start, y_end = evaluation_regions[region_idx]
                
                # 绘制检测区域边框
                rect = patches.Rectangle((x_start, y_start), 
                                       x_end - x_start, y_end - y_start,
                                       linewidth=3, edgecolor='cyan', 
                                       facecolor='none', alpha=0.9)
                ax.add_patch(rect)
                
                # 添加区域信息
                center_x = (x_start + x_end) / 2
                center_y = (y_start + y_end) / 2
                
                # 标记中心点
                ax.plot(center_x, center_y, '+', color='white', 
                       markersize=15, markeredgewidth=3)
                
                # 显示坐标信息
                coord_text = f'({center_x:.0f},{center_y:.0f})'
                ax.text(center_x, y_end + 5, coord_text, 
                       ha='center', va='bottom', color='cyan', 
                       fontsize=8, fontweight='bold')
            
            # 添加标识
            wl_nm = int(wavelengths[wl_idx] * 1e9) if wl_idx < len(wavelengths) else wl_idx+1
            ax.text(0.02, 0.98, f'模式 {mode_idx+1}', 
                   transform=ax.transAxes, color='white', 
                   fontsize=12, fontweight='bold', verticalalignment='top')
            
            if mode_idx == 0:
                ax.text(0.5, 0.98, f'波长 {wl_nm}nm', 
                       transform=ax.transAxes, color='yellow', 
                       fontsize=10, horizontalalignment='center', 
                       verticalalignment='top')
            
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        print(f"✅ 叠加图像已保存: {save_path}")
    
    plt.show()

def create_single_mode_multi_wavelength_regions(H, W, radius, detectsize, num_wavelengths, offsets=None):
    """
    单模式多波长情况：将各波长的检测区域水平排列
    """
    print(f"🎯 单模式多波长：创建水平排列布局，{num_wavelengths}个波长")
    
    evaluation_regions = []
    
    # 水平排列布局参数
    padding_ratio = 0.2
    available_width = W * (1 - 2 * padding_ratio)
    center_y = H // 2  # 垂直居中
    
    if num_wavelengths == 1:
        # 单波长时居中
        centers_x = [W // 2]
    else:
        # 多波长时均匀分布
        spacing = available_width / (num_wavelengths - 1)
        start_x = W * padding_ratio
        centers_x = [start_x + i * spacing for i in range(num_wavelengths)]
    
    print(f"  图像中心Y: {center_y}")
    print(f"  波长水平位置: {centers_x}")
    
    for wl_idx in range(num_wavelengths):
        center_x = centers_x[wl_idx]
        center_y_adjusted = center_y
        
        # 应用偏移（如果有）
        if offsets is not None and wl_idx < len(offsets):
            offset_y, offset_x = offsets[wl_idx]
            center_x += offset_x
            center_y_adjusted += offset_y
        
        # 边界保护
        center_x = max(detectsize // 2, min(W - detectsize // 2, center_x))
        center_y_adjusted = max(detectsize // 2, min(H - detectsize // 2, center_y_adjusted))
        
        # 计算检测区域边界
        half_size = detectsize // 2
        x_start = max(0, int(center_x - half_size))
        x_end = min(W, int(center_x + half_size))
        y_start = max(0, int(center_y_adjusted - half_size))
        y_end = min(H, int(center_y_adjusted + half_size))
        
        evaluation_regions.append((x_start, x_end, y_start, y_end))
        
        print(f"  波长{wl_idx+1}: 中心({center_x:.1f}, {center_y_adjusted:.1f}), 区域({x_start}, {x_end}, {y_start}, {y_end})")
    
    return evaluation_regions
