import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

import torch
import numpy as np
import os
import time
from config import Config

start_time = time.time()

# 设置随机种子，确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("多模式多波长光场调制系统 - 训练-仿真集成版 (多波长支持 - 波长标记显示版)")
print("=" * 60)

# ===== 创建增强配置 =====

# 基本参数
num_modes = 3                                    # 模式数量
# wavelengths = np.array([1310e-9])      # 波长列表(m)
wavelengths = np.array([1310e-9, 1550e-9])      # 波长列表(m) 
base_wavelength_idx = 1                          # 基准波长索引


# 空间参数
field_size = 50                                  # 场大小(像素)
layer_size = 300                                 # 层大小(像素)
focus_radius = 10                                # 焦点半径(像素)
detectsize = 20                                  # 检测区域大小(像素)

# 物理参数
z_layers = 40e-6                                 # 层间距离(m)
z_prop = 150e-6                                  # 传播距离(m)
z_step = 20e-6                                   # 传播步长(m)
pixel_size = 1e-6                                # 像素大小(m)

# 检测区域偏移
offsets = [(0,0), (0,0)]                         # 每个波长的检测区域偏移

# 训练参数
learning_rate = 0.01                             # 学习率
lr_decay = 0.99                                  # 学习率衰减
epochs = 700                                     # 训练轮数
batch_size = 16                                  # 批量大小

# Zero Padding 参数
padding_ratio = 0.01                             # Padding 比例 (1%)
use_apodization = True                           # 启用边界衰减
apodization_width = 15                           # 衰减宽度

# MaskLoader 参数
fallback_focal_lengths = [40e-6, 60e-6, 80e-6, 100e-6, 120e-6]  # 备用掩码的焦距列表
default_num_layers = 3                           # 默认层数

# 保存参数
save_dir = f"ODNN_WAVE/results/{num_modes}_mode_{len(wavelengths)}_wl_basewl_{wavelengths[base_wavelength_idx]}_z_prop_{z_prop}_focus_{focus_radius}/"
# save_dir = f"ODNN_WAVE/results/{len(wavelengths)}_wl_basewl_{wavelengths[base_wavelength_idx]}_z_prop_{z_prop}_focus_{focus_radius}/"
flag_savemat = True

# ===== 创建Config对象 =====
config = Config(
    num_modes=num_modes,
    wavelengths=wavelengths,
    base_wavelength_idx=base_wavelength_idx,
    field_size=field_size,
    layer_size=layer_size,
    focus_radius=focus_radius,
    detectsize=detectsize,
    z_layers=z_layers,
    z_prop=z_prop,
    z_step=z_step,
    pixel_size=pixel_size,
    offsets=offsets,
    learning_rate=learning_rate,
    lr_decay=lr_decay,
    epochs=epochs,
    batch_size=batch_size,
    padding_ratio=padding_ratio,
    use_apodization=use_apodization,
    apodization_width=apodization_width,
    fallback_focal_lengths=fallback_focal_lengths,
    default_num_layers=default_num_layers,
    save_dir=save_dir,
    flag_savemat=flag_savemat
)

print(f"✅ 配置创建成功！")
print(f"波长数量: {len(config.wavelengths)}")
print(f"模式数量: {config.num_modes}")
print(f"保存目录: {config.save_dir}")

def load_field_data_multiwavelength(config):
    """加载多波长光场数据"""
    
    print("🔍 加载多波长光场数据...")
    
    # 查找所有.npy文件
    field_files = []
    base_dir = config.save_dir
    
    for file in os.listdir(base_dir):
        if file.endswith('.npy'):
            field_files.append(os.path.join(base_dir, file))
    
    print(f"✅ 找到 {len(field_files)} 个光场数据文件")
    
    # 按波长、层数和模式组织数据: wavelength -> layer -> mode -> data_list
    field_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for file_path in field_files:
        filename = os.path.basename(file_path)
        
        try:
            # 解析文件名获取波长、层数和模式信息
            parts = filename.replace('.npy', '').split('_')
            
            wavelength_nm = None
            mode_num = None
            layer_num = None
            
            for part in parts:
                if part.endswith('nm'):
                    wavelength_nm = int(part.replace('nm', ''))
                elif part.startswith('mode'):
                    mode_num = int(part.replace('mode', ''))
                elif part.endswith('layers'):
                    layer_num = int(part.replace('layers', ''))
            
            if wavelength_nm is None or mode_num is None or layer_num is None:
                print(f"❌ 跳过文件 (解析失败): {filename}")
                continue
            
            # 加载光场数据
            field = np.load(file_path)
            
            if field.size == 0:
                continue
            
            # 计算强度
            if np.iscomplexobj(field):
                intensity = np.abs(field)**2
            else:
                intensity = field**2
            
            field_data[wavelength_nm][layer_num][mode_num].append({
                'intensity': intensity,
                'field': field,
                'filename': filename,
                'wavelength': wavelength_nm,
                'layer': layer_num,
                'mode': mode_num,
                'max_intensity': np.max(intensity),
                'total_power': np.sum(intensity),
                'mean_intensity': np.mean(intensity)
            })
            
            print(f"  ✓ {wavelength_nm}nm {layer_num}层 模式{mode_num}: {field.shape}, 最大强度: {np.max(intensity):.6f}")
            
        except Exception as e:
            print(f"❌ 处理失败 {filename}: {e}")
            continue
    
    return field_data

def get_wavelength_detector_labels(wavelengths_nm, num_modes):
    """生成波长检测器标签映射"""
    # 按波长排序
    sorted_wavelengths = sorted(wavelengths_nm)
    
    # 创建标签映射：检测器索引 -> 波长标记
    detector_labels = {}
    for wl_idx, wavelength_nm in enumerate(sorted_wavelengths):
        for mode_idx in range(num_modes):
            detector_idx = wl_idx * num_modes + mode_idx
            detector_labels[detector_idx] = f'{wavelength_nm}nm-Det{mode_idx+1}'
    
    return detector_labels

def visualize_individual_mode_regions_wavelength_display(field_data, wavelength, layer_num, evaluation_regions, detector_labels, save_dir=None):
    """为每个模式单独可视化检测区域 - 波长标记显示版本（保持原有区域创建逻辑）"""
    
    print(f"   🎨 为{wavelength}nm第{layer_num}层的每个模式生成波长标记显示图片...")
    
    if wavelength not in field_data or layer_num not in field_data[wavelength]:
        print(f"   ❌ {wavelength}nm第{layer_num}层无数据")
        return
    
    layer_data = field_data[wavelength][layer_num]
    
    # 为每个模式单独生成图片
    for mode_idx in range(1, 4):  # 模式1, 2, 3
        if mode_idx not in layer_data or not layer_data[mode_idx]:
            print(f"   ⚠️ 模式{mode_idx}无数据")
            continue
        
        # 获取该模式的强度数据
        mode_data = layer_data[mode_idx][0]  # 取第一个样本
        intensity = mode_data['intensity']
        
        # 创建图片
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：该模式的纯强度分布
        im1 = ax1.imshow(intensity, cmap='hot', origin='upper')
        ax1.set_title(f"{wavelength}nm Layer {layer_num} - Mode {mode_idx} Intensity")
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 右图：该模式的强度 + 所有检测区域（显示波长标记）
        im2 = ax2.imshow(intensity, cmap='hot', origin='upper')
        ax2.set_title(f"{wavelength}nm Layer {layer_num} - Mode {mode_idx} + Detection Regions")
        ax2.axis('off')
        
        # 绘制所有检测区域，使用波长标记显示
        colors = ['cyan', 'lime', 'yellow', 'magenta', 'orange', 'red']
        for i, (x_start, x_end, y_start, y_end) in enumerate(evaluation_regions):
            color = colors[i % len(colors)]
            
            # 绘制矩形框
            rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                               linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            # 添加波长标记标签（仅用于显示）
            center_x = (x_start + x_end) / 2
            center_y = (y_start + y_end) / 2
            
            # 使用波长标记显示
            if i in detector_labels:
                label_text = detector_labels[i]
            else:
                label_text = f'Det{i+1}'
                
            ax2.text(center_x, center_y, label_text, 
                    ha='center', va='center', fontsize=7, 
                    color='white', weight='bold')
        
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        plt.tight_layout()
        
        # 保存图片
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{wavelength}nm_layer_{layer_num}_mode_{mode_idx}_wavelength_display.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ 保存: {filename}")
        
        plt.close()

def evaluate_output(intensity_field, evaluation_regions):
    """
    计算每个检测区域内的平均强度
    
    参数:
        intensity_field: 2D numpy数组，强度分布
        evaluation_regions: 列表，每个元素为 (x_start, x_end, y_start, y_end)
    
    返回:
        region_intensities: 每个区域的平均强度列表
    """
    region_intensities = []
    
    for region in evaluation_regions:
        x_start, x_end, y_start, y_end = region
        x_start, x_end, y_start, y_end = int(x_start), int(x_end), int(y_start), int(y_end)
        
        # 确保索引在有效范围内
        x_start = max(0, min(x_start, intensity_field.shape[1]-1))
        x_end = max(0, min(x_end, intensity_field.shape[1]))
        y_start = max(0, min(y_start, intensity_field.shape[0]-1))
        y_end = max(0, min(y_end, intensity_field.shape[0]))
        
        # 提取区域并计算平均强度
        if x_end > x_start and y_end > y_start:
            region_data = intensity_field[y_start:y_end, x_start:x_end]
            avg_intensity = np.mean(region_data)
            region_intensities.append(avg_intensity)
        else:
            region_intensities.append(0.0)
    
    return np.array(region_intensities)

def calculate_separate_wavelength_matrices(field_data, config):
    """为每个波长创建独立的6×3矩阵"""
    
    print("\n📊 计算独立波长强度矩阵 (每个波长6×3)...")
    
    from label_utils import create_evaluation_regions_by_wavelength
    
    # 获取参数
    num_modes = config.num_modes  # 🔧 使用配置中的正确模式数量
    focus_radius = config.focus_radius
    detectsize = config.detectsize
    num_wavelengths = len(config.wavelengths)
    wavelengths_nm = [int(wl * 1e9) for wl in config.wavelengths]

    # 获取所有波长、层数和模式
    all_wavelengths = sorted(field_data.keys())
    all_layers = sorted(set().union(*[wl_data.keys() for wl_data in field_data.values()]))
    all_modes = sorted(set().union(*[
        set().union(*[layer_data.keys() for layer_data in wl_data.values()])
        for wl_data in field_data.values()
    ]))

    if config.num_modes == 1:
        # 单模式多波长：每个波长1个检测器
        total_detectors = num_wavelengths * 1  # 2个波长 × 1个检测器 = 2个检测器
        print(f"   单模式多波长配置: {num_wavelengths}个波长 × 1个模式 = {total_detectors}个检测区域")
    else:
        # 多模式配置：每个波长多个检测器
        total_detectors = num_wavelengths * num_modes
        print(f"   多模式配置: {num_wavelengths}个波长 × {num_modes}个模式 = {total_detectors}个检测区域")

    
    # 创建波长检测器标签映射
    detector_labels = get_wavelength_detector_labels(wavelengths_nm, num_modes)
    
    print(f"📋 分析计划:")
    print(f"   波长: {all_wavelengths}nm")
    print(f"   层数: {all_layers}")
    print(f"   模式: {all_modes}")
    print(f"   每个波长将创建 6×3 矩阵 (6个检测区域 × 3个输入模式)")
    
    # 结果存储: wavelength -> layer -> matrix
    wavelength_matrices = {}
    wavelength_normalized_matrices = {}
    
    # 处理每个层配置
    for layer_idx, layer_num in enumerate(all_layers):
        print(f"\n🔄 [{layer_idx+1}/{len(all_layers)}] 处理 {layer_num} 层配置...")
        
        # 获取样本数据以确定field_size
        sample_data = None
        for wavelength in all_wavelengths:
            if wavelength in field_data and layer_num in field_data[wavelength]:
                for mode_data_list in field_data[wavelength][layer_num].values():
                    if mode_data_list:
                        sample_data = mode_data_list[0]
                        break
                if sample_data:
                    break
        
        if sample_data is None:
            print(f"  ❌ 无有效数据")
            continue
            
        field_size = sample_data['intensity'].shape[0]
        
        # 创建所有检测区域（6个区域：每个波长3个）
        all_evaluation_regions = []
        wavelength_detector_mapping = {}
        
        for wl_idx, wavelength in enumerate(all_wavelengths):
            # 为这个波长创建检测区域
            wl_evaluation_regions = create_evaluation_regions_by_wavelength(
                field_size, field_size, focus_radius, detectsize, 
                offsets=config.offsets, num_modes=num_modes
            )
            
            start_idx = wl_idx * num_modes
            end_idx = start_idx + num_modes
            wavelength_detector_mapping[wavelength] = (start_idx, end_idx, wl_evaluation_regions)
            
            all_evaluation_regions.extend(wl_evaluation_regions)
            print(f"    ✓ {wavelength}nm: 检测区域 {start_idx}-{end_idx-1}")
        
        # 🎨 生成检测区域可视化 (保留原有功能)
        vis_save_dir = os.path.join(config.save_dir, "detection_visualization_wavelength_display")
        for wavelength in all_wavelengths:
            if wavelength in field_data and layer_num in field_data[wavelength]:
                start_idx, end_idx, wl_evaluation_regions = wavelength_detector_mapping[wavelength]
                visualize_individual_mode_regions_wavelength_display(
                    field_data, wavelength, layer_num, all_evaluation_regions, detector_labels, vis_save_dir)
        
        # 为每个波长创建独立的6×3矩阵
        for target_wavelength in all_wavelengths:
            if target_wavelength not in wavelength_matrices:
                wavelength_matrices[target_wavelength] = {}
                wavelength_normalized_matrices[target_wavelength] = {}
            
            # 🔧 根据实际配置创建矩阵
            if config.num_modes == 1:
                # 单模式：创建 2×1 矩阵（2个波长检测器 × 1个输入模式）
                intensity_matrix = np.zeros((total_detectors, 1))
                print(f"    创建 {total_detectors}×1 矩阵 (单模式多波长)")
            else:
                # 多模式：创建 6×3 矩阵（6个检测区域 × 3个输入模式）
                intensity_matrix = np.zeros((total_detectors, config.num_modes))
                print(f"    创建 {total_detectors}×{config.num_modes} 矩阵 (多模式)")

            
            if target_wavelength not in field_data or layer_num not in field_data[target_wavelength]:
                continue
                
            layer_data = field_data[target_wavelength][layer_num]
            
            print(f"  🎯 处理 {target_wavelength}nm 波长...")
            
            # 对于每个输入模式
            actual_modes = all_modes if config.num_modes > 1 else [1]  # 单模式时只处理模式1

            for input_mode_idx, input_mode in enumerate(actual_modes):
                if input_mode not in layer_data:
                    continue
                    
                mode_data_list = layer_data[input_mode]
                if not mode_data_list:
                    continue
                
                # 计算该波长在所有检测区域的强度
                total_intensities = []
                for data_entry in mode_data_list:
                    intensity = data_entry['intensity']
                    # 在所有检测区域中评估强度
                    region_intensities = evaluate_output(intensity, all_evaluation_regions)
                    total_intensities.append(region_intensities)
                
                if total_intensities:
                    avg_intensities = np.mean(total_intensities, axis=0)
                    
                    # 🔧 填充矩阵：根据实际配置
                    for detector_idx in range(total_detectors):
                        if config.num_modes == 1:
                            # 单模式：只有一列
                            intensity_matrix[detector_idx, 0] = avg_intensities[detector_idx]
                        else:
                            # 多模式：多列
                            if input_mode_idx < config.num_modes:
                                intensity_matrix[detector_idx, input_mode_idx] = avg_intensities[detector_idx]
                    
                    print(f"    ✓ {target_wavelength}nm 模式{input_mode}: 在{total_detectors}个区域的平均强度 {np.mean(avg_intensities):.6f}")

            
            # 保存原始矩阵
            wavelength_matrices[target_wavelength][layer_num] = intensity_matrix.copy()
            
            # 归一化：按列归一化
            normalized_matrix = np.zeros_like(intensity_matrix)
            for col in range(intensity_matrix.shape[1]):
                col_sum = np.sum(intensity_matrix[:, col])
                if col_sum > 0:
                    normalized_matrix[:, col] = intensity_matrix[:, col] / col_sum

            wavelength_normalized_matrices[target_wavelength][layer_num] = normalized_matrix

            print(f"    📊 {target_wavelength}nm 矩阵形状: {intensity_matrix.shape} ({'单模式' if config.num_modes == 1 else '多模式'})")

    
    print(f"\n✅ 独立波长矩阵计算完成！")
    return wavelength_matrices, wavelength_normalized_matrices, all_wavelengths, all_layers, all_modes, detector_labels

def plot_separate_wavelength_matrices(wavelength_normalized_matrices, all_wavelengths, all_layers, all_modes, detector_labels, save_dir):
    """绘制每个波长的独立6×3矩阵"""
    
    print("\n🎨 绘制独立波长强度矩阵...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    num_layers = len(all_layers)
    num_modes = len(all_modes)
    num_wavelengths = len(all_wavelengths)
    total_detectors = num_wavelengths * num_modes
    
    # 🎨 统一字体大小设置
    TITLE_FONTSIZE = 30      # 标题字体
    LABEL_FONTSIZE = 30      # 轴标签字体
    TICK_FONTSIZE = 30      # 刻度标签字体
    VALUE_FONTSIZE = 30      # 数值标注字体
    
    # 为每个波长创建独立的图
    for wavelength in all_wavelengths:
        if wavelength not in wavelength_normalized_matrices:
            continue
            
        wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
        
        # 创建子图
        if num_layers == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 12))  # 🔧 增大图像尺寸
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, num_layers, figsize=(8 * num_layers, 12), constrained_layout=True)  # 🔧 增大图像尺寸
            if num_layers == 1:
                axes = [axes]
        
        for idx, layer_num in enumerate(all_layers):
            if layer_num not in wavelength_normalized_matrices[wavelength]:
                continue
                
            normalized_matrix = wavelength_normalized_matrices[wavelength][layer_num]
            
            ax = axes[idx]
            
            # 绘制热力图
            im = ax.imshow(normalized_matrix, cmap='Oranges', interpolation='nearest', 
                          vmin=0, vmax=1, origin='upper')
            
            # 🎨 统一字体大小
            ax.set_xlabel('Input Mode Index', fontsize=LABEL_FONTSIZE, fontweight='bold')
            ax.set_ylabel('All Detection Regions', fontsize=LABEL_FONTSIZE, fontweight='bold')  
            ax.set_title(f'{wl_nm}nm - {layer_num} Layers', fontsize=TITLE_FONTSIZE, fontweight='bold')

            # 🔧 设置X轴刻度标签
            ax.set_xticks(np.arange(num_modes))
            ax.set_xticklabels([f'Mode{m}' for m in all_modes], fontsize=TICK_FONTSIZE, fontweight='bold') 
            
            # 🔧 设置Y轴刻度标签
            y_labels = [detector_labels.get(i, f'Det{i+1}') for i in range(total_detectors)]
            y_ticks = list(range(total_detectors))
            
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=TICK_FONTSIZE, fontweight='bold')  # 🔧 统一为24
            
            # 添加波长分隔线
            for wl_idx in range(1, num_wavelengths):
                y_pos = wl_idx * num_modes - 0.5
                ax.axhline(y=y_pos, color='white', linewidth=2, linestyle='-')
            
            # 添加数值标注
            for i in range(normalized_matrix.shape[0]):
                for j in range(normalized_matrix.shape[1]):
                    value = normalized_matrix[i, j] * 100
                    ax.text(j, i, f"{value:.1f}", ha='center', va='center', 
                           color='black', fontsize=VALUE_FONTSIZE, weight='bold')  # 🔧 统一为24
        
        # # 添加colorbar
        # if num_layers > 1:
        #     cbar = fig.colorbar(im, ax=axes, shrink=0.6, location='right')
        #     cbar.set_label("Normalized Intensity (%)")
        # else:
        #     cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        #     cbar.set_label("Normalized Intensity (%)")
        

        # 保存图像
        save_path = os.path.join(save_dir, f'intensity_matrix_{wl_nm}nm_separate.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 保存: intensity_matrix_{wl_nm}nm_separate.png")

def save_separate_wavelength_data(wavelength_matrices, wavelength_normalized_matrices, all_wavelengths, all_layers, all_modes, detector_labels, save_dir):
    """保存每个波长的独立矩阵数据"""
    
    print("\n💾 保存独立波长矩阵数据...")
    
    num_modes = len(all_modes)
    num_wavelengths = len(all_wavelengths)
    total_detectors = num_wavelengths * num_modes
    
    for wavelength in all_wavelengths:
        if wavelength not in wavelength_normalized_matrices:
            continue
            
        wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
        
        for layer_num in all_layers:
            if layer_num not in wavelength_normalized_matrices[wavelength]:
                continue
            
            # 保存为numpy文件
            np.save(os.path.join(save_dir, f'intensity_matrix_{wl_nm}nm_{layer_num}layers_raw.npy'), 
                   wavelength_matrices[wavelength][layer_num])
            
            np.save(os.path.join(save_dir, f'intensity_matrix_{wl_nm}nm_{layer_num}layers_normalized.npy'), 
                   wavelength_normalized_matrices[wavelength][layer_num])
            
            # 保存为CSV文件
            row_labels = [detector_labels.get(i, f'Det{i+1}') for i in range(total_detectors)]
            col_labels = [f'Mode{m}' for m in all_modes]
            
            df = pd.DataFrame(
                wavelength_normalized_matrices[wavelength][layer_num],
                index=row_labels,
                columns=col_labels
            )
            
            csv_path = os.path.join(save_dir, f'intensity_matrix_{wl_nm}nm_{layer_num}layers.csv')
            df.to_csv(csv_path, encoding='utf-8-sig')
            
            print(f"  ✅ 保存: intensity_matrix_{wl_nm}nm_{layer_num}layers.csv")

def print_separate_wavelength_summary(wavelength_matrices, wavelength_normalized_matrices, all_wavelengths, all_modes, detector_labels):
    """打印独立波长矩阵摘要"""
    
    print("\n📋 独立波长强度矩阵摘要")
    print("="*60)
    
    num_modes = len(all_modes)
    num_wavelengths = len(all_wavelengths)
    total_detectors = num_wavelengths * num_modes
    
    print(f"配置信息:")
    print(f"  波长: {sorted([int(wl*1e9) for wl in [1310e-9, 1550e-9]])}nm")
    print(f"  每个波长矩阵大小: {total_detectors}×{num_modes} (6个检测区域 × 3个输入模式)")
    print(f"  检测器标记: {', '.join(list(detector_labels.values()))}")
    
    for wavelength in all_wavelengths:
        if wavelength not in wavelength_matrices:
            continue
            
        wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
        
        print(f"\n{wl_nm}nm 波长矩阵:")
        print("-" * 30)
        
        for layer_num in sorted(wavelength_matrices[wavelength].keys()):
            matrix = wavelength_matrices[wavelength][layer_num]
            normalized_matrix = wavelength_normalized_matrices[wavelength][layer_num]
            
            print(f"  {layer_num:2d}层: 矩阵形状 {matrix.shape}")
            print(f"        原始矩阵总和: {np.sum(matrix):.6f}")
            print(f"        对角线元素 (同波长检测器):")
            
            # 显示同波长检测器的对角线元素
            wl_idx = 0 if wavelength == sorted(all_wavelengths)[0] else 1
            start_idx = wl_idx * num_modes
            
            diag_values = []
            for mode_idx in range(num_modes):
                detector_idx = start_idx + mode_idx
                if detector_idx < normalized_matrix.shape[0] and mode_idx < normalized_matrix.shape[1]:
                    diag_values.append(normalized_matrix[detector_idx, mode_idx] * 100)
            
            if diag_values:
                print(f"          {wl_nm}nm检测器: {[f'{v:.1f}%' for v in diag_values]}")

def main_separate_wavelength_analysis(config):
    """主要的独立波长矩阵分析函数"""
    
    print("\n" + "="*60)
    print("独立波长强度矩阵分析 (每个波长6×3矩阵) + 保留检测区域可视化")
    print("="*60)
    
    print(f"📋 分析目标:")
    print(f"   为每个波长创建独立的6×3矩阵")
    print(f"   行：6个检测区域 (1310nm-Det1,2,3 + 1550nm-Det1,2,3)")
    print(f"   列：3个输入模式 (Mode1, Mode2, Mode3)")
    print(f"   显示该波长在所有检测区域的响应强度")
    print(f"   🎨 同时保留原有的检测区域可视化功能")
    
    # 1. 加载数据
    field_data = load_field_data_multiwavelength(config)
    
    if not field_data:
        print("❌ 未找到有效的光场数据")
        return None, None
    
    # 2. 计算独立波长矩阵 (包含检测区域可视化)
    wavelength_matrices, wavelength_normalized_matrices, all_wavelengths, all_layers, all_modes, detector_labels = \
        calculate_separate_wavelength_matrices(field_data, config)
    
    if not wavelength_matrices:
        print("❌ 独立波长矩阵计算失败")
        return None, None
    
    # 3. 创建保存目录
    save_dir = os.path.join(config.save_dir, "separate_wavelength_matrices")
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 绘制矩阵
    plot_separate_wavelength_matrices(
        wavelength_normalized_matrices, all_wavelengths, all_layers, all_modes, detector_labels, save_dir)
    
    # 5. 保存数据
    save_separate_wavelength_data(
        wavelength_matrices, wavelength_normalized_matrices, all_wavelengths, all_layers, all_modes, detector_labels, save_dir)
    
    # 6. 打印摘要
    print_separate_wavelength_summary(
        wavelength_matrices, wavelength_normalized_matrices, all_wavelengths, all_modes, detector_labels)
    
    print(f"\n🎉 独立波长强度矩阵分析完成！")
    print(f"📁 结果保存在: {save_dir}")
    print(f"📊 查看矩阵图像: intensity_matrix_1310nm_separate.png, intensity_matrix_1550nm_separate.png")
    print(f"📋 查看数据表: intensity_matrix_*nm_*layers.csv")
    print(f"🎨 查看检测区域可视化: detection_visualization_wavelength_display/")
    
    return wavelength_matrices, wavelength_normalized_matrices

# 在您的主程序最后添加这个分析
print("\n" + "="*60)
print("执行独立波长强度矩阵分析 + 保留检测区域可视化")
print("="*60)

try:
    wavelength_matrices, wavelength_normalized_matrices = main_separate_wavelength_analysis(config)
    
    if wavelength_matrices is not None:
        print("✅ 独立波长强度矩阵分析成功完成！")
        
        # 显示简要结果
        print("\n📊 生成的矩阵:")
        for wavelength in sorted(wavelength_matrices.keys()):
            wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
            print(f"   {wl_nm}nm:")
            for layer_num, matrix in wavelength_matrices[wavelength].items():
                print(f"     {layer_num}层: {matrix.shape} 矩阵, 总强度: {np.sum(matrix):.6f}")
        
        print("\n📋 矩阵含义:")
        print("   每个6×3矩阵显示:")
        print("   - 行：6个检测区域 (1310nm-Det1,2,3 + 1550nm-Det1,2,3)")
        print("   - 列：3个输入模式 (Mode1, Mode2, Mode3)")
        print("   - 值：该波长在对应检测区域的归一化强度百分比")
        
        print("\n📁 生成的文件:")
        print("   📊 独立波长矩阵:")
        print("      - intensity_matrix_1310nm_separate.png")
        print("      - intensity_matrix_1550nm_separate.png")
        print("      - intensity_matrix_*nm_*layers.csv")
        
        print("   🎨 检测区域可视化 (保留原有功能):")
        vis_dir = os.path.join(config.save_dir, "detection_visualization_wavelength_display")
        if os.path.exists(vis_dir):
            png_files = [f for f in os.listdir(vis_dir) if f.endswith('.png')]
            for png_file in sorted(png_files)[:6]:  # 显示前6个文件
                print(f"      - {png_file}")
            if len(png_files) > 6:
                print(f"      - ... 还有 {len(png_files)-6} 个文件")
        
    else:
        print("❌ 分析失败，请检查数据文件")
        
except Exception as e:
    print(f"❌ 分析过程中出现错误: {e}")
    import traceback
    traceback.print_exc()

print(f"\n总运行时间: {time.time() - start_time:.2f} 秒")
print("🎉 独立波长矩阵分析 + 检测区域可视化完成！")


