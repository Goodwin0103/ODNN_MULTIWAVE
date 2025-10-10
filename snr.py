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
print("多模式多波长光场调制系统 - SNR分析版 (多波长支持 - 波长标记显示版)")
print("=" * 60)

# ===== 创建增强配置 =====

# 基本参数
num_modes = 3                                    # 模式数量
# wavelengths = np.array([1550e-9])      # 波长列表(m)
wavelengths = np.array([1310e-9, 1550e-9])      # 波长列表(m) 
base_wavelength_idx = 1                          # 基准波长索引

# 空间参数
field_size = 50                                  # 场大小(像素)
layer_size = 300                                 # 层大小(像素)
focus_radius = 10                                # 焦点半径(像素)
detectsize = 30                                  # 检测区域大小(像素)

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

# ===== SNR分析专用参数 =====
snr_thresholds = [6.0, 10.0, 15.0, 20.0]        # SNR阈值 (dB)
background_exclusion_radius = 15                  # 背景区域排除半径（避免目标区域影响）
min_background_samples = 100                      # 最小背景样本数

# 保存参数
save_dir = f"ODNN_WAVE/results/{num_modes}_mode_{len(wavelengths)}_wl_basewl_{wavelengths[base_wavelength_idx]}_z_prop_{z_prop}_focus_{focus_radius}/"
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
print(f"SNR阈值: {snr_thresholds} dB")
print(f"背景排除半径: {background_exclusion_radius} 像素")

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

def create_background_mask(field_size, target_regions, exclusion_radius):
    """创建背景区域掩码，排除目标检测区域及其周围"""
    
    # 初始化背景掩码（True=背景区域）
    background_mask = np.ones((field_size, field_size), dtype=bool)
    
    # 创建坐标网格
    y_coords, x_coords = np.ogrid[:field_size, :field_size]
    
    # 排除每个目标区域及其周围区域
    for region in target_regions:
        x_start, x_end, y_start, y_end = region
        center_x = (x_start + x_end) / 2
        center_y = (y_start + y_end) / 2
        
        # 计算到中心的距离
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # 排除半径内的区域
        background_mask[distances <= exclusion_radius] = False
    
    return background_mask

def calculate_detection_snr(intensity_field, target_regions, background_mask):
    """计算目标检测区域相对于背景区域的检测SNR"""
    
    # 提取背景区域数据
    background_data = intensity_field[background_mask]
    
    if len(background_data) < min_background_samples:
        print(f"⚠️ 背景样本不足: {len(background_data)} < {min_background_samples}")
        return []
    
    # 计算背景统计量
    bg_mean = np.mean(background_data)
    bg_std = np.std(background_data)
    
    # 避免除零
    if bg_std <= 0:
        bg_std = 1e-12
    
    snr_results = []
    
    # 分析每个目标区域
    for region_idx, region in enumerate(target_regions):
        x_start, x_end, y_start, y_end = region
        x_start, x_end, y_start, y_end = int(x_start), int(x_end), int(y_start), int(y_end)
        
        # 确保索引在有效范围内
        x_start = max(0, min(x_start, intensity_field.shape[1]-1))
        x_end = max(0, min(x_end, intensity_field.shape[1]))
        y_start = max(0, min(y_start, intensity_field.shape[0]-1))
        y_end = max(0, min(y_end, intensity_field.shape[0]))
        
        # 提取目标区域数据
        if x_end > x_start and y_end > y_start:
            target_data = intensity_field[y_start:y_end, x_start:x_end]
            
            if target_data.size == 0:
                snr_results.append(0.0)
                continue
            
            # 计算目标区域统计量
            target_mean = np.mean(target_data)
            
            # 计算检测SNR - 基于统计检测理论
            detection_metric = (target_mean - bg_mean) / bg_std
            detection_snr_db = 20 * np.log10(abs(detection_metric)) if detection_metric != 0 else -np.inf
            
            snr_results.append(detection_snr_db if np.isfinite(detection_snr_db) else 0.0)
        else:
            snr_results.append(0.0)
    
    return np.array(snr_results)

def visualize_individual_mode_regions_snr_display(field_data, wavelength, layer_num, evaluation_regions, detector_labels, save_dir=None):
    """为每个模式单独可视化检测区域 - SNR显示版本（保持原有区域创建逻辑）"""
    
    print(f"   🎨 为{wavelength}nm第{layer_num}层的每个模式生成SNR可视化图片...")
    
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
        
        # 创建背景掩码
        background_mask = create_background_mask(intensity.shape[0], evaluation_regions, background_exclusion_radius)
        
        # 计算SNR
        snr_values = calculate_detection_snr(intensity, evaluation_regions, background_mask)
        
        # 创建图片
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：该模式的纯强度分布
        im1 = ax1.imshow(intensity, cmap='hot', origin='upper')
        ax1.set_title(f"{wavelength}nm Layer {layer_num} - Mode {mode_idx} Intensity")
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 右图：该模式的强度 + 所有检测区域（显示SNR值）
        im2 = ax2.imshow(intensity, cmap='hot', origin='upper')
        ax2.set_title(f"{wavelength}nm Layer {layer_num} - Mode {mode_idx} + Detection SNR (dB)")
        ax2.axis('off')
        
        # 绘制所有检测区域，显示SNR值
        colors = ['cyan', 'lime', 'yellow', 'magenta', 'orange', 'red']
        for i, (x_start, x_end, y_start, y_end) in enumerate(evaluation_regions):
            color = colors[i % len(colors)]
            
            # 绘制矩形框
            rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                               linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            # 添加SNR值标签
            center_x = (x_start + x_end) / 2
            center_y = (y_start + y_end) / 2
            
            # 使用波长标记和SNR值显示
            if i in detector_labels:
                label_text = detector_labels[i]
            else:
                label_text = f'Det{i+1}'
            
            snr_value = snr_values[i] if i < len(snr_values) else 0.0
            display_text = f'{label_text}\n{snr_value:.1f}dB'
                
            ax2.text(center_x, center_y, display_text, 
                    ha='center', va='center', fontsize=6, 
                    color='white', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        plt.tight_layout()
        
        # 保存图片
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{wavelength}nm_layer_{layer_num}_mode_{mode_idx}_snr_display.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ 保存: {filename}")
        
        plt.close()

def calculate_separate_wavelength_snr_matrices(field_data, config):
    """为每个波长创建独立的6×3 SNR矩阵"""
    
    print("\n📊 计算独立波长SNR矩阵 (每个波长6×3)...")
    
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
    
    print(f"📋 SNR分析计划:")
    print(f"   波长: {all_wavelengths}nm")
    print(f"   层数: {all_layers}")
    print(f"   模式: {all_modes}")
    print(f"   每个波长将创建 6×3 SNR矩阵 (6个检测区域 × 3个输入模式)")
    print(f"   SNR阈值: {snr_thresholds} dB")
    
    # 结果存储: wavelength -> layer -> matrix
    wavelength_snr_matrices = {}
    wavelength_snr_raw_matrices = {}
    
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
        
        # 🎨 生成检测区域SNR可视化 (保留原有功能)
        vis_save_dir = os.path.join(config.save_dir, "detection_snr_visualization_wavelength_display")
        for wavelength in all_wavelengths:
            if wavelength in field_data and layer_num in field_data[wavelength]:
                start_idx, end_idx, wl_evaluation_regions = wavelength_detector_mapping[wavelength]
                visualize_individual_mode_regions_snr_display(
                    field_data, wavelength, layer_num, all_evaluation_regions, detector_labels, vis_save_dir)
        
        # 为每个波长创建独立的6×3 SNR矩阵
        for target_wavelength in all_wavelengths:
            if target_wavelength not in wavelength_snr_matrices:
                wavelength_snr_matrices[target_wavelength] = {}
                wavelength_snr_raw_matrices[target_wavelength] = {}
            
            # 🔧 根据实际配置创建矩阵
            if config.num_modes == 1:
                # 单模式：创建 2×1 矩阵（2个波长检测器 × 1个输入模式）
                snr_matrix = np.zeros((total_detectors, 1))
                print(f"    创建 {total_detectors}×1 SNR矩阵 (单模式多波长)")
            else:
                # 多模式：创建 6×3 矩阵（6个检测区域 × 3个输入模式）
                snr_matrix = np.zeros((total_detectors, config.num_modes))
                print(f"    创建 {total_detectors}×{config.num_modes} SNR矩阵 (多模式)")

            
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
                
                # 计算该波长在所有检测区域的SNR
                total_snr_values = []
                for data_entry in mode_data_list:
                    intensity = data_entry['intensity']
                    
                    # 创建背景掩码
                    background_mask = create_background_mask(intensity.shape[0], all_evaluation_regions, background_exclusion_radius)
                    
                    # 在所有检测区域中评估SNR
                    region_snr_values = calculate_detection_snr(intensity, all_evaluation_regions, background_mask)
                    total_snr_values.append(region_snr_values)
                
                if total_snr_values:
                    avg_snr_values = np.mean(total_snr_values, axis=0)
                    
                    # 🔧 填充矩阵：根据实际配置
                    for detector_idx in range(total_detectors):
                        if config.num_modes == 1:
                            # 单模式：只有一列
                            snr_matrix[detector_idx, 0] = avg_snr_values[detector_idx]
                        else:
                            # 多模式：多列
                            if input_mode_idx < config.num_modes:
                                snr_matrix[detector_idx, input_mode_idx] = avg_snr_values[detector_idx]
                    
                    print(f"    ✓ {target_wavelength}nm 模式{input_mode}: 在{total_detectors}个区域的平均SNR {np.mean(avg_snr_values):.2f}dB")

            
            # 保存原始SNR矩阵
            wavelength_snr_raw_matrices[target_wavelength][layer_num] = snr_matrix.copy()
            
            # 处理SNR矩阵（可选择不同的处理方式）
            processed_matrix = snr_matrix.copy()
            
            # 将负无穷值和NaN替换为0
            processed_matrix[~np.isfinite(processed_matrix)] = 0.0
            
            wavelength_snr_matrices[target_wavelength][layer_num] = processed_matrix

            print(f"    📊 {target_wavelength}nm SNR矩阵形状: {snr_matrix.shape} ({'单模式' if config.num_modes == 1 else '多模式'})")
            print(f"    📈 SNR范围: {np.min(processed_matrix):.2f} 到 {np.max(processed_matrix):.2f} dB")

    
    print(f"\n✅ 独立波长SNR矩阵计算完成！")
    return wavelength_snr_matrices, wavelength_snr_raw_matrices, all_wavelengths, all_layers, all_modes, detector_labels

def plot_separate_wavelength_snr_matrices(wavelength_snr_matrices, all_wavelengths, all_layers, all_modes, detector_labels, save_dir):
    """绘制每个波长的独立6×3 SNR矩阵"""
    
    print("\n🎨 绘制独立波长SNR矩阵...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    num_layers = len(all_layers)
    num_modes = len(all_modes)
    num_wavelengths = len(all_wavelengths)
    total_detectors = num_wavelengths * num_modes
    
    # 为每个波长创建独立的图
    for wavelength in all_wavelengths:
        if wavelength not in wavelength_snr_matrices:
            continue
            
        wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
        
        # 创建子图
        if num_layers == 1:
            fig, ax = plt.subplots(1, 1, figsize=(6, 8))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 10), constrained_layout=True)
            if num_layers == 1:
                axes = [axes]
        
        for idx, layer_num in enumerate(all_layers):
            if layer_num not in wavelength_snr_matrices[wavelength]:
                continue
                
            snr_matrix = wavelength_snr_matrices[wavelength][layer_num]
            
            ax = axes[idx]
            
            # 计算colorbar范围
            finite_values = snr_matrix[np.isfinite(snr_matrix)]
            if len(finite_values) > 0:
                vmin = max(0, np.percentile(finite_values, 5))  # SNR最小值设为0
                vmax = np.percentile(finite_values, 95)
                if vmax - vmin < 1:
                    vmax = vmin + 10
            else:
                vmin, vmax = 0, 20
            
            # 绘制热力图
            im = ax.imshow(snr_matrix, cmap='RdYlBu_r', interpolation='nearest', 
                          vmin=vmin, vmax=vmax, origin='upper')
            
            ax.set_xlabel('Input Mode Index', fontsize=14)
            ax.set_ylabel('All Detection Regions', fontsize=14)  
            ax.set_title(f'{wl_nm}nm - {layer_num} Layers\nDetection SNR (dB)', fontsize=16)

            # 🔧 设置X轴刻度标签
            ax.set_xticks(np.arange(num_modes))
            ax.set_xticklabels([f'Mode{m}' for m in all_modes], fontsize=12) 
            
            # 🔧 设置Y轴刻度标签
            y_labels = [detector_labels.get(i, f'Det{i+1}') for i in range(total_detectors)]
            y_ticks = list(range(total_detectors))
            
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=12)
            
            # 添加波长分隔线
            for wl_idx in range(1, num_wavelengths):
                y_pos = wl_idx * num_modes - 0.5
                ax.axhline(y=y_pos, color='white', linewidth=2, linestyle='-')
            
            # 添加数值标注
            for i in range(snr_matrix.shape[0]):
                for j in range(snr_matrix.shape[1]):
                    value = snr_matrix[i, j]
                    if np.isfinite(value):
                        ax.text(j, i, f"{value:.1f}", ha='center', va='center', 
                               color='black', fontsize=10, weight='bold')
                    else:
                        ax.text(j, i, "N/A", ha='center', va='center', 
                               color='gray', fontsize=9)
            
            # 添加SNR阈值等高线
            for threshold in snr_thresholds[:2]:  # 只显示前两个阈值
                try:
                    contour = ax.contour(snr_matrix, levels=[threshold], colors='red', 
                                       linewidths=1.5, alpha=0.8, linestyles='--')
                    ax.clabel(contour, inline=True, fontsize=8, fmt=f'{threshold:.0f}dB')
                except:
                    pass
        
        # 添加colorbar
        if num_layers > 1:
            cbar = fig.colorbar(im, ax=axes, shrink=0.6, location='right')
            cbar.set_label("Detection SNR (dB)", fontsize=12)
        else:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Detection SNR (dB)", fontsize=12)
        
        plt.suptitle(f"{wl_nm}nm Wavelength Detection SNR Matrix (6 Detectors × 3 Input Modes)\n"
                    f"Shows {wl_nm}nm detection SNR in all 6 detection regions", 
                    fontsize=14)
        
        # 保存图像
        save_path = os.path.join(save_dir, f'snr_matrix_{wl_nm}nm_separate.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 保存: snr_matrix_{wl_nm}nm_separate.png")

def save_separate_wavelength_snr_data(wavelength_snr_matrices, wavelength_snr_raw_matrices, all_wavelengths, all_layers, all_modes, detector_labels, save_dir):
    """保存每个波长的独立SNR矩阵数据"""
    
    print("\n💾 保存独立波长SNR矩阵数据...")
    
    num_modes = len(all_modes)
    num_wavelengths = len(all_wavelengths)
    total_detectors = num_wavelengths * num_modes
    
    for wavelength in all_wavelengths:
        if wavelength not in wavelength_snr_matrices:
            continue
            
        wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
        
        for layer_num in all_layers:
            if layer_num not in wavelength_snr_matrices[wavelength]:
                continue
            
            # 保存为numpy文件
            np.save(os.path.join(save_dir, f'snr_matrix_{wl_nm}nm_{layer_num}layers_raw.npy'), 
                   wavelength_snr_raw_matrices[wavelength][layer_num])
            
            np.save(os.path.join(save_dir, f'snr_matrix_{wl_nm}nm_{layer_num}layers_processed.npy'), 
                   wavelength_snr_matrices[wavelength][layer_num])
            
            # 保存为CSV文件
            row_labels = [detector_labels.get(i, f'Det{i+1}') for i in range(total_detectors)]
            col_labels = [f'Mode{m}' for m in all_modes]
            
            df = pd.DataFrame(
                wavelength_snr_matrices[wavelength][layer_num],
                index=row_labels,
                columns=col_labels
            )
            
            csv_path = os.path.join(save_dir, f'snr_matrix_{wl_nm}nm_{layer_num}layers.csv')
            df.to_csv(csv_path, encoding='utf-8-sig')
            
            print(f"  ✅ 保存: snr_matrix_{wl_nm}nm_{layer_num}layers.csv")

def create_snr_threshold_analysis(wavelength_snr_matrices, all_wavelengths, all_layers, detector_labels, save_dir):
    """创建SNR阈值达标分析图"""
    
    print("\n📊 创建SNR阈值达标分析...")
    
    for wavelength in all_wavelengths:
        if wavelength not in wavelength_snr_matrices:
            continue
            
        wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
        
        # 创建阈值达标分析图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 收集所有层的数据
        all_layer_data = []
        layer_labels = []
        
        for layer_num in sorted(all_layers):
            if layer_num in wavelength_snr_matrices[wavelength]:
                snr_matrix = wavelength_snr_matrices[wavelength][layer_num]
                finite_values = snr_matrix[np.isfinite(snr_matrix)]
                if len(finite_values) > 0:
                    all_layer_data.append(finite_values)
                    layer_labels.append(f'{layer_num} Layers')
        
        if not all_layer_data:
            continue
        
        # 左图：SNR分布直方图
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_layer_data)))
        for i, (data, label) in enumerate(zip(all_layer_data, layer_labels)):
            ax1.hist(data, bins=20, alpha=0.7, label=label, color=colors[i])
        
        # 添加阈值线
        for threshold in snr_thresholds:
            ax1.axvline(x=threshold, color='red', linestyle='--', alpha=0.8)
            ax1.text(threshold, ax1.get_ylim()[1]*0.9, f'{threshold}dB', 
                    rotation=90, ha='right', va='top', color='red')
        
        ax1.set_xlabel('Detection SNR (dB)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{wl_nm}nm SNR Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：阈值达标率
        threshold_achievement = []
        
        for layer_idx, (data, label) in enumerate(zip(all_layer_data, layer_labels)):
            layer_achievements = []
            for threshold in snr_thresholds:
                achievement_rate = np.sum(data >= threshold) / len(data) * 100
                layer_achievements.append(achievement_rate)
            threshold_achievement.append(layer_achievements)
        
        # 绘制达标率柱状图
        x = np.arange(len(snr_thresholds))
        width = 0.35
        
        for i, (achievements, label) in enumerate(zip(threshold_achievement, layer_labels)):
            offset = (i - len(layer_labels)/2 + 0.5) * width
            bars = ax2.bar(x + offset, achievements, width, label=label, color=colors[i], alpha=0.8)
            
            # 添加数值标注
            for bar, value in zip(bars, achievements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('SNR Threshold (dB)')
        ax2.set_ylabel('Achievement Rate (%)')
        ax2.set_title(f'{wl_nm}nm SNR Threshold Achievement')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{t}dB' for t in snr_thresholds])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)
        
        plt.suptitle(f'{wl_nm}nm Wavelength Detection SNR Analysis', fontsize=16)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(save_dir, f'snr_threshold_analysis_{wl_nm}nm.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 保存: snr_threshold_analysis_{wl_nm}nm.png")

def print_separate_wavelength_snr_summary(wavelength_snr_matrices, wavelength_snr_raw_matrices, all_wavelengths, all_modes, detector_labels):
    """打印独立波长SNR矩阵摘要"""
    
    print("\n📋 独立波长Detection SNR矩阵摘要")
    print("="*60)
    
    num_modes = len(all_modes)
    num_wavelengths = len(all_wavelengths)
    total_detectors = num_wavelengths * num_modes
    
    print(f"配置信息:")
    print(f"  波长: {sorted([int(wl*1e9) for wl in [1310e-9, 1550e-9]])}nm")
    print(f"  每个波长矩阵大小: {total_detectors}×{num_modes} (6个检测区域 × 3个输入模式)")
    print(f"  检测器标记: {', '.join(list(detector_labels.values()))}")
    print(f"  SNR阈值: {snr_thresholds} dB")
    print(f"  背景排除半径: {background_exclusion_radius} 像素")
    
    for wavelength in all_wavelengths:
        if wavelength not in wavelength_snr_matrices:
            continue
            
        wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
        
        print(f"\n{wl_nm}nm 波长Detection SNR矩阵:")
        print("-" * 30)
        
        for layer_num in sorted(wavelength_snr_matrices[wavelength].keys()):
            snr_matrix = wavelength_snr_matrices[wavelength][layer_num]
            raw_matrix = wavelength_snr_raw_matrices[wavelength][layer_num]
            
            finite_values = snr_matrix[np.isfinite(snr_matrix)]
            
            print(f"  {layer_num:2d}层: 矩阵形状 {snr_matrix.shape}")
            
            if len(finite_values) > 0:
                print(f"        Detection SNR统计:")
                print(f"          平均值: {np.mean(finite_values):.2f} dB")
                print(f"          标准差: {np.std(finite_values):.2f} dB")
                print(f"          最大值: {np.max(finite_values):.2f} dB")
                print(f"          最小值: {np.min(finite_values):.2f} dB")
                
                # SNR阈值达标统计
                print(f"        阈值达标率:")
                for threshold in snr_thresholds:
                    achievement_count = np.sum(finite_values >= threshold)
                    achievement_rate = achievement_count / len(finite_values) * 100
                    print(f"          ≥{threshold}dB: {achievement_rate:.1f}% ({achievement_count}/{len(finite_values)})")
                
                # 显示同波长检测器的对角线元素
                print(f"        对角线Detection SNR (同波长检测器):")
                wl_idx = 0 if wavelength == sorted(all_wavelengths)[0] else 1
                start_idx = wl_idx * num_modes
                
                diag_values = []
                for mode_idx in range(num_modes):
                    detector_idx = start_idx + mode_idx
                    if detector_idx < snr_matrix.shape[0] and mode_idx < snr_matrix.shape[1]:
                        diag_value = snr_matrix[detector_idx, mode_idx]
                        if np.isfinite(diag_value):
                            diag_values.append(diag_value)
                        else:
                            diag_values.append(0.0)
                
                if diag_values:
                    print(f"          {wl_nm}nm检测器: {[f'{v:.1f}dB' for v in diag_values]}")
            else:
                print(f"        ⚠️ 无有效Detection SNR数据")

def create_comparative_snr_analysis(wavelength_snr_matrices, all_wavelengths, all_layers, detector_labels, save_dir):
    """创建波长间比较SNR分析"""
    
    print("\n🔍 创建波长间比较Detection SNR分析...")
    
    if len(all_wavelengths) < 2:
        print("  ⚠️ 波长数量不足，跳过比较分析")
        return
    
    # 创建比较分析图
    for layer_num in all_layers:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 收集两个波长的数据
        wl_data = {}
        wl_matrices = {}
        
        for wavelength in all_wavelengths:
            if wavelength in wavelength_snr_matrices and layer_num in wavelength_snr_matrices[wavelength]:
                wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
                matrix = wavelength_snr_matrices[wavelength][layer_num]
                finite_values = matrix[np.isfinite(matrix)]
                
                wl_data[wl_nm] = finite_values
                wl_matrices[wl_nm] = matrix
        
        if len(wl_data) < 2:
            plt.close()
            continue
        
        wavelengths_nm = sorted(wl_data.keys())
        
        # 1. SNR分布比较 (左上)
        for wl_nm in wavelengths_nm:
            ax1.hist(wl_data[wl_nm], bins=20, alpha=0.7, label=f'{wl_nm}nm', density=True)
        
        ax1.set_xlabel('Detection SNR (dB)')
        ax1.set_ylabel('Density')
        ax1.set_title('SNR Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加阈值线
        for threshold in snr_thresholds[:2]:
            ax1.axvline(x=threshold, color='red', linestyle='--', alpha=0.5)
        
        # 2. 箱线图比较 (右上)
        box_data = [wl_data[wl_nm] for wl_nm in wavelengths_nm]
        box_labels = [f'{wl_nm}nm' for wl_nm in wavelengths_nm]
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Detection SNR (dB)')
        ax2.set_title('SNR Distribution Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # 添加阈值线
        for threshold in snr_thresholds[:2]:
            ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
            ax2.text(len(wavelengths_nm), threshold, f'{threshold}dB', 
                    ha='left', va='center', color='red', fontsize=9)
        
        # 3. 阈值达标率比较 (左下)
        x = np.arange(len(snr_thresholds))
        width = 0.35
        
        for i, wl_nm in enumerate(wavelengths_nm):
            achievements = []
            for threshold in snr_thresholds:
                rate = np.sum(wl_data[wl_nm] >= threshold) / len(wl_data[wl_nm]) * 100
                achievements.append(rate)
            
            offset = (i - 0.5) * width
            bars = ax3.bar(x + offset, achievements, width, 
                          label=f'{wl_nm}nm', alpha=0.8, color=colors[i])
            
            # 添加数值标注
            for bar, value in zip(bars, achievements):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax3.set_xlabel('SNR Threshold (dB)')
        ax3.set_ylabel('Achievement Rate (%)')
        ax3.set_title('SNR Threshold Achievement Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{t}dB' for t in snr_thresholds])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 105)
        
        # 4. 对角线元素比较 (右下)
        num_modes = len([m for m in [1, 2, 3]])  # 假设3个模式
        
        diag_comparison = {}
        for wl_nm in wavelengths_nm:
            matrix = wl_matrices[wl_nm]
            wl_idx = 0 if wl_nm == min(wavelengths_nm) else 1
            start_idx = wl_idx * num_modes
            
            diag_values = []
            for mode_idx in range(num_modes):
                detector_idx = start_idx + mode_idx
                if detector_idx < matrix.shape[0] and mode_idx < matrix.shape[1]:
                    value = matrix[detector_idx, mode_idx]
                    diag_values.append(value if np.isfinite(value) else 0.0)
                else:
                    diag_values.append(0.0)
            
            diag_comparison[wl_nm] = diag_values
        
        # 绘制对角线元素比较
        mode_labels = [f'Mode{i+1}' for i in range(num_modes)]
        x_pos = np.arange(len(mode_labels))
        
        for i, wl_nm in enumerate(wavelengths_nm):
            offset = (i - 0.5) * width
            bars = ax4.bar(x_pos + offset, diag_comparison[wl_nm], width, 
                          label=f'{wl_nm}nm', alpha=0.8, color=colors[i])
            
            # 添加数值标注
            for bar, value in zip(bars, diag_comparison[wl_nm]):
                height = bar.get_height()
                if height > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax4.set_xlabel('Input Mode')
        ax4.set_ylabel('Detection SNR (dB)')
        ax4.set_title('Diagonal Elements Comparison\n(Same-wavelength detectors)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(mode_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Wavelength Comparison Detection SNR Analysis - {layer_num} Layers', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(save_dir, f'wavelength_comparison_snr_{layer_num}layers.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 保存: wavelength_comparison_snr_{layer_num}layers.png")

def main_separate_wavelength_snr_analysis(config):
    """主要的独立波长SNR矩阵分析函数"""
    
    print("\n" + "="*60)
    print("独立波长Detection SNR矩阵分析 (每个波长6×3矩阵) + 保留检测区域可视化")
    print("="*60)
    
    print(f"📋 SNR分析目标:")
    print(f"   为每个波长创建独立的6×3 Detection SNR矩阵")
    print(f"   行：6个检测区域 (1310nm-Det1,2,3 + 1550nm-Det1,2,3)")
    print(f"   列：3个输入模式 (Mode1, Mode2, Mode3)")
    print(f"   显示该波长在所有检测区域的Detection SNR (dB)")
    print(f"   🎨 同时保留原有的检测区域可视化功能")
    print(f"   📊 SNR阈值: {snr_thresholds} dB")
    print(f"   🔍 背景排除半径: {background_exclusion_radius} 像素")
    
    # 1. 加载数据
    field_data = load_field_data_multiwavelength(config)
    
    if not field_data:
        print("❌ 未找到有效的光场数据")
        return None, None
    
    # 2. 计算独立波长SNR矩阵 (包含检测区域可视化)
    wavelength_snr_matrices, wavelength_snr_raw_matrices, all_wavelengths, all_layers, all_modes, detector_labels = \
        calculate_separate_wavelength_snr_matrices(field_data, config)
    
    if not wavelength_snr_matrices:
        print("❌ 独立波长SNR矩阵计算失败")
        return None, None
    
    # 3. 创建保存目录
    save_dir = os.path.join(config.save_dir, "separate_wavelength_snr_matrices")
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 绘制SNR矩阵
    plot_separate_wavelength_snr_matrices(
        wavelength_snr_matrices, all_wavelengths, all_layers, all_modes, detector_labels, save_dir)
    
    # 5. 创建SNR阈值分析
    create_snr_threshold_analysis(wavelength_snr_matrices, all_wavelengths, all_layers, detector_labels, save_dir)
    
    # 6. 创建波长比较分析
    create_comparative_snr_analysis(wavelength_snr_matrices, all_wavelengths, all_layers, detector_labels, save_dir)
    
    # 7. 保存数据
    save_separate_wavelength_snr_data(
        wavelength_snr_matrices, wavelength_snr_raw_matrices, all_wavelengths, all_layers, all_modes, detector_labels, save_dir)
    
    # 8. 打印摘要
    print_separate_wavelength_snr_summary(
        wavelength_snr_matrices, wavelength_snr_raw_matrices, all_wavelengths, all_modes, detector_labels)
    
    print(f"\n🎉 独立波长Detection SNR矩阵分析完成！")
    print(f"📁 结果保存在: {save_dir}")
    print(f"📊 查看SNR矩阵图像: snr_matrix_1310nm_separate.png, snr_matrix_1550nm_separate.png")
    print(f"📈 查看阈值分析: snr_threshold_analysis_*nm.png")
    print(f"🔍 查看波长比较: wavelength_comparison_snr_*layers.png")
    print(f"📋 查看数据表: snr_matrix_*nm_*layers.csv")
    print(f"🎨 查看检测区域可视化: detection_snr_visualization_wavelength_display/")
    
    return wavelength_snr_matrices, wavelength_snr_raw_matrices

# 在您的主程序最后添加这个分析
print("\n" + "="*60)
print("执行独立波长Detection SNR矩阵分析 + 保留检测区域可视化")
print("="*60)

try:
    wavelength_snr_matrices, wavelength_snr_raw_matrices = main_separate_wavelength_snr_analysis(config)
    
    if wavelength_snr_matrices is not None:
        print("✅ 独立波长Detection SNR矩阵分析成功完成！")
        
        # 显示简要结果
        print("\n📊 生成的SNR矩阵:")
        for wavelength in sorted(wavelength_snr_matrices.keys()):
            wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
            print(f"   {wl_nm}nm:")
            for layer_num, matrix in wavelength_snr_matrices[wavelength].items():
                finite_values = matrix[np.isfinite(matrix)]
                if len(finite_values) > 0:
                    avg_snr = np.mean(finite_values)
                    max_snr = np.max(finite_values)
                    print(f"     {layer_num}层: {matrix.shape} 矩阵, 平均SNR: {avg_snr:.2f}dB, 最大SNR: {max_snr:.2f}dB")
                else:
                    print(f"     {layer_num}层: {matrix.shape} 矩阵, 无有效SNR数据")
        
        print("\n📋 SNR矩阵含义:")
        print("   每个6×3矩阵显示:")
        print("   - 行：6个检测区域 (1310nm-Det1,2,3 + 1550nm-Det1,2,3)")
        print("   - 列：3个输入模式 (Mode1, Mode2, Mode3)")
        print("   - 值：该波长在对应检测区域的Detection SNR (dB)")
        print("   - Detection SNR = 20*log10(|(目标强度-背景均值)/背景标准差|)")
        
        print("\n📁 生成的文件:")
        print("   📊 独立波长SNR矩阵:")
        print("      - snr_matrix_1310nm_separate.png")
        print("      - snr_matrix_1550nm_separate.png")
        print("      - snr_matrix_*nm_*layers.csv")
        
        print("   📈 SNR分析图:")
        print("      - snr_threshold_analysis_1310nm.png")
        print("      - snr_threshold_analysis_1550nm.png")
        print("      - wavelength_comparison_snr_*layers.png")
        
        print("   🎨 检测区域SNR可视化 (保留原有功能):")
        vis_dir = os.path.join(config.save_dir, "detection_snr_visualization_wavelength_display")
        if os.path.exists(vis_dir):
            png_files = [f for f in os.listdir(vis_dir) if f.endswith('.png')]
            for png_file in sorted(png_files)[:6]:  # 显示前6个文件
                print(f"      - {png_file}")
            if len(png_files) > 6:
                print(f"      - ... 还有 {len(png_files)-6} 个文件")
        
        # 显示SNR统计摘要
        print(f"\n📊 Detection SNR统计摘要:")
        print(f"   SNR阈值: {snr_thresholds} dB")
        print(f"   背景排除半径: {background_exclusion_radius} 像素")
        print(f"   最小背景样本数: {min_background_samples}")
        
    else:
        print("❌ SNR分析失败，请检查数据文件")
        
except Exception as e:
    print(f"❌ SNR分析过程中出现错误: {e}")
    import traceback
    traceback.print_exc()

print(f"\n总运行时间: {time.time() - start_time:.2f} 秒")
print("🎉 独立波长Detection SNR矩阵分析 + 检测区域可视化完成！")

# ===== 额外的SNR性能评估函数 =====

def create_snr_performance_summary(wavelength_snr_matrices, all_wavelengths, all_layers, detector_labels, save_dir):
    """创建SNR性能总结报告"""
    
    print("\n📋 创建SNR性能总结报告...")
    
    # 创建性能总结图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 收集所有数据
    performance_data = {}
    
    for wavelength in all_wavelengths:
        if wavelength not in wavelength_snr_matrices:
            continue
            
        wl_nm = int([wl for wl in [1310e-9, 1550e-9] if int(wl*1e9) == wavelength][0] * 1e9)
        performance_data[wl_nm] = {}
        
        for layer_num in all_layers:
            if layer_num not in wavelength_snr_matrices[wavelength]:
                continue
                
            matrix = wavelength_snr_matrices[wavelength][layer_num]
            finite_values = matrix[np.isfinite(matrix)]
            
            if len(finite_values) > 0:
                performance_data[wl_nm][layer_num] = {
                    'mean_snr': np.mean(finite_values),
                    'std_snr': np.std(finite_values),
                    'max_snr': np.max(finite_values),
                    'min_snr': np.min(finite_values),
                    'median_snr': np.median(finite_values),
                    'valid_count': len(finite_values),
                    'total_count': matrix.size
                }
    
    if not performance_data:
        plt.close()
        return
    
    wavelengths_nm = sorted(performance_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. 平均SNR对比 (左上)
    for layer_num in all_layers:
        layer_means = []
        layer_stds = []
        
        for wl_nm in wavelengths_nm:
            if layer_num in performance_data[wl_nm]:
                layer_means.append(performance_data[wl_nm][layer_num]['mean_snr'])
                layer_stds.append(performance_data[wl_nm][layer_num]['std_snr'])
            else:
                layer_means.append(0)
                layer_stds.append(0)
        
        x_pos = np.arange(len(wavelengths_nm))
        width = 0.35
        offset = (list(all_layers).index(layer_num) - len(all_layers)/2 + 0.5) * width
        
        bars = ax1.bar(x_pos + offset, layer_means, width, 
                      yerr=layer_stds, capsize=5,
                      label=f'{layer_num} Layers', 
                      color=colors[list(all_layers).index(layer_num) % len(colors)],
                      alpha=0.8)
        
        # 添加数值标注
        for bar, mean_val in zip(bars, layer_means):
            if mean_val > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Wavelength')
    ax1.set_ylabel('Mean Detection SNR (dB)')
    ax1.set_title('Average SNR Performance by Wavelength')
    ax1.set_xticks(np.arange(len(wavelengths_nm)))
    ax1.set_xticklabels([f'{wl}nm' for wl in wavelengths_nm])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. SNR范围对比 (右上)
    for layer_idx, layer_num in enumerate(all_layers):
        layer_mins = []
        layer_maxs = []
        layer_medians = []
        
        for wl_nm in wavelengths_nm:
            if layer_num in performance_data[wl_nm]:
                data = performance_data[wl_nm][layer_num]
                layer_mins.append(data['min_snr'])
                layer_maxs.append(data['max_snr'])
                layer_medians.append(data['median_snr'])
            else:
                layer_mins.append(0)
                layer_maxs.append(0)
                layer_medians.append(0)
        
        x_pos = np.arange(len(wavelengths_nm)) + layer_idx * 0.2
        
        # 绘制范围条
        for i, (min_val, max_val, median_val) in enumerate(zip(layer_mins, layer_maxs, layer_medians)):
            if max_val > min_val:
                ax2.plot([x_pos[i], x_pos[i]], [min_val, max_val], 
                        color=colors[layer_idx], linewidth=3, alpha=0.7)
                ax2.plot(x_pos[i], median_val, 'o', 
                        color=colors[layer_idx], markersize=6, 
                        label=f'{layer_num} Layers' if i == 0 else "")
    
    ax2.set_xlabel('Wavelength')
    ax2.set_ylabel('Detection SNR Range (dB)')
    ax2.set_title('SNR Range (Min-Max) with Median')
    ax2.set_xticks(np.arange(len(wavelengths_nm)) + 0.1)
    ax2.set_xticklabels([f'{wl}nm' for wl in wavelengths_nm])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 数据有效性统计 (左下)
    for layer_idx, layer_num in enumerate(all_layers):
        valid_rates = []
        
        for wl_nm in wavelengths_nm:
            if layer_num in performance_data[wl_nm]:
                data = performance_data[wl_nm][layer_num]
                valid_rate = data['valid_count'] / data['total_count'] * 100
                valid_rates.append(valid_rate)
            else:
                valid_rates.append(0)
        
        x_pos = np.arange(len(wavelengths_nm))
        width = 0.35
        offset = (layer_idx - len(all_layers)/2 + 0.5) * width
        
        bars = ax3.bar(x_pos + offset, valid_rates, width,
                      label=f'{layer_num} Layers', 
                      color=colors[layer_idx % len(colors)],
                      alpha=0.8)
        
        # 添加数值标注
        for bar, rate in zip(bars, valid_rates):
            if rate > 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax3.set_xlabel('Wavelength')
    ax3.set_ylabel('Valid SNR Data Rate (%)')
    ax3.set_title('Data Validity (Finite SNR Values)')
    ax3.set_xticks(np.arange(len(wavelengths_nm)))
    ax3.set_xticklabels([f'{wl}nm' for wl in wavelengths_nm])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)
    
    # 4. 综合性能雷达图 (右下)
    # 计算综合性能指标
    performance_metrics = {}
    
    for wl_nm in wavelengths_nm:
        metrics = []
        for layer_num in all_layers:
            if layer_num in performance_data[wl_nm]:
                data = performance_data[wl_nm][layer_num]
                # 综合指标：平均SNR、稳定性(1/std)、最大SNR、数据有效性
                mean_score = min(data['mean_snr'] / 20, 1.0)  # 归一化到0-1
                stability_score = min(1.0 / (data['std_snr'] + 1), 1.0)
                max_score = min(data['max_snr'] / 30, 1.0)
                validity_score = data['valid_count'] / data['total_count']
                
                overall_score = (mean_score + stability_score + max_score + validity_score) / 4
                metrics.append(overall_score)
            else:
                metrics.append(0)
        
        performance_metrics[wl_nm] = metrics
    
    # 绘制综合性能对比
    x_pos = np.arange(len(all_layers))
    width = 0.35
    
    for wl_idx, wl_nm in enumerate(wavelengths_nm):
        offset = (wl_idx - len(wavelengths_nm)/2 + 0.5) * width
        bars = ax4.bar(x_pos + offset, performance_metrics[wl_nm], width,
                      label=f'{wl_nm}nm', 
                      color=colors[wl_idx % len(colors)],
                      alpha=0.8)
        
        # 添加数值标注
        for bar, score in zip(bars, performance_metrics[wl_nm]):
            if score > 0:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax4.set_xlabel('Layer Configuration')
    ax4.set_ylabel('Overall Performance Score')
    ax4.set_title('Comprehensive Performance Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{layer} Layers' for layer in all_layers])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    plt.suptitle('Detection SNR Performance Summary Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, 'snr_performance_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 保存: snr_performance_summary.png")
    
    # 保存性能数据到CSV
    summary_data = []
    for wl_nm in wavelengths_nm:
        for layer_num in all_layers:
            if layer_num in performance_data[wl_nm]:
                data = performance_data[wl_nm][layer_num]
                summary_data.append({
                    'Wavelength_nm': wl_nm,
                    'Layers': layer_num,
                    'Mean_SNR_dB': data['mean_snr'],
                    'Std_SNR_dB': data['std_snr'],
                    'Max_SNR_dB': data['max_snr'],
                    'Min_SNR_dB': data['min_snr'],
                    'Median_SNR_dB': data['median_snr'],
                    'Valid_Data_Count': data['valid_count'],
                    'Total_Data_Count': data['total_count'],
                    'Data_Validity_Rate': data['valid_count'] / data['total_count'] * 100
                })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        csv_path = os.path.join(save_dir, 'snr_performance_summary.csv')
        df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ 保存: snr_performance_summary.csv")

# 在主分析函数中添加性能总结
def main_separate_wavelength_snr_analysis_enhanced(config):
    """增强版独立波长SNR矩阵分析函数"""
    
    # 执行原有分析
    wavelength_snr_matrices, wavelength_snr_raw_matrices = main_separate_wavelength_snr_analysis(config)
    
    if wavelength_snr_matrices is not None:
        # 获取必要的参数
        field_data = load_field_data_multiwavelength(config)
        _, _, all_wavelengths, all_layers, all_modes, detector_labels = \
            calculate_separate_wavelength_snr_matrices(field_data, config)
        
        save_dir = os.path.join(config.save_dir, "separate_wavelength_snr_matrices")
        
        # 9. 创建性能总结报告
        create_snr_performance_summary(wavelength_snr_matrices, all_wavelengths, all_layers, detector_labels, save_dir)
        
        print(f"\n🎉 增强版Detection SNR分析完成！")
        print(f"📈 额外生成文件:")
        print(f"   - snr_performance_summary.png (综合性能报告)")
        print(f"   - snr_performance_summary.csv (性能数据表)")
    
    return wavelength_snr_matrices, wavelength_snr_raw_matrices

# 如果需要执行增强版分析，取消下面的注释
print("\n" + "="*60)
print("执行增强版独立波长Detection SNR矩阵分析")
print("="*60)

try:
    wavelength_snr_matrices_enhanced, wavelength_snr_raw_matrices_enhanced = main_separate_wavelength_snr_analysis_enhanced(config)
    print("✅ 增强版SNR分析完成！")
except Exception as e:
    print(f"❌ 增强版SNR分析失败: {e}")

print(f"\n⏰ 程序执行完成")

print(f"\n总执行时间: {time.time() - start_time:.2f} 秒")
print("🎉 独立波长Detection SNR矩阵分析完成！")


