import numpy as np
import matplotlib.pyplot as plt
import torch

def create_labels_mode_wavelength(H, W, radius, mode_idx, wl_idx, detectsize=15):
    """
    为特定模式和波长创建空间标签掩码
    
    Args:
        H, W: 空间尺寸
        radius: 焦点半径
        mode_idx: 模式索引 (0, 1, 2)
        wl_idx: 波长索引 (0, 1, 2)
        detectsize: 检测区域大小
    
    Returns:
        numpy.ndarray: 形状为 (H, W) 的空间掩码
    """
    # 创建空的标签掩码
    label_mask = np.zeros((H, W), dtype=np.float32)
    
    # 计算目标区域索引
    target_region_idx = mode_idx * 3 + wl_idx
    
    # 创建9个区域的3x3网格布局
    grid_size = 3  # 3x3网格
    
    # 计算该区域在网格中的位置
    grid_row = target_region_idx // grid_size
    grid_col = target_region_idx % grid_size
    
    # 计算区域中心坐标
    region_height = H // grid_size
    region_width = W // grid_size
    
    center_y = grid_row * region_height + region_height // 2
    center_x = grid_col * region_width + region_width // 2
    
    # 创建圆形检测区域
    y, x = np.ogrid[:H, :W]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # 在检测区域内设置标签为1
    label_mask[distance <= detectsize] = 1.0
    
    return label_mask


def create_evaluation_regions_mode_wavelength(H, W, radius, detectsize=15):
    """
    为多模式多波长创建所有评估区域
    
    Returns:
        list: 包含9个numpy数组的列表，每个形状为 (H, W)
    """
    regions = []
    grid_size = 3
    
    for region_idx in range(9):  # 3模式 × 3波长 = 9个区域
        grid_row = region_idx // grid_size
        grid_col = region_idx % grid_size
        
        region_height = H // grid_size
        region_width = W // grid_size
        
        center_y = grid_row * region_height + region_height // 2
        center_x = grid_col * region_width + region_width // 2
        
        # 创建圆形区域掩码
        mask = np.zeros((H, W), dtype=np.float32)
        y, x = np.ogrid[:H, :W]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask[distance <= detectsize] = 1.0
        
        regions.append(mask)
    
    return regions

def compute_performance_metrics_multimode_multiwave(predictions, labels, evaluation_regions):
    """
    计算多模式多波长的性能指标
    
    Args:
        predictions: 预测结果，形状 (3, 3, H, W)
        labels: 真实标签，形状 (3, 3, H, W)  
        evaluation_regions: 评估区域列表，每个元素形状 (H, W)
    
    Returns:
        dict: 性能指标字典
    """
    results = {
        'mode_performance': {},
        'wavelength_performance': {},
        'overall_performance': {}
    }
    
    all_target_intensities = []
    all_non_target_intensities = []
    
    # 遍历所有模式和波长组合
    for mode_idx in range(3):
        for wl_idx in range(3):
            # 获取对应的评估区域
            region_idx = mode_idx * 3 + wl_idx
            if region_idx < len(evaluation_regions):
                region_mask = evaluation_regions[region_idx]
                
                # 获取预测和标签
                pred = predictions[mode_idx, wl_idx]
                label = labels[mode_idx, wl_idx]
                
                # 确保是numpy数组
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().cpu().numpy()
                if isinstance(label, torch.Tensor):
                    label = label.detach().cpu().numpy()
                if isinstance(region_mask, torch.Tensor):
                    region_mask = region_mask.detach().cpu().numpy()
                
                # 计算目标区域和非目标区域的强度
                target_mask = (region_mask > 0) & (label > 0)
                non_target_mask = (region_mask == 0) | (label == 0)
                
                if np.sum(target_mask) > 0:
                    target_intensity = np.mean(pred[target_mask])
                    all_target_intensities.append(target_intensity)
                
                if np.sum(non_target_mask) > 0:
                    non_target_intensity = np.mean(pred[non_target_mask])
                    all_non_target_intensities.append(non_target_intensity)
    
    # 计算整体性能
    if all_target_intensities and all_non_target_intensities:
        avg_target = np.mean(all_target_intensities)
        avg_non_target = np.mean(all_non_target_intensities)
        
        contrast = (avg_target - avg_non_target) / (avg_target + avg_non_target + 1e-8)
        efficiency = avg_target / (np.max(all_target_intensities) + 1e-8)
        
        results['overall_performance'] = {
            'contrast': float(contrast),
            'efficiency': float(efficiency),
            'avg_target_intensity': float(avg_target),
            'avg_non_target_intensity': float(avg_non_target)
        }
    
    return results

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
    """
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
            plt.title(f'模式 {mode_idx}, λ={int(wavelengths[wl_idx]*1e9)}nm')
            plt.axis('off')
    plt.tight_layout()
    plt.show()
def debug_region_consistency_with_config(config):
    """使用配置参数调试区域定义一致性"""
    H, W = config.layer_size, config.layer_size
    radius = config.focus_radius
    detectsize = config.detectsize
    
    print("=== 区域映射验证 ===")
    print(f"使用参数: H={H}, W={W}, radius={radius}, detectsize={detectsize}")
    
    # 检查每个模式-波长组合的目标区域
    for mode_idx in range(3):
        for wl_idx in range(3):
            # 使用标签生成函数
            labels = create_labels_mode_wavelength(H, W, radius, mode_idx, wl_idx)
            
            # 转换为numpy数组（如果不是的话）
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            
            # 找到目标区域
            found_target = False
            for wl_check in range(3):
                # 使用numpy.nonzero而不是torch.nonzero
                nonzero_regions = np.nonzero(labels[wl_check])
                if len(nonzero_regions[0]) > 0:  # nonzero返回的是元组
                    target_region = nonzero_regions[0][0]  # 第一个非零元素的索引
                    wavelengths = [450, 550, 650]
                    print(f"Mode {mode_idx+1}, {wavelengths[wl_idx]}nm -> 区域 {target_region} (在波长{wl_check}的标签中)")
                    
                    # 计算预期区域
                    expected_region = mode_idx * 3 + wl_idx
                    match = expected_region == target_region
                    print(f"  预期区域: {expected_region}, 实际区域: {target_region}, 匹配: {'✓' if match else '✗'}")
                    found_target = True
                    break
            
            if not found_target:
                print(f"Mode {mode_idx+1}, 波长索引{wl_idx} -> 未找到目标区域！")
    
    # 检查evaluation_regions是否与标签区域匹配
    evaluation_regions = create_evaluation_regions_mode_wavelength(H, W, radius, detectsize=detectsize)
    print(f"\nEvaluation regions 数量: {len(evaluation_regions)}")
    
    return evaluation_regions

def debug_label_generation(config):
    """详细调试标签生成过程"""
    H, W = config.layer_size, config.layer_size
    radius = config.focus_radius
    detectsize = config.detectsize
    
    print("=== 详细标签生成调试 ===")
    print(f"参数: H={H}, W={W}, radius={radius}, detectsize={detectsize}")
    
    # 测试一个具体的例子
    mode_idx = 0  # 模式1
    wl_idx = 1    # 550nm
    
    print(f"\n测试: 模式 {mode_idx+1}, 波长索引 {wl_idx}")
    
    # 1. 先看看evaluation_regions是什么
    print("\n--- 1. 检查evaluation_regions ---")
    evaluation_regions = create_evaluation_regions_mode_wavelength(H, W, radius, detectsize=detectsize)
    print(f"Evaluation regions 数量: {len(evaluation_regions)}")
    print(f"Evaluation regions 类型: {type(evaluation_regions)}")
    
    if len(evaluation_regions) > 0:
        print(f"第一个区域形状: {evaluation_regions[0].shape if hasattr(evaluation_regions[0], 'shape') else 'No shape'}")
        print(f"第一个区域类型: {type(evaluation_regions[0])}")
        
        # 检查每个区域的非零元素数量
        for i, region in enumerate(evaluation_regions):
            if isinstance(region, torch.Tensor):
                nonzero_count = torch.sum(region).item()
            else:
                nonzero_count = np.sum(region)
            print(f"区域 {i}: 非零元素数量 = {nonzero_count}")
    
    # 2. 生成标签并详细检查
    print(f"\n--- 2. 生成标签详细检查 ---")
    labels = create_labels_mode_wavelength(H, W, radius, mode_idx, wl_idx)
    
    print(f"Labels 形状: {labels.shape}")
    print(f"Labels 类型: {type(labels)}")
    print(f"Labels 数据类型: {labels.dtype}")
    
    # 检查每个波长维度的标签
    for wl in range(labels.shape[0]):
        if isinstance(labels, torch.Tensor):
            label_sum = torch.sum(labels[wl]).item()
            label_max = torch.max(labels[wl]).item()
            label_min = torch.min(labels[wl]).item()
        else:
            label_sum = np.sum(labels[wl])
            label_max = np.max(labels[wl])
            label_min = np.min(labels[wl])
        
        print(f"  波长维度 {wl}: 总和={label_sum:.6f}, 最大值={label_max:.6f}, 最小值={label_min:.6f}")
    
    # 3. 检查预期的目标区域
    print(f"\n--- 3. 预期目标区域检查 ---")
    expected_region = mode_idx * 3 + wl_idx
    print(f"预期目标区域索引: {expected_region}")
    
    if expected_region < len(evaluation_regions):
        target_region = evaluation_regions[expected_region]
        if isinstance(target_region, torch.Tensor):
            target_sum = torch.sum(target_region).item()
        else:
            target_sum = np.sum(target_region)
        print(f"目标区域 {expected_region} 的非零元素数量: {target_sum}")
        
        # 检查标签是否与目标区域匹配
        if wl_idx < labels.shape[0]:
            if isinstance(labels, torch.Tensor) and isinstance(target_region, torch.Tensor):
                match_sum = torch.sum(labels[wl_idx] * target_region).item()
            else:
                # 转换为numpy进行计算
                label_np = labels[wl_idx].numpy() if isinstance(labels, torch.Tensor) else labels[wl_idx]
                target_np = target_region.numpy() if isinstance(target_region, torch.Tensor) else target_region
                match_sum = np.sum(label_np * target_np)
            print(f"标签与目标区域重叠: {match_sum:.6f}")
    else:
        print(f"错误：预期区域索引 {expected_region} 超出范围 (总共 {len(evaluation_regions)} 个区域)")
    
    # 4. 检查create_labels_mode_wavelength函数的内部逻辑
    print(f"\n--- 4. 函数内部逻辑检查 ---")
    print("让我们手动执行create_labels_mode_wavelength的逻辑...")
    
    # 重新执行函数内部逻辑
    labels_manual = np.zeros((3, len(evaluation_regions)))
    target_region_idx = mode_idx * 3 + wl_idx
    print(f"计算得到的目标区域索引: {target_region_idx}")
    
    if target_region_idx < len(evaluation_regions):
        labels_manual[wl_idx, target_region_idx] = 1.0
        print(f"手动设置 labels[{wl_idx}, {target_region_idx}] = 1.0")
        print(f"手动生成的标签形状: {labels_manual.shape}")
        print(f"手动标签非零元素: {np.sum(labels_manual)}")
        
        # 与函数生成的标签比较
        if isinstance(labels, torch.Tensor):
            labels_np = labels.numpy()
        else:
            labels_np = labels
            
        print(f"函数生成的标签非零元素: {np.sum(labels_np)}")
        print(f"两者是否相等: {np.allclose(labels_manual, labels_np)}")
    
    return labels, evaluation_regions

# 在 label_utils.py 文件末尾添加以下函数

def create_improved_labels(wavelengths, num_modes, height, width, 
                          region_size_ratio=0.15, energy_balance=True):
    """
    改进的标签生成函数，修复模式-波长-区域映射问题
    
    Args:
        wavelengths: 波长列表
        num_modes: 模式数量
        height, width: 检测区域尺寸
        region_size_ratio: 区域大小比例
        energy_balance: 是否进行能量平衡
    """
    import torch
    import torch.nn.functional as F
    
    num_wavelengths = len(wavelengths)
    labels = torch.zeros(num_wavelengths, num_modes, height, width)
    
    # 创建3x3网格布局
    grid_rows, grid_cols = 3, 3
    
    print("🏷️ 创建改进标签映射:")
    print("-" * 50)
    
    for wl_idx, wavelength in enumerate(wavelengths):
        for mode_idx in range(num_modes):
            # 计算目标区域索引: 模式为主要维度
            region_idx = mode_idx * num_wavelengths + wl_idx
            
            # 计算网格位置
            grid_row = region_idx // grid_cols
            grid_col = region_idx % grid_cols
            
            # 计算区域中心位置
            center_y = int((grid_row + 0.5) * height / grid_rows)
            center_x = int((grid_col + 0.5) * width / grid_cols)
            
            # 计算区域大小
            region_radius = int(min(height, width) * region_size_ratio / 2)
            
            # 创建高斯分布
            y_coords, x_coords = torch.meshgrid(
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing='ij'
            )
            
            # 计算到中心的距离
            distances = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            
            # 创建高斯分布标签
            sigma = region_radius / 2
            gaussian = torch.exp(-distances**2 / (2 * sigma**2))
            
            # 能量平衡 - 根据波长调整强度
            if energy_balance:
                wl_nm = int(wavelength * 1e9)
                if wl_nm == 450:  # 450nm
                    intensity_factor = 1.0
                elif wl_nm == 550:  # 550nm  
                    intensity_factor = 0.6  # 降低550nm的权重
                elif wl_nm == 650:  # 650nm
                    intensity_factor = 1.5  # 增加650nm的权重
                else:
                    intensity_factor = 1.0
                    
                gaussian *= intensity_factor
            
            labels[wl_idx, mode_idx] = gaussian
            
            print(f"  模式{mode_idx+1}, {wavelength*1e9:.0f}nm → 区域{region_idx} "
                  f"(网格{grid_row},{grid_col}) 中心({center_x},{center_y})")
    
    print("-" * 50)
    return labels

def verify_label_mapping(wavelengths, num_modes, height, width):
    """验证标签映射是否正确"""
    
    print("🔍 验证标签映射...")
    
    # 生成改进的标签
    improved_labels = create_improved_labels(wavelengths, num_modes, height, width)
    
    # 生成原始标签进行对比
    original_labels = create_labels(wavelengths, num_modes, height, width)
    
    print("\n📊 标签对比分析:")
    print("=" * 80)
    print(f"{'模式':<6} {'波长':<8} {'期望区域':<8} {'原始峰值':<12} {'改进峰值':<12} {'状态'}")
    print("=" * 80)
    
    for wl_idx, wavelength in enumerate(wavelengths):
        for mode_idx in range(num_modes):
            expected_region = mode_idx * len(wavelengths) + wl_idx
            
            # 分析原始标签
            orig_label = original_labels[wl_idx, mode_idx]
            orig_peak_pos = torch.unravel_index(torch.argmax(orig_label), orig_label.shape)
            
            # 分析改进标签
            impr_label = improved_labels[wl_idx, mode_idx]
            impr_peak_pos = torch.unravel_index(torch.argmax(impr_label), impr_label.shape)
            
            # 判断是否改进
            status = "✅ 改进" if torch.max(impr_label) > torch.max(orig_label) else "➡️ 调整"
            
            print(f"{mode_idx+1:<6} {wavelength*1e9:.0f}nm{'':<3} {expected_region:<8} "
                  f"{orig_peak_pos}<{'':<8} {impr_peak_pos}<{'':<8} {status}")
    
    print("=" * 80)
    return improved_labels

# 修改原有的 create_labels 函数，添加调试选项
def create_labels_with_debug(wavelengths, num_modes, height, width, debug=True):
    """带调试信息的标签创建函数"""
    if debug:
        print("🔍 使用改进的标签生成...")
        return create_improved_labels(wavelengths, num_modes, height, width)
    else:
        return create_labels(wavelengths, num_modes, height, width)
