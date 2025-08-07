# ODNN_functions.py
import torch
import numpy as np

def generate_fields_ts(complex_weights, mmf_data, num_data, num_modes, image_size, wavelength=550e-9):
    """
    生成光场
    
    Parameters:
    -----------
    complex_weights : torch.Tensor
        复权重 [batch_size, num_modes]
    mmf_data : torch.Tensor
        多模光纤数据 [num_modes, H, W]
    num_data : int
        数据数量
    num_modes : int
        模式数量
    image_size : int
        图像尺寸
    wavelength : float
        波长
    
    Returns:
    --------
    torch.Tensor : 生成的光场 [batch_size, 1, H, W]
    """
    batch_size = complex_weights.shape[0]
    
    # 初始化输出
    fields = torch.zeros(batch_size, 1, image_size, image_size, dtype=torch.complex64)
    
    for b in range(batch_size):
        field = torch.zeros(image_size, image_size, dtype=torch.complex64)
        
        # 叠加各模式
        for m in range(num_modes):
            weight = complex_weights[b, m]
            mode_field = mmf_data[m, :, :]
            field += weight * mode_field
        
        fields[b, 0, :, :] = field
    
    return fields

def create_labels(layer_size, layer_size_y, num_detector, focus_radius, detector_index):
    """
    创建标签
    
    Parameters:
    -----------
    layer_size : int
        层尺寸 X
    layer_size_y : int
        层尺寸 Y
    num_detector : int
        检测器数量
    focus_radius : int
        聚焦半径
    detector_index : int
        检测器索引 (1-based)
    
    Returns:
    --------
    np.ndarray : 标签数组 [layer_size_y, layer_size]
    """
    label = np.zeros((layer_size_y, layer_size))
    
    # 计算检测器位置
    if num_detector == 1:
        center_x = layer_size // 2
        center_y = layer_size_y // 2
    else:
        # 多检测器情况下的位置计算
        positions = get_detector_positions(layer_size, layer_size_y, num_detector)
        center_x, center_y = positions[detector_index - 1]
    
    # 创建圆形区域
    y, x = np.ogrid[:layer_size_y, :layer_size]
    mask = (x - center_x)**2 + (y - center_y)**2 <= focus_radius**2
    label[mask] = 1.0
    
    return label

def get_detector_positions(layer_size, layer_size_y, num_detector):
    """
    获取检测器位置
    
    Parameters:
    -----------
    layer_size : int
        层尺寸 X
    layer_size_y : int
        层尺寸 Y
    num_detector : int
        检测器数量
    
    Returns:
    --------
    list : 检测器位置列表 [(x, y), ...]
    """
    positions = []
    
    if num_detector == 3:
        # 3个检测器的情况
        spacing = layer_size // 4
        positions = [
            (layer_size // 2 - spacing, layer_size_y // 2),
            (layer_size // 2, layer_size_y // 2),
            (layer_size // 2 + spacing, layer_size_y // 2)
        ]
    elif num_detector == 6:
        # 6个检测器的情况 - 2x3网格
        spacing_x = layer_size // 4
        spacing_y = layer_size_y // 4
        positions = [
            (layer_size // 2 - spacing_x, layer_size_y // 2 - spacing_y),
            (layer_size // 2, layer_size_y // 2 - spacing_y),
            (layer_size // 2 + spacing_x, layer_size_y // 2 - spacing_y),
            (layer_size // 2 - spacing_x, layer_size_y // 2 + spacing_y),
            (layer_size // 2, layer_size_y // 2 + spacing_y),
            (layer_size // 2 + spacing_x, layer_size_y // 2 + spacing_y)
        ]
    elif num_detector == 10:
        # 10个检测器的情况 - 2x5网格
        spacing_x = layer_size // 6
        spacing_y = layer_size_y // 4
        positions = []
        for i in range(2):
            for j in range(5):
                x = layer_size // 2 + (j - 2) * spacing_x
                y = layer_size_y // 2 + (i - 0.5) * spacing_y
                positions.append((int(x), int(y)))
    else:
        # 默认情况：均匀分布
        for i in range(num_detector):
            angle = 2 * np.pi * i / num_detector
            radius = min(layer_size, layer_size_y) // 4
            x = int(layer_size // 2 + radius * np.cos(angle))
            y = int(layer_size_y // 2 + radius * np.sin(angle))
            positions.append((x, y))
    
    return positions

def create_evaluation_regions(layer_size, layer_size_y, num_detector, focus_radius, detect_size):
    """
    创建评估区域
    
    Parameters:
    -----------
    layer_size : int
        层尺寸 X
    layer_size_y : int
        层尺寸 Y
    num_detector : int
        检测器数量
    focus_radius : int
        聚焦半径
    detect_size : int
        检测尺寸
    
    Returns:
    --------
    list : 评估区域列表 [(x_start, x_end, y_start, y_end), ...]
    """
    positions = get_detector_positions(layer_size, layer_size_y, num_detector)
    regions = []
    
    for x, y in positions:
        x_start = max(0, x - detect_size // 2)
        x_end = min(layer_size, x + detect_size // 2)
        y_start = max(0, y - detect_size // 2)
        y_end = min(layer_size_y, y + detect_size // 2)
        regions.append((x_start, x_end, y_start, y_end))
    
    return regions

def create_labels_4_MMF3_phase(layer_size, layer_size_y, num_detector, focus_radius, phase_values):
    """
    为MMF3相位创建标签
    """
    # 这个函数根据您的具体需求实现
    # 暂时返回基础标签
    labels = []
    for i in range(num_detector):
        label = create_labels(layer_size, layer_size_y, num_detector, focus_radius, i+1)
        labels.append(label)
    return np.array(labels)

def create_evaluation_regions_4_MMF3_phase(layer_size, layer_size_y, num_detector, focus_radius, detect_size):
    """
    为MMF3相位创建评估区域
    """
    # 暂时使用基础评估区域
    return create_evaluation_regions(layer_size, layer_size_y, num_detector, focus_radius, detect_size)

def generate_complex_weights(num_modes, phase_option=4):
    """
    生成复权重
    
    Parameters:
    -----------
    num_modes : int
        模式数量
    phase_option : int
        相位选项
    
    Returns:
    --------
    np.ndarray : 复权重数组
    """
    if phase_option == 4:
        amplitudes = np.eye(num_modes)
        phases = np.eye(num_modes)
        complex_weights = amplitudes * np.exp(1j * phases)
    else:
        # 其他相位选项的实现
        complex_weights = np.eye(num_modes, dtype=complex)
    
    return complex_weights
