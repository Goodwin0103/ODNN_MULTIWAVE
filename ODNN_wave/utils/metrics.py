import torch
import numpy as np
import matplotlib.pyplot as plt

def create_evaluation_regions_mode_wavelength(H, W, radius, num_modes, num_wavelengths, offsets=None):
    """
    创建评估区域
    
    参数:
        H, W: 图像尺寸
        radius: 区域半径
        num_modes: 模式数量
        num_wavelengths: 波长数量
        offsets: 区域偏移列表
        
    返回:
        评估区域列表
    """
    if offsets is None:
        offsets = [(0, 0)]  # 默认在中心
    
    regions = []
    
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wavelengths):
            for offset_idx, (offset_x, offset_y) in enumerate(offsets):
                center_x = W // 2 + offset_x
                center_y = H // 2 + offset_y
                
                # 创建区域掩膜
                y, x = np.ogrid[-center_y:H-center_y, -center_x:W-center_x]
                mask = x*x + y*y <= radius*radius
                
                regions.append({
                    'mask': mask,
                    'mode_idx': mode_idx,
                    'wl_idx': wl_idx,
                    'offset_idx': offset_idx,
                    'center': (center_x, center_y),
                    'radius': radius
                })
    
    return regions

def evaluate_output(output, regions, idx=0):
    """
    评估单个输出
    
    参数:
        output: 输出场
        regions: 评估区域列表
        idx: 区域索引
        
    返回:
        区域内的能量
    """
    if torch.is_tensor(output):
        output = output.detach().cpu().numpy()
    
    if np.iscomplexobj(output):
        output = np.abs(output)**2  # 计算强度
    
    region = regions[idx]
    mask = region['mask']
    
    # 计算区域内的能量
    energy = np.sum(output * mask)
    
    return energy

def evaluate_all_regions(output, regions):
    """
    评估所有区域
    
    参数:
        output: 输出场
        regions: 评估区域列表
        
    返回:
        各区域能量列表
    """
    energies = []
    
    for idx in range(len(regions)):
        energy = evaluate_output(output, regions, idx)
        energies.append(energy)
    
    return energies

def visualize_labels(regions, H, W):
    """
    可视化标签区域
    
    参数:
        regions: 评估区域列表
        H, W: 图像尺寸
    """
    # 创建空白图像
    image = np.zeros((H, W))
    
    # 绘制每个区域
    for idx, region in enumerate(regions):
        mask = region['mask']
        image[mask] = idx + 1
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='viridis')
    plt.colorbar(label='区域索引')
    plt.title('标签区域可视化')
    plt.tight_layout()
