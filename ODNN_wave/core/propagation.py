import torch
import numpy as np
import matplotlib.pyplot as plt

def propagation(E, z, lam, dx, device=None):
    """
    使用角谱法模拟光场传播
    
    参数:
        E: 输入光场
        z: 传播距离(m)
        lam: 波长(m)
        dx: 像素大小(m)
        device: 计算设备
        
    返回:
        传播后的光场
    """
    if device is None:
        device = E.device
        
    # 确保输入是复数张量
    if not torch.is_complex(E):
        E = E.to(torch.complex64)
        
    # 获取光场尺寸
    if len(E.shape) == 2:
        M, N = E.shape
        E = E.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    elif len(E.shape) == 3:
        if E.shape[0] == 1:  # [1, M, N]
            E = E.unsqueeze(0)  # 添加批次维度 [1, 1, M, N]
        else:  # [B, M, N]
            E = E.unsqueeze(1)  # 添加通道维度 [B, 1, M, N]
    
    B, C, M, N = E.shape
    
    # 构建频率网格
    fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(M, d=dx)).to(device)
    fx, fy = torch.meshgrid(fx, fy, indexing='ij')
    
    # 计算传播相位
    k = 2 * np.pi / lam
    phase = k * z * torch.sqrt(1 - (lam*fx)**2 - (lam*fy)**2 + 0j)
    phase = phase.to(device)
    
    # 应用传播相位
    E_fft = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2, -1))
    E_fft = E_fft * torch.exp(1j * phase)
    E_prop = torch.fft.ifft2(torch.fft.ifftshift(E_fft, dim=(-2, -1)))
    
    return E_prop

def propagation_multi(E, z, wavelengths, dx, wavelength_idx=None, device=None):
    """
    多波长光场传播
    
    参数:
        E: 输入光场 [B, C, H, W] 或 [B, H, W]
        z: 传播距离(m)
        wavelengths: 波长列表(m)
        dx: 像素大小(m)
        wavelength_idx: 指定波长索引(可选)
        device: 计算设备
        
    返回:
        传播后的光场
    """
    if device is None:
        device = E.device if torch.is_tensor(E) else torch.device('cpu')
    
    # 如果指定了波长索引
    if wavelength_idx is not None:
        wavelength = wavelengths[wavelength_idx]
        return propagation(E, z, wavelength, dx, device)
    
    # 处理多波长情况
    if len(E.shape) == 4:  # [B, C, H, W]
        batch_size, num_channels, H, W = E.shape
        
        if num_channels == len(wavelengths):
            # 每个通道对应一个波长
            outputs = []
            for wl_idx, wavelength in enumerate(wavelengths):
                E_wl = E[:, wl_idx:wl_idx+1]
                E_prop = propagation(E_wl, z, wavelength, dx, device)
                outputs.append(E_prop)
            
            return torch.cat(outputs, dim=1)
        else:
            # 单通道应用于所有波长
            outputs = []
            for wavelength in wavelengths:
                E_prop = propagation(E, z, wavelength, dx, device)
                outputs.append(E_prop)
            
            if len(outputs) > 1:
                return torch.cat(outputs, dim=1)
            else:
                return outputs[0]
    else:
        # 单一波长处理（使用第一个波长）
        return propagation(E, z, wavelengths[0], dx, device)

def plot_propagated_field(field, title=None):
    """
    绘制传播场的幅度
    
    参数:
        field: 光场
        title: 标题
    """
    if torch.is_tensor(field):
        field = field.detach().cpu().numpy()
    
    if np.iscomplexobj(field):
        field = np.abs(field)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(field, cmap='viridis')
    plt.colorbar(label='幅度')
    if title:
        plt.title(title)
    plt.tight_layout()
