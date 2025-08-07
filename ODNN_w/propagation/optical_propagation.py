# propagation/optical_propagation.py
import torch
import numpy as np

def propagation_multi(u_in, z, wavelengths, pixel_size, device='cuda:0'):
    """
    多波长光学传播函数
    
    Parameters:
    -----------
    u_in : torch.Tensor
        输入光场 [num_wavelengths, H, W] 或 [batch, num_wavelengths, H, W]
    z : float
        传播距离 (米)
    wavelengths : array_like
        波长数组 (米)
    pixel_size : float
        像素尺寸 (米)
    device : str
        计算设备
    
    Returns:
    --------
    torch.Tensor : 传播后的光场
    """
    # 确保输入是复数类型
    u_in = u_in.to(dtype=torch.complex64, device=device)
    
    # 处理输入维度
    if u_in.dim() == 3:
        # [num_wavelengths, H, W]
        batch_size = 1
        num_wavelengths, H, W = u_in.shape
        u_in = u_in.unsqueeze(0)  # [1, num_wavelengths, H, W]
    elif u_in.dim() == 4:
        # [batch, num_wavelengths, H, W]
        batch_size, num_wavelengths, H, W = u_in.shape
    else:
        raise ValueError(f"Unsupported input dimension: {u_in.dim()}")
    
    # 转换波长为张量
    wavelengths = torch.as_tensor(wavelengths, dtype=torch.float32, device=device)
    
    # 创建频率网格
    fx = torch.fft.fftfreq(W, d=pixel_size, device=device)
    fy = torch.fft.fftfreq(H, d=pixel_size, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    
    # 输出张量
    u_out = torch.zeros_like(u_in)
    
    # 对每个波长进行传播
    for w_idx, wavelength in enumerate(wavelengths):
        # 计算传播常数
        k = 2 * np.pi / wavelength
        
        # 计算传递函数
        fx2_fy2 = FX**2 + FY**2
        
        # 避免数值不稳定
        kz_squared = k**2 - (2 * np.pi)**2 * fx2_fy2
        
        # 只保留传播模式 (kz^2 > 0)
        propagating_mask = kz_squared > 0
        kz = torch.sqrt(torch.abs(kz_squared))
        kz = torch.where(propagating_mask, kz, torch.zeros_like(kz))
        
        # 传递函数
        H_transfer = torch.exp(1j * kz * z)
        H_transfer = torch.where(propagating_mask, H_transfer, torch.zeros_like(H_transfer))
        
        # 对每个批次进行传播
        for b in range(batch_size):
            # FFT
            U_fft = torch.fft.fft2(u_in[b, w_idx])
            
            # 应用传递函数
            U_prop = U_fft * H_transfer
            
            # IFFT
            u_out[b, w_idx] = torch.fft.ifft2(U_prop)
    
    # 如果原始输入是3D，返回3D
    if batch_size == 1 and u_in.dim() == 4:
        u_out = u_out.squeeze(0)
    
    return u_out

def fresnel_propagation(u_in, z, wavelength, pixel_size, device='cuda:0'):
    """
    单波长菲涅尔传播
    
    Parameters:
    -----------
    u_in : torch.Tensor
        输入光场 [H, W] 或 [batch, H, W]
    z : float
        传播距离 (米)
    wavelength : float
        波长 (米)
    pixel_size : float
        像素尺寸 (米)
    device : str
        计算设备
    
    Returns:
    --------
    torch.Tensor : 传播后的光场
    """
    u_in = u_in.to(dtype=torch.complex64, device=device)
    
    if u_in.dim() == 2:
        H, W = u_in.shape
        u_in = u_in.unsqueeze(0)  # [1, H, W]
        squeeze_output = True
    else:
        batch_size, H, W = u_in.shape
        squeeze_output = False
    
    # 波数
    k = 2 * np.pi / wavelength
    
    # 频率网格
    fx = torch.fft.fftfreq(W, d=pixel_size, device=device)
    fy = torch.fft.fftfreq(H, d=pixel_size, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    
    # 菲涅尔传递函数
    H_fresnel = torch.exp(1j * k * z) * torch.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    # 传播
    u_out = torch.zeros_like(u_in)
    for b in range(u_in.shape[0]):
        U_fft = torch.fft.fft2(u_in[b])
        U_prop = U_fft * H_fresnel
        u_out[b] = torch.fft.ifft2(U_prop)
    
    if squeeze_output:
        u_out = u_out.squeeze(0)
    
    return u_out

def angular_spectrum_propagation(u_in, z, wavelength, pixel_size, device='cuda:0'):
    """
    角谱传播方法
    
    Parameters:
    -----------
    u_in : torch.Tensor
        输入光场
    z : float
        传播距离
    wavelength : float
        波长
    pixel_size : float
        像素尺寸
    device : str
        设备
    
    Returns:
    --------
    torch.Tensor : 传播后的光场
    """
    return propagation_multi(u_in.unsqueeze(0), z, [wavelength], pixel_size, device).squeeze(0)

def propagation_kernel(size, wavelength, z, pixel_size, device='cuda:0'):
    """
    生成传播核
    
    Parameters:
    -----------
    size : tuple
        核尺寸 (H, W)
    wavelength : float
        波长
    z : float
        传播距离
    pixel_size : float
        像素尺寸
    device : str
        设备
    
    Returns:
    --------
    torch.Tensor : 传播核
    """
    H, W = size
    k = 2 * np.pi / wavelength
    
    # 频率网格
    fx = torch.fft.fftfreq(W, d=pixel_size, device=device)
    fy = torch.fft.fftfreq(H, d=pixel_size, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    
    # 传播核
    fx2_fy2 = FX**2 + FY**2
    kz_squared = k**2 - (2 * np.pi)**2 * fx2_fy2
    
    # 传播模式掩膜
    propagating_mask = kz_squared > 0
    kz = torch.sqrt(torch.abs(kz_squared))
    kz = torch.where(propagating_mask, kz, torch.zeros_like(kz))
    
    # 传递函数
    kernel = torch.exp(1j * kz * z)
    kernel = torch.where(propagating_mask, kernel, torch.zeros_like(kernel))
    
    return kernel

# 兼容性函数 - 如果原代码使用了不同的函数名
def propagate_light(u_in, z, wavelength, pixel_size, method='angular_spectrum'):
    """
    光传播的通用接口
    """
    if method == 'angular_spectrum':
        return angular_spectrum_propagation(u_in, z, wavelength, pixel_size)
    elif method == 'fresnel':
        return fresnel_propagation(u_in, z, wavelength, pixel_size)
    else:
        raise ValueError(f"Unknown propagation method: {method}")

# 批处理版本
def batch_propagation(u_in, z, wavelengths, pixel_size, device='cuda:0'):
    """
    批处理传播
    """
    return propagation_multi(u_in, z, wavelengths, pixel_size, device)
