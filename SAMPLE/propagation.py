import torch
import numpy as np

def angular_spectrum_propagation(field, distance, wavelength, pixel_size):
    """
    使用角谱方法传播光场
    
    Args:
        field: 输入光场 (complex tensor)
        distance: 传播距离 (float)
        wavelength: 波长 (float)
        pixel_size: 像素大小 (float)
    
    Returns:
        传播后的光场 (complex tensor)
    """
    # 确保输入是复数张量
    if not torch.is_complex(field):
        field = torch.complex(field, torch.zeros_like(field))
    
    # 获取场的尺寸
    M, N = field.shape
    
    # 计算波数
    k = 2 * np.pi / wavelength
    
    # 创建频率网格
    fx = torch.fft.fftfreq(N, pixel_size).to(field.device)
    fy = torch.fft.fftfreq(M, pixel_size).to(field.device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    
    # 计算传播相位
    # 注意：对于太大的频率，我们需要确保平方根的参数是正的
    temp = 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    temp = torch.clamp(temp, min=0)  # 确保非负
    phase = k * distance * torch.sqrt(temp)
    
    # 计算传递函数
    H = torch.exp(1j * phase)
    
    # 应用角谱传播
    F = torch.fft.fft2(field)
    F_propagated = F * H
    field_propagated = torch.fft.ifft2(F_propagated)
    
    return field_propagated

