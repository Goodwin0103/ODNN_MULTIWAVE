# -*- coding: utf-8 -*-
"""
Created on Fri May  2 16:41:28 2025

@author: zhang
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

#%%

def propagation(E, z, lam,layer_size,pixel_size,device):
        # Convert input to PyTorch tensor and move to GPU
        E = torch.tensor(E, dtype=torch.complex64, device=device)
        
        fft_c = torch.fft.fft2(E)
        c = torch.fft.fftshift(fft_c)
        
        fx = torch.fft.fftshift(torch.fft.fftfreq(layer_size, d=pixel_size)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx)
        argument = (2 * np.pi) ** 2 * ((1. / lam) ** 2 - fxx ** 2 - fyy ** 2)
        
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp)
        
        E = torch.fft.ifft2(torch.fft.ifftshift(c * torch.exp(1j * kz * z)))
        return E

def propagation_multi(
    E: torch.Tensor, 
    *, 
    z: float, 
    wavelengths: list, 
    pixel_size: float, 
    device: torch.device = None
) -> torch.Tensor:
    """
    多波长光场传播函数 (支持批量处理)
    
    参数:
        E (Tensor)      : 输入光场，支持以下形状:
                           [H, W]          → 单样本单通道
                           [C, H, W]       → 单样本多通道
                           [B, C, H, W]    → 批量多通道
                           [B, M, C, H, W] → 批量多模式多通道 (会自动转为[B*M, C, H, W])
        z (float)       : 传播距离 (米)
        wavelengths     : 波长列表 (米), 长度必须等于通道数 C
        pixel_size (float) : 像素物理尺寸 (米)
        device          : 强制指定计算设备 (默认跟随输入张量)
    
    返回:
        Tensor: 传播后的光场，形状与输入维度一致
    """

    # ===================================================================
    # 1. 输入张量标准化 (强制转为 [B, C, H, W])
    # ===================================================================
    original_shape = E.shape
    original_dim = E.dim()
    
    if E.dim() == 2:  # [H, W] → [1, 1, H, W]
        E = E.unsqueeze(0).unsqueeze(0)
    elif E.dim() == 3:  # [C, H, W] → [1, C, H, W]
        E = E.unsqueeze(0)
    elif E.dim() == 5:  # [B, M, C, H, W] → [B*M, C, H, W]
        B, M, C, H, W = E.shape
        E = E.reshape(B*M, C, H, W)
    elif E.dim() != 4:
        raise ValueError(f"输入张量维度错误: 支持 2D/3D/4D/5D, 但输入为 {E.dim()}D")
    
    B, C, H, W = E.shape

    # ===================================================================
    # 2. 设备与数据类型处理
    # ===================================================================
    # 自动推断设备 (优先使用输入张量的设备)
    if device is None:
        device = E.device
    E = E.to(device=device, dtype=torch.complex64)

    # ===================================================================
    # 3. 波长参数校验与标准化
    # ===================================================================
    wavelengths_tensor = torch.tensor(wavelengths, dtype=torch.float32, device=device)
    
    if wavelengths_tensor.numel() == 1:  # 标量 → 复制为通道数
        wavelengths_tensor = wavelengths_tensor.repeat(C)
    elif wavelengths_tensor.numel() != C:
        raise ValueError(
            f"波长数量 ({wavelengths_tensor.numel()}) 必须与输入通道数 (C={C}) 一致"
        )
    wavelengths_tensor = wavelengths_tensor.view(1, C, 1, 1)  # [1, C, 1, 1] 用于广播

    # ===================================================================
    # 4. 计算频率网格 (预计算优化)
    # ===================================================================
    # 频率采样改用 fftfreq，完全一致
    fx = torch.fft.fftshift(torch.fft.fftfreq(H, d=pixel_size)).to(device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(W, d=pixel_size)).to(device)
    fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')  # [H, W]
    # ===================================================================
    # 5. 计算波矢 kz (向量化操作)
    # ===================================================================
    k0 = 2 * torch.pi / wavelengths_tensor  # [1, C, 1, 1]
    
    # 展开频率网格以匹配输入维度 [B, C, H, W]
    fxx_expanded = fxx.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    fyy_expanded = fyy.unsqueeze(0).unsqueeze(0)
    
    # 计算 kz 分量 (考虑 evanescent 波)
    kz_sq = k0**2 - (2 * torch.pi * fxx_expanded)**2 - (2 * torch.pi * fyy_expanded)**2
    kz = torch.sqrt(kz_sq + 0j)  # 强制复数类型 [B, C, H, W]

    # ===================================================================
    # 6. 角谱传播核心计算
    # ===================================================================
    # FFT 变换
    E_fft = torch.fft.fft2(E, dim=(-2, -1))  # [B, C, H, W]
    E_fft_shifted = torch.fft.fftshift(E_fft, dim=(-2, -1))

    # 应用传递函数
    H_z = torch.exp(1j * kz * z)  # [B, C, H, W]
    E_propagated_shifted = E_fft_shifted * H_z

    # IFFT 变换
    E_propagated = torch.fft.ifft2(
        torch.fft.ifftshift(E_propagated_shifted, dim=(-2, -1)), 
        dim=(-2, -1)
    )

    # ===================================================================
    # 7. 恢复原始形状 (如果是5D输入)
    # ===================================================================
    if original_dim == 5:
        B_orig, M, C, H, W = original_shape
        E_propagated = E_propagated.reshape(B_orig, M, C, H, W)

    return E_propagated

def plot_propagated_field(E0, z_start, z_end, z_step, dx, lam):
    """
    Simulates the free-space propagation of a complex optical field over a range of distances 
    using an external propagation function, and visualizes the amplitude at each distance.

    Parameters:
    -----------
    E0 : torch.Tensor
        A 2D complex-valued tensor representing the input optical field. Shape: [Nx, Nx].
    z_start : float
        Starting propagation distance along the z-axis (in meters).
    z_end : float
        Ending propagation distance along the z-axis (in meters).
    z_step : float
        Step size between each propagation distance (in meters).
    dx : float
        Spatial sampling interval (pixel size) in meters. Passed to the propagation function.
    lam : float
        Wavelength of light (in meters). Passed to the propagation function.

    Functionality:
    --------------
    - Iteratively propagates the input field E0 over multiple z distances using a function `propagation(E0, z)`.
    - At each step, computes the amplitude (magnitude) of the propagated field.
    - Displays the amplitude distribution in a grid of subplots using matplotlib.
    - Returns all complex propagated fields as a stacked tensor.

    Returns:
    --------
    torch.Tensor
        A tensor containing the complex propagated fields at each z distance.
        Shape: [num_z_steps, Nx, Nx], dtype: complex64.
    """
            
    # print("E0 contains NaN:", torch.isnan(E0).any().item())
    # print("Mask contains NaN:", torch.isnan(mask).any().item())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    z_values = np.arange(z_start, z_end + z_step, z_step)
    
    propagated_fields = []

    num_plots = len(z_values)
    cols = min(num_plots, 5)  
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, z in enumerate(z_values):
        E_propagated = propagation_multi(
            E0, z=z, wavelengths=lam,
            pixel_size=dx, device=device
        )
        amplitude = torch.abs(E_propagated)       # (C,H,W) or (H,W)

        if amplitude.dim() == 4 and amplitude.shape[0] == 1:
            amplitude = amplitude.squeeze(0)

        if amplitude.dim() == 3:   # 多通道 (C,H,W)
            amp_show = amplitude.sum(dim=0)  # 合并成 (H,W)
        else:
            amp_show = amplitude           # 已经是 (H,W)

        propagated_fields.append(E_propagated.cpu())

        ax = axes[idx]
        ax.imshow(amp_show.cpu().numpy(), cmap='viridis')
        ax.set_title(f"z = {z*1e6:.0f} µm", fontsize=12)
        ax.axis('off')


    # hide the rest subplot
    for idx in range(len(z_values), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    #plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
    plt.show()
    
    # transfer to Tensor and return
    return torch.stack(propagated_fields)


# #%% test 
# if 0:
        
#     test_dataset = train_dataset
#     propagated_fieds = []
#     zo = 10e-6 # from the light field to first layer
#     zo  = 0
#     z_start = 0  
#     z_layer= 40e-6 #47.2356e-3#45.5768e-3#8e-2#16e-2  
#     z_end = z_layer
#     z_step = 20e-6#1e-2 
#     z_prop = 300e-6#0#5e-2
#     temp_E_index = 1
#     temp_E = test_dataset[temp_E_index][0].squeeze()
    
#     for i_model in range(len(all_phase_masks)):
    
#         print(f'\nVisulizing model {i_model + 1}/{len(all_phase_masks)} (Model index: {i_model})')
    
#         temp_model =  all_phase_masks[i_model]
#         propagated_fieds = []
#         # input 
#         Eo = temp_E
#         # save input to the list 
#         propagated_fieds.append(Eo)
        
#         #first propagation: from input to the first layer, 
#         temp_fields = plot_propagated_field(Eo,z_start,zo,z_step, pixel_size, wavelength).to(device)
#         propagated_fieds.append(temp_fields)
#         Ei=propagation(Eo, zo,wavelength,Eo.size()[0],pixel_size,device)
#         propagated_fieds.append(Ei)
#         for i_layer in range(len(temp_model)):
#             print(f'  Layer {i_layer + 1}/{len(temp_model)}...')
    
#             # calcualte the propogation after the last layer        
#             # read the masks one by one
#             temp_mask = torch.from_numpy(temp_model[i_layer]).to(device) 
#             # light modulation using phase masks
#             Ei =  Ei * torch.exp(1j * temp_mask) 
#             # propagated_fieds.append(Ei)
            
#             # calculate the light fields during the propogation 
#             temp_fields = plot_propagated_field(Ei, z_start,z_end,z_step, pixel_size, wavelength).to(device)
#             propagated_fieds.append(temp_fields)
#             # calculate the light after propogation 
#             Ei=propagation(Ei, z_end,wavelength,Ei.size()[0],pixel_size,device)
#             propagated_fieds.append(Ei)
#             # calculate the propogation before the last layer
#             if i_layer == len(temp_model)-1:
#                 temp_fields = plot_propagated_field(Ei, z_start,z_prop,z_step, pixel_size, wavelength).to(device)
#                 propagated_fieds.append(temp_fields)
#                 Ei=propagation(Ei, z_end,wavelength,Ei.size()[0],pixel_size,device)
#                 propagated_fieds.append(Ei)
#                 print('end')
