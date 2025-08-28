import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from light_propagation_simulation_qz import propagation_multi
import matplotlib.pyplot as plt

def apply_zero_padding(field, padding_ratio=0.2):
    """
    对光场应用 zero padding
    
    Args:
        field: 输入光场 (B, C, H, W) 或 (H, W)
        padding_ratio: padding 比例，例如 0.2 表示在每边添加 20% 的零
    
    Returns:
        padded_field: 添加 padding 后的光场
        original_slice: 用于恢复原始尺寸的切片信息
    """
    if field.dim() == 2:
        # 处理 2D 相位掩膜
        H, W = field.shape
        pad_h = int(H * padding_ratio)
        pad_w = int(W * padding_ratio)
        padded_field = F.pad(field, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        original_slice = (slice(pad_h, pad_h + H), slice(pad_w, pad_w + W))
    else:
        # 处理 4D 光场
        B, C, H, W = field.shape
        pad_h = int(H * padding_ratio)
        pad_w = int(W * padding_ratio)
        padded_field = F.pad(field, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        original_slice = (slice(pad_h, pad_h + H), slice(pad_w, pad_w + W))
    
    return padded_field, original_slice

def remove_padding(field, original_slice):
    """
    移除 padding，恢复到原始尺寸
    
    Args:
        field: 带 padding 的光场
        original_slice: 原始区域的切片信息
    
    Returns:
        cropped_field: 裁剪后的光场
    """
    return field[..., original_slice[0], original_slice[1]]

def apply_boundary_apodization(field, apodization_width=10):
    """
    在边界应用渐变衰减，避免硬截断
    
    Args:
        field: 输入光场
        apodization_width: 衰减区域宽度
    
    Returns:
        apodized_field: 应用边界衰减后的光场
    """
    if field.dim() == 2:
        H, W = field.shape
    else:
        H, W = field.shape[-2:]
    
    # 创建衰减掩膜
    y, x = torch.meshgrid(torch.arange(H, device=field.device), 
                         torch.arange(W, device=field.device), indexing='ij')
    
    # 计算到边界的最小距离
    dist_to_edge = torch.minimum(
        torch.minimum(x, W - 1 - x),
        torch.minimum(y, H - 1 - y)
    ).float()
    
    # 创建平滑的衰减函数
    apodization_mask = torch.clamp(dist_to_edge / apodization_width, 0, 1)
    
    return field * apodization_mask

class WavelengthDependentDiffractionLayer(nn.Module):
    def __init__(self, units: int, dx: float, wavelengths: np.ndarray, z: float, 
                 layer_idx: int = 0, padding_ratio: float = 0.2, 
                 use_apodization: bool = True, apodization_width: int = 10):
        super().__init__()
        self.units = units
        self.dx = dx
        self.wavelengths = wavelengths
        self.z = z
        self.layer_idx = layer_idx
        self.padding_ratio = padding_ratio  # 新增：padding 比例
        self.use_apodization = use_apodization  # 新增：是否使用边界衰减
        self.apodization_width = apodization_width  # 新增：衰减宽度
        
        self.register_buffer("lam_list", torch.as_tensor(wavelengths, dtype=torch.float32))
        
        # 基础相位掩膜
        self.phase = nn.Parameter(torch.rand(units, units) * np.pi)  
        
        # 设置基准波长索引 选择最大波长
        self.base_wavelength_idx = 2
        
        # 根据波长反比关系计算相位延迟系数
        self.wavelength_coefficients = self._calculate_wavelength_coefficients(wavelengths)
        
        # 层深度衰减因子
        self.depth_factor = 1.0 ** layer_idx
        
    def _calculate_wavelength_coefficients(self, wavelengths):
        """
        根据波长反比关系计算相位延迟系数
        相位延迟 φ = 2π·OPD/λ，因此系数与波长成反比
        """
        base_wavelength = wavelengths[self.base_wavelength_idx]
        coefficients = []
        
        print(f"计算波长系数 (基准波长: {base_wavelength*1e9:.1f}nm)")
        
        for wl in wavelengths:
            # 波长反比关系：λ₀/λ
            coef = base_wavelength / wl
            coefficients.append(coef)
            print(f"  波长 {wl*1e9:.1f}nm: 系数 = {coef:.4f}")
        
        # 转换为torch张量并注册为buffer（不参与训练）
        coefficients = torch.tensor(coefficients, dtype=torch.float32)
        self.register_buffer("wavelength_coefficients", coefficients)
        
        return coefficients
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        results = []
        
        phase_base = self.phase.to(dtype=x.dtype, device=x.device)
        
        for c in range(C):
            # 使用计算得到的物理系数（不参与训练）
            phase_multiplier = self.wavelength_coefficients[c] * self.depth_factor
            
            phase_scaled = phase_base * phase_multiplier
            
            x_c = x[:, c:c+1]
            
            # 1. 应用 zero padding
            x_c_padded, original_slice = apply_zero_padding(x_c, self.padding_ratio)
            
            # 2. 相位掩膜也需要相应地 padding（用零填充）
            phase_scaled_padded, _ = apply_zero_padding(phase_scaled, self.padding_ratio)
            
            # 3. 可选：应用边界衰减，避免硬截断效应
            if self.use_apodization:
                x_c_padded = apply_boundary_apodization(x_c_padded, self.apodization_width)
            
            # 4. 能量保持的相位调制
            phase_modulation = torch.exp(1j * phase_scaled_padded)
            x_c_padded = x_c_padded * phase_modulation
            
            # 5. 能量归一化
            input_energy = torch.mean(torch.abs(x_c_padded)**2)
            
            # 6. 传播（在 padding 后的空间中进行）
            x_c_padded = propagation_multi(
                x_c_padded, z=self.z,
                wavelengths=[self.lam_list[c]], 
                pixel_size=self.dx, 
                device=x.device
            )
            
            # 7. 能量恢复
            output_energy = torch.mean(torch.abs(x_c_padded)**2)
            if output_energy > 1e-8:
                energy_ratio = torch.sqrt(input_energy / output_energy)
                x_c_padded = x_c_padded * energy_ratio.clamp(0.5, 2.0)
            
            # 8. 移除 padding，恢复到原始尺寸
            x_c = remove_padding(x_c_padded, original_slice)
            
            results.append(x_c)
        
        return torch.cat(results, dim=1)

    def get_phase_masks(self):
        """获取该层所有波长的相位掩膜"""
        base_mask = self.phase.detach().cpu().numpy()
        wavelength_masks = []
        
        for c in range(len(self.wavelengths)):
            phase_multiplier = self.wavelength_coefficients[c] * self.depth_factor
            
            scaled_mask = base_mask * phase_multiplier.item()
            # 确保相位在[0, 2π]范围内
            scaled_mask = np.mod(scaled_mask, 2 * np.pi)
            wavelength_masks.append(scaled_mask)
        
        return {
            'layer_idx': self.layer_idx,
            'base_mask': base_mask,
            'wavelength_masks': wavelength_masks,
            'coefficients': self.wavelength_coefficients.detach().cpu().numpy(),
            'wavelengths': self.wavelengths,
            'coefficient_type': 'wavelength_inverse',
            'padding_ratio': self.padding_ratio,
            'use_apodization': self.use_apodization
        }

class WavelengthDependentPropagation(nn.Module):
    def __init__(self, dx: float, wavelengths: np.ndarray, z: float, 
                 padding_ratio: float = 0.2, use_apodization: bool = True, 
                 apodization_width: int = 10):
        super().__init__()
        self.dx = dx
        self.wavelengths = wavelengths
        self.z = z
        self.padding_ratio = padding_ratio
        self.use_apodization = use_apodization
        self.apodization_width = apodization_width
        self.register_buffer("lam_list", torch.as_tensor(wavelengths, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        if C != len(self.wavelengths):
            print(f"警告: 传播层中输入通道数({C})与波长数量({len(self.wavelengths)})不匹配")
        
        results = []
        
        for c in range(min(C, len(self.wavelengths))):
            x_c = x[:, c:c+1]
            
            # 1. 应用 zero padding
            x_c_padded, original_slice = apply_zero_padding(x_c, self.padding_ratio)
            
            # 2. 可选：应用边界衰减
            if self.use_apodization:
                x_c_padded = apply_boundary_apodization(x_c_padded, self.apodization_width)
            
            # 3. 传播
            x_c_padded = propagation_multi(
                x_c_padded, z=self.z, 
                wavelengths=[self.lam_list[c]],
                pixel_size=self.dx, 
                device=x.device
            )
            
            # 4. 移除 padding
            x_c = remove_padding(x_c_padded, original_slice)
            
            results.append(x_c)
        
        output = torch.cat(results, dim=1)
        return output

class RegressionDetector(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.abs(inputs))

class WavelengthDependentD2NNModel(nn.Module):
    def __init__(self, config, num_layers: int, padding_ratio: float = 0.2, 
                 use_apodization: bool = True, apodization_width: int = 10):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.padding_ratio = padding_ratio
        self.use_apodization = use_apodization
        self.apodization_width = apodization_width
        
        self.layers = nn.ModuleList([
            WavelengthDependentDiffractionLayer(
                config.layer_size, config.pixel_size,
                config.wavelengths, config.z_layers, 
                layer_idx=i, padding_ratio=padding_ratio,
                use_apodization=use_apodization,
                apodization_width=apodization_width
            ) for i in range(num_layers)
        ])
        
        self.propagation = WavelengthDependentPropagation(
            config.pixel_size, config.wavelengths, config.z_prop,
            padding_ratio=padding_ratio, use_apodization=use_apodization,
            apodization_width=apodization_width
        )
        self.regression = RegressionDetector()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        x = self.propagation(x)
        return self.regression(x)

    def get_phase_masks_for_simulation(self):
        """获取用于仿真的相位掩码格式"""
        masks = []
        
        for layer in self.layers:
            layer_data = layer.get_phase_masks()
            wavelength_masks = layer_data['wavelength_masks']
            masks.append(wavelength_masks)
        
        return masks

    def save_trained_masks(self, save_path):
        """保存训练好的相位掩码"""
        masks_data = {}
        
        for layer_idx, layer in enumerate(self.layers):
            layer_data = layer.get_phase_masks()
            masks_data[f'layer_{layer_idx}'] = layer_data['wavelength_masks']
            masks_data[f'layer_{layer_idx}_coefficients'] = layer_data['coefficients']
            masks_data[f'layer_{layer_idx}_base_mask'] = layer_data['base_mask']
        
        masks_data['config'] = {
            'wavelengths': self.config.wavelengths.tolist(),
            'num_layers': len(self.layers),
            'layer_size': self.config.layer_size,
            'pixel_size': self.config.pixel_size,
            'z_layers': self.config.z_layers,
            'z_prop': self.config.z_prop,
            'padding_ratio': self.padding_ratio,
            'use_apodization': self.use_apodization,
            'apodization_width': self.apodization_width
        }
        
        np.savez(save_path, **masks_data)
        print(f"✓ 训练好的相位掩码已保存到: {save_path}")
        print(f"  - Padding ratio: {self.padding_ratio}")
        print(f"  - Use apodization: {self.use_apodization}")
        print(f"  - Apodization width: {self.apodization_width}")
        
        return save_path

    @classmethod
    def load_trained_masks(cls, load_path):
        """加载训练好的相位掩码"""
        try:
            data = np.load(load_path, allow_pickle=True)
            
            masks = []
            config_data = data['config'].item()
            num_layers = config_data['num_layers']
            
            print(f"✓ 加载掩码配置:")
            print(f"  层数: {num_layers}")
            print(f"  波长数: {len(config_data['wavelengths'])}")
            print(f"  掩码尺寸: {config_data['layer_size']}")
            print(f"  Padding ratio: {config_data.get('padding_ratio', 0.2)}")
            print(f"  Use apodization: {config_data.get('use_apodization', True)}")
            print(f"  Apodization width: {config_data.get('apodization_width', 10)}")
            
            for layer_idx in range(num_layers):
                layer_key = f'layer_{layer_idx}'
                if layer_key in data:
                    layer_masks = data[layer_key]
                    masks.append(layer_masks)
                    print(f"  ✓ 加载第 {layer_idx+1} 层掩码: {len(layer_masks)} 个波长")
            
            print(f"✓ 成功加载训练好的相位掩码: {load_path}")
            return masks, config_data
            
        except Exception as e:
            print(f"✗ 加载掩码文件失败: {e}")
            return None, None

    def get_all_phase_masks(self):
        """获取所有层的相位掩膜数据"""
        all_masks = []
        for layer in self.layers:
            all_masks.append(layer.get_phase_masks())
        return all_masks

    def print_phase_masks(self, save_path=None):
        """打印所有层的相位掩膜"""
        print("\n====== 模型所有相位掩膜信息 ======")
        print(f"Padding ratio: {self.padding_ratio}")
        print(f"Use apodization: {self.use_apodization}")
        print(f"Apodization width: {self.apodization_width}")
        
        for i, layer in enumerate(self.layers):
            mask_data = layer.get_phase_masks()
            base_mask = mask_data['base_mask']
            wavelength_masks = mask_data['wavelength_masks']
            coefficients = mask_data['coefficients']
            wavelengths = mask_data['wavelengths']
            
            print(f"\n== 层 {i} 相位掩膜信息 ==")
            print(f"基础相位掩膜形状: {base_mask.shape}")
            print(f"基础相位掩膜范围: [{np.min(base_mask):.4f}, {np.max(base_mask):.4f}]")
            print(f"波长系数 (λ₀/λ): {coefficients}")
            
            # 创建图形
            n_wavelengths = len(wavelengths)
            fig, axes = plt.subplots(1, n_wavelengths + 1, figsize=(5*(n_wavelengths+1), 5))
            
            # 绘制基础相位掩膜
            im0 = axes[0].imshow(base_mask, cmap='viridis')
            axes[0].set_title(f"Layer {i} Base Phase")
            plt.colorbar(im0, ax=axes[0], label='Phase (rad)')
            
            # 绘制每个波长的相位掩膜
            for j, (mask, coef, wl) in enumerate(zip(wavelength_masks, coefficients, wavelengths)):
                im = axes[j+1].imshow(mask, cmap='viridis')
                axes[j+1].set_title(f"λ={wl*1e9:.0f}nm\nCoef={coef:.4f}")
                plt.colorbar(im, ax=axes[j+1], label='Phase (rad)')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/phase_mask_layer_{i}.png")
                print(f"已保存层 {i} 的相位掩膜图像到 {save_path}/phase_mask_layer_{i}.png")
            
            plt.close()

class PhysicsBasedMultiWavelengthLayer(nn.Module):
    def __init__(self, units, pixel_size, wavelengths, z_distance, num_modes, 
                 layer_idx=0, padding_ratio=0.2, use_apodization=True, 
                 apodization_width=10):
        super().__init__()
        
        self.units = units
        self.pixel_size = pixel_size
        self.wavelengths = wavelengths
        self.z_distance = z_distance
        self.num_modes = num_modes
        self.layer_idx = layer_idx
        self.padding_ratio = padding_ratio
        self.use_apodization = use_apodization
        self.apodization_width = apodization_width
        
        # 基础相位掩膜
        self.phase = nn.Parameter(torch.rand(units, units) * np.pi)  
        
        # 设置基准波长索引
        self.base_wavelength_idx = 2
        
        # 根据波长反比关系计算相位延迟系数
        self.wavelength_coefficients = self._calculate_wavelength_coefficients(wavelengths)
        
        # 层深度衰减因子
        self.depth_factor = 1.0 ** layer_idx
        
        # 初始化传播器
        self.propagator = WavelengthDependentPropagation(
            pixel_size, wavelengths, z_distance, padding_ratio, 
            use_apodization, apodization_width
        )
    
    def _calculate_wavelength_coefficients(self, wavelengths):
        """
        根据波长反比关系计算相位延迟系数
        """
        base_wavelength = wavelengths[self.base_wavelength_idx]
        coefficients = []
        
        print(f"计算波长系数 (基准波长: {base_wavelength*1e9:.1f}nm)")
        
        for wl in wavelengths:
            # 波长反比关系：λ₀/λ
            coef = base_wavelength / wl
            coefficients.append(coef)
            print(f"  波长 {wl*1e9:.1f}nm: 系数 = {coef:.4f}")
        
        # 注册为buffer（不参与训练）
        coefficients = torch.tensor(coefficients, dtype=torch.float32)
        self.register_buffer("wavelength_coefficients", coefficients)
        
        return coefficients
    
    def forward(self, x):
        """前向传播"""
        batch_size, num_wavelengths, height, width = x.shape
        
        assert num_wavelengths == len(self.wavelengths), \
            f"输入波长通道数 {num_wavelengths} 与配置的波长数 {len(self.wavelengths)} 不匹配"
        
        outputs = []
        for i in range(num_wavelengths):
            field = x[:, i:i+1, :, :]
            
            # 1. 应用 zero padding
            field_padded, original_slice = apply_zero_padding(field, self.padding_ratio)
            
            # 2. 相位掩膜也需要相应地 padding
            coefficient = self.wavelength_coefficients[i]
            effective_phase = self.phase * coefficient * self.depth_factor
            effective_phase_padded, _ = apply_zero_padding(effective_phase, self.padding_ratio)
            
            # 3. 可选：应用边界衰减
            if self.use_apodization:
                field_padded = apply_boundary_apodization(field_padded, self.apodization_width)
            
            # 4. 应用相位掩膜
            field_padded = field_padded * torch.exp(1j * effective_phase_padded)
            
            # 5. 移除 padding，恢复到原始尺寸
            field = remove_padding(field_padded, original_slice)
            
            outputs.append(field)
        
        combined = torch.cat(outputs, dim=1)
        propagated = self.propagator(combined)
        
        return propagated
    
    def get_effective_phase_masks(self):
        """获取每个波长的有效相位掩膜"""
        effective_phases = []
        
        for i, wl in enumerate(self.wavelengths):
            coefficient = self.wavelength_coefficients[i]
            effective_phase = self.phase * coefficient * self.depth_factor
            
            phase_np = effective_phase.detach().cpu().numpy()
            phase_np = np.mod(phase_np, 2 * np.pi)
            effective_phases.append(phase_np)
        
        return effective_phases

    def get_phase_masks(self):
        """获取该层所有波长的相位掩膜"""
        base_mask = self.phase.detach().cpu().numpy()
        wavelength_masks = self.get_effective_phase_masks()
        
        return {
            'layer_idx': self.layer_idx,
            'base_mask': base_mask,
            'wavelength_masks': wavelength_masks,
            'coefficients': self.wavelength_coefficients.detach().cpu().numpy(),
            'wavelengths': self.wavelengths,
            'coefficient_type': 'wavelength_inverse',
            'padding_ratio': self.padding_ratio,
            'use_apodization': self.use_apodization
        }

class MultiModeMultiWavelengthModel(nn.Module):
    def __init__(self, config, num_layers, padding_ratio=0.2, use_apodization=True, 
                 apodization_width=10):
        super().__init__()
        
        self.padding_ratio = padding_ratio
        self.use_apodization = use_apodization
        self.apodization_width = apodization_width
        
        self.layers = nn.ModuleList([
            PhysicsBasedMultiWavelengthLayer(
                config.layer_size, 
                config.pixel_size,
                config.wavelengths, 
                config.z_layers,
                config.num_modes, 
                layer_idx=i,
                padding_ratio=padding_ratio,
                use_apodization=use_apodization,
                apodization_width=apodization_width
            ) for i in range(num_layers)
        ])
        
        self.final_propagation = WavelengthDependentPropagation(
            config.pixel_size, config.wavelengths, config.z_prop,
            padding_ratio, use_apodization, apodization_width
        )
        self.detector = RegressionDetector()
        self.config = config
    
    def forward(self, x):
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_propagation(x)
        output = self.detector(x)
        
        return output

    def get_phase_masks_for_simulation(self):
        """获取用于仿真的相位掩码格式"""
        masks = []
        
        for layer in self.layers:
            layer_data = layer.get_phase_masks()
            wavelength_masks = layer_data['wavelength_masks']
            masks.append(wavelength_masks)
        
        return masks

    def save_trained_masks(self, save_path):
        """保存训练好的相位掩码"""
        masks_data = {}
        
        for layer_idx, layer in enumerate(self.layers):
            layer_data = layer.get_phase_masks()
            masks_data[f'layer_{layer_idx}'] = layer_data['wavelength_masks']
            masks_data[f'layer_{layer_idx}_coefficients'] = layer_data['coefficients']
            masks_data[f'layer_{layer_idx}_base_mask'] = layer_data['base_mask']
        
        masks_data['config'] = {
            'wavelengths': self.config.wavelengths.tolist(),
            'num_layers': len(self.layers),
            'layer_size': self.config.layer_size,
            'pixel_size': self.config.pixel_size,
            'z_layers': self.config.z_layers,
            'z_prop': self.config.z_prop,
            'num_modes': getattr(self.config, 'num_modes', 3),
            'padding_ratio': self.padding_ratio,
            'use_apodization': self.use_apodization,
            'apodization_width': self.apodization_width
        }
        
        np.savez(save_path, **masks_data)
        print(f"✓ 训练好的相位掩码已保存到: {save_path}")
        print(f"  - Padding ratio: {self.padding_ratio}")
        print(f"  - Use apodization: {self.use_apodization}")
        print(f"  - Apodization width: {self.apodization_width}")
        
        return save_path

    @classmethod
    def load_trained_masks(cls, load_path):
        """加载训练好的相位掩码"""
        try:
            data = np.load(load_path, allow_pickle=True)
            
            masks = []
            config_data = data['config'].item()
            num_layers = config_data['num_layers']
            
            print(f"✓ 加载掩码配置:")
            print(f"  层数: {num_layers}")
            print(f"  波长数: {len(config_data['wavelengths'])}")
            print(f"  掩码尺寸: {config_data['layer_size']}")
            print(f"  Padding ratio: {config_data.get('padding_ratio', 0.2)}")
            print(f"  Use apodization: {config_data.get('use_apodization', True)}")
            print(f"  Apodization width: {config_data.get('apodization_width', 10)}")
            
            for layer_idx in range(num_layers):
                layer_key = f'layer_{layer_idx}'
                if layer_key in data:
                    layer_masks = data[layer_key]
                    masks.append(layer_masks)
                    print(f"  ✓ 加载第 {layer_idx+1} 层掩码: {len(layer_masks)} 个波长")
            
            print(f"✓ 成功加载训练好的相位掩码: {load_path}")
            return masks, config_data
            
        except Exception as e:
            print(f"✗ 加载掩码文件失败: {e}")
            return None, None

    def get_all_phase_masks(self):
        """获取所有层的相位掩膜数据"""
        all_masks = []
        for layer in self.layers:
            all_masks.append(layer.get_phase_masks())
        return all_masks

    def print_phase_masks(self, save_path=None):
        """打印所有层的相位掩膜"""
        print("\n====== 多模式多波长模型所有相位掩膜信息 ======")
        print(f"Padding ratio: {self.padding_ratio}")
        print(f"Use apodization: {self.use_apodization}")
        print(f"Apodization width: {self.apodization_width}")
        
        for i, layer in enumerate(self.layers):
            mask_data = layer.get_phase_masks()
            base_mask = mask_data['base_mask']
            wavelength_masks = mask_data['wavelength_masks']
            coefficients = mask_data['coefficients']
            wavelengths = mask_data['wavelengths']
            
            print(f"\n== 层 {i} 相位掩膜信息 ==")
            print(f"基础相位掩膜形状: {base_mask.shape}")
            print(f"基础相位掩膜范围: [{np.min(base_mask):.4f}, {np.max(base_mask):.4f}]")
            print(f"波长系数 (λ₀/λ): {coefficients}")
            
            # 创建图形
            n_wavelengths = len(wavelengths)
            fig, axes = plt.subplots(1, n_wavelengths + 1, figsize=(5*(n_wavelengths+1), 5))
            
            # 绘制基础相位掩膜
            im0 = axes[0].imshow(base_mask, cmap='viridis')
            axes[0].set_title(f"Layer {i} Base Phase")
            plt.colorbar(im0, ax=axes[0], label='Phase (rad)')
            
            # 绘制每个波长的相位掩膜
            for j, (mask, coef, wl) in enumerate(zip(wavelength_masks, coefficients, wavelengths)):
                im = axes[j+1].imshow(mask, cmap='viridis')
                axes[j+1].set_title(f"λ={wl*1e9:.0f}nm\nCoef={coef:.4f}")
                plt.colorbar(im, ax=axes[j+1], label='Phase (rad)')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/phase_mask_layer_{i}.png")
                print(f"已保存层 {i} 的相位掩膜图像到 {save_path}/phase_mask_layer_{i}.png")
            
            plt.close()