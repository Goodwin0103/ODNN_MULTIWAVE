import os
import re
import torch
import torch.nn as nn
import numpy as np
from light_propagation_simulation_qz import propagation_multi
import matplotlib.pyplot as plt

class WavelengthDependentDiffractionLayer(nn.Module):
    def __init__(self, units: int, dx: float, wavelengths: np.ndarray, z: float, layer_idx: int = 0):
        super().__init__()
        self.units = units
        self.dx = dx
        self.wavelengths = wavelengths
        self.z = z
        self.layer_idx = layer_idx
        self.register_buffer("lam_list", torch.as_tensor(wavelengths, dtype=torch.float32))
        
        # 基础相位掩膜
        self.phase = nn.Parameter(torch.rand(units, units) * np.pi)  
        
        # 设置基准波长索引（通常选择中间波长）
        self.base_wavelength_idx = len(wavelengths) // 2
        
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
            
            # 能量保持的相位调制
            phase_modulation = torch.exp(1j * phase_scaled)
            x_c = x_c * phase_modulation
            
            # 能量归一化
            input_energy = torch.mean(torch.abs(x_c)**2)
            
            x_c = propagation_multi(
                x_c, z=self.z,
                wavelengths=[self.lam_list[c]], 
                pixel_size=self.dx, 
                device=x.device
            )
            
            # 能量恢复
            output_energy = torch.mean(torch.abs(x_c)**2)
            if output_energy > 1e-8:
                energy_ratio = torch.sqrt(input_energy / output_energy)
                x_c = x_c * energy_ratio.clamp(0.5, 2.0)
            
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
            'coefficient_type': 'wavelength_inverse'
        }

class WavelengthDependentPropagation(nn.Module):
    def __init__(self, dx: float, wavelengths: np.ndarray, z: float):
        super().__init__()
        self.dx = dx
        self.wavelengths = wavelengths
        self.z = z
        self.register_buffer("lam_list", torch.as_tensor(wavelengths, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        if C != len(self.wavelengths):
            print(f"警告: 传播层中输入通道数({C})与波长数量({len(self.wavelengths)})不匹配")
        
        results = []
        
        for c in range(min(C, len(self.wavelengths))):
            x_c = x[:, c:c+1]
            x_c = propagation_multi(
                x_c, z=self.z, 
                wavelengths=[self.lam_list[c]],
                pixel_size=self.dx, 
                device=x.device
            )
            results.append(x_c)
        
        output = torch.cat(results, dim=1)
        return output

class RegressionDetector(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.abs(inputs))

class WavelengthDependentD2NNModel(nn.Module):
    def __init__(self, config, num_layers: int):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            WavelengthDependentDiffractionLayer(
                config.layer_size, config.pixel_size,
                config.wavelengths, config.z_layers, 
                layer_idx=i
            ) for i in range(num_layers)
        ])
        
        self.propagation = WavelengthDependentPropagation(
            config.pixel_size, config.wavelengths, config.z_prop
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
            'z_prop': self.config.z_prop
        }
        
        np.savez(save_path, **masks_data)
        print(f"✓ 训练好的相位掩码已保存到: {save_path}")
        
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
            else:
                plt.show()

class PhysicsBasedMultiWavelengthLayer(nn.Module):
    def __init__(self, units, pixel_size, wavelengths, z_distance, num_modes, layer_idx=0):
        super().__init__()
        
        self.units = units
        self.pixel_size = pixel_size
        self.wavelengths = wavelengths
        self.z_distance = z_distance
        self.num_modes = num_modes
        self.layer_idx = layer_idx
        
        # 基础相位掩膜
        self.phase = nn.Parameter(torch.rand(units, units) * np.pi)  
        
        # 设置基准波长索引
        self.base_wavelength_idx = len(wavelengths) // 2
        
        # 根据波长反比关系计算相位延迟系数
        self.wavelength_coefficients = self._calculate_wavelength_coefficients(wavelengths)
        
        # 层深度衰减因子
        self.depth_factor = 1.0 ** layer_idx
        
        # 初始化传播器
        self.propagator = WavelengthDependentPropagation(
            pixel_size, wavelengths, z_distance
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
            
            # 使用物理计算的系数
            coefficient = self.wavelength_coefficients[i]
            effective_phase = self.phase * coefficient * self.depth_factor
            
            # 应用相位掩膜
            field = field * torch.exp(1j * effective_phase)
            
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
            'coefficient_type': 'wavelength_inverse'
        }

class MultiModeMultiWavelengthModel(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        
        self.layers = nn.ModuleList([
            PhysicsBasedMultiWavelengthLayer(
                config.layer_size, 
                config.pixel_size,
                config.wavelengths, 
                config.z_layers,
                config.num_modes, 
                layer_idx=i
            ) for i in range(num_layers)
        ])
        
        self.final_propagation = WavelengthDependentPropagation(
            config.pixel_size, config.wavelengths, config.z_prop
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
            'num_modes': getattr(self.config, 'num_modes', 3)
        }
        
        np.savez(save_path, **masks_data)
        print(f"✓ 训练好的相位掩码已保存到: {save_path}")
        
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
        """获取所有相位掩膜"""
        all_masks = []
        for i, layer in enumerate(self.layers):
            wavelength_masks = layer.get_effective_phase_masks()
            for j, mask in enumerate(wavelength_masks):
                all_masks.append(mask)
                print(f"层 {i} 相位掩膜 {j} (对应波长 {self.config.wavelengths[j]*1e9:.1f}nm):")
                print(f"  形状: {mask.shape}")
                print(f"  范围: [{mask.min():.4f}, {mask.max():.4f}]")
                print(f"  平均值: {mask.mean():.4f}")
                print(f"  标准差: {mask.std():.4f}")
        return all_masks
    
    def print_phase_masks(self, save_path=None):
        """打印所有层的相位掩膜"""
        print("\n====== 模型所有相位掩膜信息 ======")
        
        for i, layer in enumerate(self.layers):
            wavelength_masks = layer.get_effective_phase_masks()
            
            print(f"\n== 层 {i} 相位掩膜信息 ==")
            print(f"基础相位掩膜形状: {wavelength_masks[0].shape}")
            print(f"基础相位掩膜范围: [{np.min(wavelength_masks[0]):.4f}, {np.max(wavelength_masks[0]):.4f}]")
            print(f"波长系数: {layer.wavelength_coefficients.detach().cpu().numpy()}")
            
            n_wavelengths = len(self.config.wavelengths)
            fig, axes = plt.subplots(1, n_wavelengths, figsize=(5*n_wavelengths, 5))
            
            if n_wavelengths == 1:
                axes = [axes]
            
            for j, (mask, wl) in enumerate(zip(wavelength_masks, self.config.wavelengths)):
                im = axes[j].imshow(mask, cmap='viridis')
                axes[j].set_title(f"λ={wl*1e9:.0f}nm\nCoef={layer.wavelength_coefficients[j].item():.4f}")
                plt.colorbar(im, ax=axes[j], label='Phase (rad)')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/phase_mask_layer_{i}.png")
                print(f"已保存层 {i} 的相位掩膜图像到 {save_path}/phase_mask_layer_{i}.png")
            else:
                plt.show()
