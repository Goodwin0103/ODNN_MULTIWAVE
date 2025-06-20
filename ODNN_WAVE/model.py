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
        
        # 基础相位掩膜 - 减小初始范围
        self.phase = nn.Parameter(torch.rand(units, units) * np.pi)  
        
        # 设置基准波长索引（假设550nm是第二个波长）
        self.base_wavelength_idx = 1  # 假设wavelengths=[450nm, 550nm, 650nm]
        
        # 根据物理规律计算初始波长系数
        # 波长系数与波长成反比：550nm/λ
        base_wavelength = wavelengths[self.base_wavelength_idx]
        wavelength_ratios = [base_wavelength/wl for wl in wavelengths]
        
        # 可学习的波长系数，初始化为物理理论值
        self.wavelength_coefficients = nn.Parameter(
            torch.tensor(wavelength_ratios, dtype=torch.float32)
        )
        
        # 层深度衰减因子
        self.depth_factor = 1 ** layer_idx  # 深层使用更小的调制
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在每次前向传播时固定基准波长的系数为1.0
        with torch.no_grad():
            self.wavelength_coefficients[self.base_wavelength_idx] = 1.0
            
        # 其余代码保持不变
        B, C, H, W = x.shape
        results = []
        
        phase_base = self.phase.to(dtype=x.dtype, device=x.device)
        
        for c in range(C):
            # 约束相位系数
            phase_multiplier = torch.clamp(
                self.wavelength_coefficients[c], 
                0.5, 1.5  
            ) * self.depth_factor  # 深层衰减
            
            phase_scaled = phase_base * phase_multiplier
            
            # 使用固定的传播距离，或者根据波长调整
            adaptive_z = self.z
            
            x_c = x[:, c:c+1]
            
            # 能量保持的相位调制
            phase_modulation = torch.exp(1j * phase_scaled)
            x_c = x_c * phase_modulation
            
            # 能量归一化
            input_energy = torch.mean(torch.abs(x_c)**2)
            
            x_c = propagation_multi(
                x_c, z=adaptive_z,
                wavelengths=[self.lam_list[c]], 
                pixel_size=self.dx, 
                device=x.device
            )
            
            # 能量恢复
            output_energy = torch.mean(torch.abs(x_c)**2)
            if output_energy > 1e-8:
                energy_ratio = torch.sqrt(input_energy / output_energy)
                x_c = x_c * energy_ratio.clamp(0.5, 2.0)  # 限制能量补偿
            
            results.append(x_c)
        
        return torch.cat(results, dim=1)

    def get_phase_masks(self):
        """获取该层所有波长的相位掩膜"""
        base_mask = self.phase.detach().cpu().numpy()
        wavelength_masks = []
        
        for c in range(len(self.wavelengths)):
            phase_multiplier = torch.clamp(
                self.wavelength_coefficients[c], 
                0.5, 1.5
            ) * self.depth_factor
            
            scaled_mask = base_mask * phase_multiplier.item()
            wavelength_masks.append(scaled_mask)
        
        return {
            'layer_idx': self.layer_idx,
            'base_mask': base_mask,
            'wavelength_masks': wavelength_masks,
            'coefficients': self.wavelength_coefficients.detach().cpu().numpy(),
            'wavelengths': self.wavelengths
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
        
        # 检查通道数与波长数量是否匹配
        if C != len(self.wavelengths):
            print(f"警告: 传播层中输入通道数({C})与波长数量({len(self.wavelengths)})不匹配")
        
        results = []
        
        # 分别处理每个通道
        for c in range(min(C, len(self.wavelengths))):
            x_c = x[:, c:c+1]
            x_c = propagation_multi(
                x_c, z=self.z, 
                wavelengths=[self.lam_list[c]],  # 只传递对应通道的波长
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
            
            # 使用稳定化层
            self.layers = nn.ModuleList([
                WavelengthDependentDiffractionLayer(
                    config.layer_size, config.pixel_size,
                    config.wavelengths, config.z_layers, 
                    layer_idx=i  # 传递层索引
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

    def get_phase_masks(self):
        """获取该层所有波长的相位掩膜"""
        base_mask = self.phase.detach().cpu().numpy()
        wavelength_masks = []
        
        for c in range(len(self.wavelengths)):
            phase_multiplier = torch.clamp(
                self.wavelength_coefficients[c], 
                0.5, 1.5
            ) * self.depth_factor
            
            scaled_mask = base_mask * phase_multiplier.item()
            wavelength_masks.append(scaled_mask)
        
        return {
            'layer_idx': self.layer_idx,
            'base_mask': base_mask,
            'wavelength_masks': wavelength_masks,
            'coefficients': self.wavelength_coefficients.detach().cpu().numpy(),
            'wavelengths': self.wavelengths
        }

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
            print(f"波长系数: {coefficients}")
            
            # 创建图形
            n_wavelengths = len(wavelengths)
            fig, axes = plt.subplots(1, n_wavelengths + 1, figsize=(5*(n_wavelengths+1), 5))
            
            # 绘制基础相位掩膜
            im0 = axes[0].imshow(base_mask, cmap='viridis')
            axes[0].set_title(f"layer {i} base phase mask")
            plt.colorbar(im0, ax=axes[0], label='Phase (rad)')
            
            # 绘制每个波长的相位掩膜
            for j, (mask, coef, wl) in enumerate(zip(wavelength_masks, coefficients, wavelengths)):
                im = axes[j+1].imshow(mask, cmap='viridis')
                axes[j+1].set_title(f"Wavelength {wl*1e9:.1f}nm\nCoef: {coef:.4f}")
                plt.colorbar(im, ax=axes[j+1], label='Phase (rad)')
            
            plt.tight_layout()
            
            if save_path:
                # 不要在save_path后面再添加子目录
                plt.savefig(f"{save_path}/phase_mask_layer_{i}.png")
                print(f"已保存层 {i} 的相位掩膜图像到 {save_path}/phase_mask_layer_{i}.png")
            else:
                plt.show()

    def get_all_phase_masks(self):
        """获取所有层的相位掩膜数据"""
        all_masks = []
        for layer in self.layers:
            all_masks.append(layer.get_phase_masks())
        return all_masks
    

import torch
import torch.nn as nn
import numpy as np
from light_propagation_simulation_qz import propagation_multi

class ModeModeWavelengthDependentDiffractionLayer(nn.Module):
    def __init__(self, units: int, dx: float, wavelengths: np.ndarray, z: float, 
                 num_modes: int, layer_idx: int = 0):
        super().__init__()
        self.units = units
        self.dx = dx
        self.wavelengths = wavelengths
        self.z = z
        self.layer_idx = layer_idx
        self.num_modes = num_modes
        
        # 注册波长列表为缓冲区
        self.register_buffer("lam_list", torch.as_tensor(wavelengths, dtype=torch.float32))
        
        # 为每个模式创建基础相位掩膜
        self.phase_masks = nn.ParameterList([
            nn.Parameter(torch.rand(units, units) * np.pi)  
            for _ in range(num_modes)
        ])
        
        # 为每个模式创建波长系数
        self.wavelength_coefficients = nn.ParameterList([
            nn.Parameter(torch.ones(len(wavelengths), dtype=torch.float32))
            for _ in range(num_modes)
        ])
        
        # 设置基准波长索引
        self.base_wavelength_idx = 1  # 假设wavelengths=[450nm, 550nm, 650nm]
        
        # 层深度衰减因子
        self.depth_factor = 1.0 / (1.0 + 0.1 * layer_idx)  # 深层使用更小的调制
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理多波长输入
        
        参数:
            x: 形状为 [B, C, H, W] 的张量，其中:
               B - 批次大小 (等于模式数量)
               C - 波长通道数
               H, W - 高度和宽度
        
        返回:
            形状相同的输出张量
        """
        B, C, H, W = x.shape
        results = []
        
        # 处理每个模式(批次)
        for b in range(B):
            # 固定基准波长的系数为1.0
            with torch.no_grad():
                self.wavelength_coefficients[b][self.base_wavelength_idx] = 1.0
            
            # 获取该模式的相位掩膜
            phase_base = self.phase_masks[b].to(dtype=torch.float32, device=x.device)
            
            # 处理该模式的所有波长
            mode_results = []
            for c in range(C):
                # 约束相位系数
                phase_multiplier = torch.clamp(
                    self.wavelength_coefficients[b][c], 
                    0.5, 1.5  
                ) * self.depth_factor
                
                # 调整相位掩膜
                phase_scaled = phase_base * phase_multiplier
                
                # 提取当前模式和波长的数据
                x_bc = x[b:b+1, c:c+1]
                
                # 相位调制
                phase_modulation = torch.exp(1j * phase_scaled)
                x_bc = x_bc * phase_modulation
                
                # 能量归一化
                input_energy = torch.mean(torch.abs(x_bc)**2)
                
                # 传播
                x_bc = propagation_multi(
                    x_bc, z=self.z,
                    wavelengths=[self.lam_list[c]], 
                    pixel_size=self.dx, 
                    device=x.device
                )
                
                # 能量恢复
                output_energy = torch.mean(torch.abs(x_bc)**2)
                if output_energy > 1e-8:
                    energy_ratio = torch.sqrt(input_energy / output_energy)
                    x_bc = x_bc * energy_ratio.clamp(0.5, 2.0)
                
                mode_results.append(x_bc)
            
            # 合并该模式的所有波长结果
            mode_output = torch.cat(mode_results, dim=1)
            results.append(mode_output)
        
        # 合并所有模式的结果
        return torch.cat(results, dim=0)
    
    def get_mode_specific_phase_masks(self):
        """获取所有模式的相位掩膜"""
        mode_masks = []
        
        for m in range(self.num_modes):
            base_mask = self.phase_masks[m].detach().cpu().numpy()
            wavelength_masks = []
            
            for c in range(len(self.wavelengths)):
                phase_multiplier = torch.clamp(
                    self.wavelength_coefficients[m][c], 
                    0.5, 1.5
                ) * self.depth_factor
                
                scaled_mask = base_mask * phase_multiplier.item()
                wavelength_masks.append(scaled_mask)
            
            mode_masks.append(wavelength_masks)
        
        return mode_masks

class WavelengthDependentPropagation(nn.Module):
    def __init__(self, dx: float, wavelengths: np.ndarray, z: float):
        super().__init__()
        self.dx = dx
        self.wavelengths = wavelengths
        self.z = z
        self.register_buffer("lam_list", torch.as_tensor(wavelengths, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 形状为 [B, C, H, W] 的张量
        """
        B, C, H, W = x.shape
        results = []
        
        # 分别处理每个批次(模式)
        for b in range(B):
            batch_results = []
            
            # 分别处理每个通道(波长)
            for c in range(C):
                x_bc = x[b:b+1, c:c+1]
                
                # 传播
                x_bc = propagation_multi(
                    x_bc, z=self.z, 
                    wavelengths=[self.lam_list[c]],
                    pixel_size=self.dx, 
                    device=x.device
                )
                
                batch_results.append(x_bc)
            
            # 合并该批次的所有通道
            batch_output = torch.cat(batch_results, dim=1)
            results.append(batch_output)
        
        # 合并所有批次
        return torch.cat(results, dim=0)

class MultiModeMultiWavelengthModel(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        
        # 使用物理原理的多波长衍射层
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
        
        # 最终传播和探测器
        self.final_propagation = WavelengthDependentPropagation(
            config.pixel_size, config.wavelengths, config.z_prop
        )
        self.detector = RegressionDetector()
        self.config = config
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入光场，形状为 [B, C, H, W]
               B - 批次大小 (可以是模式数量)
               C - 波长通道数
               H, W - 空间分辨率
        """
        # 通过所有衍射层
        for layer in self.layers:
            x = layer(x)
        
        # 最终传播和探测
        x = self.final_propagation(x)
        output = self.detector(x)
        
        return output
    
    def get_all_phase_masks(self):
        """获取所有相位掩膜（仅用于可视化和分析）"""
        all_masks = []
        for i, layer in enumerate(self.layers):
            # 获取每个波长的有效相位掩膜
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
            # 获取每个波长的有效相位掩膜
            wavelength_masks = layer.get_effective_phase_masks()
            
            print(f"\n== 层 {i} 相位掩膜信息 ==")
            print(f"基础相位掩膜形状: {wavelength_masks[0].shape}")
            print(f"基础相位掩膜范围: [{np.min(wavelength_masks[0]):.4f}, {np.max(wavelength_masks[0]):.4f}]")
            print(f"波长系数: {layer.wavelength_coefficients.detach().cpu().numpy()}")
            
            # 创建图形
            n_wavelengths = len(self.config.wavelengths)
            fig, axes = plt.subplots(1, n_wavelengths, figsize=(5*n_wavelengths, 5))
            
            # 如果只有一个波长，确保axes是列表
            if n_wavelengths == 1:
                axes = [axes]
            
            # 绘制每个波长的相位掩膜
            for j, (mask, wl) in enumerate(zip(wavelength_masks, self.config.wavelengths)):
                im = axes[j].imshow(mask, cmap='viridis')
                axes[j].set_title(f"Wavelength {wl*1e9:.1f}nm\nCoef: {layer.wavelength_coefficients[j].item():.4f}")
                plt.colorbar(im, ax=axes[j], label='Phase (rad)')
            
            plt.tight_layout()
            
            if save_path:
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
        
        # 基础相位掩膜 - 减小初始范围
        self.phase = nn.Parameter(torch.rand(units, units) * np.pi)  
        
        # 设置基准波长索引（假设550nm是第二个波长）
        self.base_wavelength_idx = 1  # 假设wavelengths=[450nm, 550nm, 650nm]
        
        # 根据物理规律计算初始波长系数
        # 波长系数与波长成反比：550nm/λ
        base_wavelength = wavelengths[self.base_wavelength_idx]
        wavelength_ratios = [base_wavelength/wl for wl in wavelengths]
        
        # 可学习的波长系数，初始化为物理理论值
        self.wavelength_coefficients = nn.Parameter(
            torch.tensor(wavelength_ratios, dtype=torch.float32)
        )
        
        # 层深度衰减因子
        self.depth_factor = 1 ** layer_idx  # 深层使用更小的调制
        
        # 初始化传播器
        self.propagator = WavelengthDependentPropagation(
            pixel_size, wavelengths, z_distance
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入光场，形状为 [B, C, H, W]
               B - 批次大小
               C - 波长通道数
               H, W - 空间分辨率
        """
        batch_size, num_wavelengths, height, width = x.shape
        
        # 确保波长数量匹配
        assert num_wavelengths == len(self.wavelengths), \
            f"输入波长通道数 {num_wavelengths} 与配置的波长数 {len(self.wavelengths)} 不匹配"
        
        # 分别处理每个波长
        outputs = []
        for i in range(num_wavelengths):
            # 获取当前波长的光场
            field = x[:, i:i+1, :, :]  # 保持维度 [B, 1, H, W]
            
            # 计算当前波长的有效相位掩膜
            coefficient = self.wavelength_coefficients[i]
            effective_phase = self.phase * coefficient * self.depth_factor
            
            # 应用对应波长的相位掩膜
            field = field * torch.exp(1j * effective_phase)
            
            outputs.append(field)
        
        # 合并所有波长的结果
        combined = torch.cat(outputs, dim=1)
        
        # 传播
        propagated = self.propagator(combined)
        
        return propagated
    
    def get_effective_phase_masks(self):
        """获取每个波长的有效相位掩膜（用于可视化和分析）"""
        effective_phases = []
        
        for i, wl in enumerate(self.wavelengths):
            # 根据波长系数调整基础相位
            coefficient = self.wavelength_coefficients[i]
            effective_phase = self.phase * coefficient * self.depth_factor
            
            # 转换为NumPy数组用于返回
            phase_np = effective_phase.detach().cpu().numpy()
            effective_phases.append(phase_np)
        
        return effective_phases

