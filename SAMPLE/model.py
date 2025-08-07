import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from propagation import angular_spectrum_propagation

class ImprovedWavelengthSeparationLoss(nn.Module):
    """
    参考文档的改进损失函数
    集成：效率 + 分离 + 串扰 + 平滑 + 集中度
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.field_size = config.field_size
        self.detect_size = config.detect_size
        self.wavelengths = config.wavelengths
        self.offsets = config.offsets
        self.device = config.device
        
        # 🔥 改进的损失权重 - 参考文档优化
        self.efficiency_weight = 2.0      # 提高效率权重
        self.separation_weight = 1.5      # 分离损失权重  
        self.crosstalk_weight = 1.0       # 串扰损失权重
        self.concentration_weight = 0.8   # 能量集中损失权重
        self.smoothing_weight = 0.1       # 新增：相位平滑权重
        
    def forward(self, output_fields, phase_masks=None):
        """计算综合损失函数"""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 1. 效率损失 - 最大化目标区域能量
        efficiency_loss = self.compute_efficiency_loss(output_fields)
        
        # 2. 分离损失 - 确保不同波长聚焦在不同位置
        separation_loss = self.compute_separation_loss(output_fields)
        
        # 3. 串扰损失 - 最小化波长间干扰
        crosstalk_loss = self.compute_crosstalk_loss(output_fields)
        
        # 4. 能量集中损失
        concentration_loss = self.compute_concentration_loss(output_fields)
        
        # 5. 🔥 新增：相位平滑损失 - 参考文档
        smoothing_loss = torch.tensor(0.0, device=self.device)
        if phase_masks is not None:
            smoothing_loss = self.compute_phase_smoothing_loss(phase_masks)
        
        # 综合损失
        total_loss = (self.efficiency_weight * efficiency_loss +
                     self.separation_weight * separation_loss +
                     self.crosstalk_weight * crosstalk_loss +
                     self.concentration_weight * concentration_loss +
                     self.smoothing_weight * smoothing_loss)
        
        return {
            'total_loss': total_loss,
            'efficiency_loss': efficiency_loss,
            'separation_loss': separation_loss,
            'crosstalk_loss': crosstalk_loss,
            'concentration_loss': concentration_loss,
            'smoothing_loss': smoothing_loss
        }
    
    def compute_efficiency_loss(self, output_fields):
        """计算效率损失"""
        efficiency_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for w_idx, field in enumerate(output_fields):
            field = field.to(self.device)
            intensity = torch.abs(field)**2
            total_energy = torch.sum(intensity)
            
            # 获取目标区域
            offset_x, offset_y = self.offsets[w_idx]
            center_x = self.field_size // 2 + offset_x
            center_y = self.field_size // 2 + offset_y
            
            half_size = self.detect_size // 2
            x_start = max(0, center_x - half_size)
            x_end = min(self.field_size, center_x + half_size)
            y_start = max(0, center_y - half_size)
            y_end = min(self.field_size, center_y + half_size)
            
            target_energy = torch.sum(intensity[y_start:y_end, x_start:x_end])
            efficiency = target_energy / (total_energy + 1e-10)
            efficiency_loss = efficiency_loss + (1.0 - efficiency)
        
        return efficiency_loss / len(output_fields)
    
    def compute_separation_loss(self, output_fields):
        """改进的分离损失 - 使用高斯目标分布"""
        separation_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for w_idx, field in enumerate(output_fields):
            field = field.to(self.device)
            intensity = torch.abs(field)**2
            
            # 创建目标高斯分布
            offset_x, offset_y = self.offsets[w_idx]
            center_x = self.field_size // 2 + offset_x
            center_y = self.field_size // 2 + offset_y
            
            # 创建坐标网格
            y_coords = torch.arange(self.field_size, device=self.device).float()
            x_coords = torch.arange(self.field_size, device=self.device).float()
            Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # 🔥 改进：自适应高斯宽度
            sigma = self.detect_size / 2.5  # 调整高斯宽度
            target_gaussian = torch.exp(-((Y - center_y)**2 + (X - center_x)**2) / (2 * sigma**2))
            target_gaussian = target_gaussian / torch.sum(target_gaussian)
            
            # 归一化当前强度分布
            intensity_norm = intensity / (torch.sum(intensity) + 1e-10)
            
            # 🔥 改进：使用KL散度替代交叉熵
            kl_div = torch.sum(intensity_norm * torch.log((intensity_norm + 1e-10) / (target_gaussian + 1e-10)))
            separation_loss = separation_loss + kl_div
        
        return separation_loss / len(output_fields)
    
    def compute_crosstalk_loss(self, output_fields):
        """计算串扰损失"""
        crosstalk_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for w_idx, field in enumerate(output_fields):
            field = field.to(self.device)
            intensity = torch.abs(field)**2
            total_energy = torch.sum(intensity)
            
            # 计算在其他波长目标区域的能量
            for other_w_idx in range(len(output_fields)):
                if other_w_idx != w_idx:
                    other_offset_x, other_offset_y = self.offsets[other_w_idx]
                    other_center_x = self.field_size // 2 + other_offset_x
                    other_center_y = self.field_size // 2 + other_offset_y
                    
                    half_size = self.detect_size // 2
                    other_x_start = max(0, other_center_x - half_size)
                    other_x_end = min(self.field_size, other_center_x + half_size)
                    other_y_start = max(0, other_center_y - half_size)
                    other_y_end = min(self.field_size, other_center_y + half_size)
                    
                    other_target_energy = torch.sum(intensity[other_y_start:other_y_end, other_x_start:other_x_end])
                    crosstalk_ratio = other_target_energy / (total_energy + 1e-10)
                    
                    crosstalk_loss = crosstalk_loss + crosstalk_ratio
        
        return crosstalk_loss / (len(output_fields) * (len(output_fields) - 1))
    
    def compute_concentration_loss(self, output_fields):
        """计算能量集中损失"""
        concentration_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for w_idx, field in enumerate(output_fields):
            field = field.to(self.device)
            intensity = torch.abs(field)**2
            
            total_intensity = torch.sum(intensity)
            
            if total_intensity > 1e-10:
                # 计算质心
                y_coords = torch.arange(self.field_size, device=self.device).float()
                x_coords = torch.arange(self.field_size, device=self.device).float()
                Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
                
                center_y = torch.sum(intensity * Y) / total_intensity
                center_x = torch.sum(intensity * X) / total_intensity
                
                # 计算围绕质心的方差
                variance = torch.sum(intensity * ((Y - center_y)**2 + (X - center_x)**2)) / total_intensity
                concentration_loss = concentration_loss + variance
        
        return concentration_loss / len(output_fields)
    
    def compute_phase_smoothing_loss(self, phase_masks):
        """🔥 新增：相位平滑损失 - 参考文档技术"""
        smoothing_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Sobel算子用于计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
        for phase_mask in phase_masks:
            phase = phase_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            
            # 计算相位梯度
            grad_x = F.conv2d(phase, sobel_x, padding=1)
            grad_y = F.conv2d(phase, sobel_y, padding=1)
            
            # 平滑损失
            smoothness = torch.mean(grad_x**2 + grad_y**2)
            smoothing_loss = smoothing_loss + smoothness
        
        return smoothing_loss / len(phase_masks)


class ImprovedMultiWavelengthModel(nn.Module):
    """
    改进的多波长分离模型
    参考文档：独立波长优化 + 差分检测 + 相位平滑
    """
    def __init__(self, config, num_layers=1):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.field_size = config.field_size
        self.layer_size = config.layer_size
        self.pixel_size = config.pixel_size
        self.wavelengths = config.wavelengths
        self.device = config.device
        self.num_wavelengths = len(config.wavelengths)
        
        # 参考波长
        self.reference_wavelength = 550e-9
        
        # 🔥 核心改进1：每个波长独立的相位掩膜 - 参考文档
        self.wavelength_dependent_masks = nn.ParameterList([
            nn.Parameter(self._initialize_wavelength_phase_masks())
            for _ in range(num_layers)
        ])
        
        # 🔥 核心改进2：使用改进的损失函数
        self.criterion = ImprovedWavelengthSeparationLoss(config)
        
        # 传播距离
        self.propagation_distances = [50e-6] * num_layers
        
        # 🔥 改进3：添加差分检测机制
        self.use_differential_detection = True
        
    def _initialize_wavelength_phase_masks(self):
        """🔥 改进：为每个波长初始化独立的相位掩膜"""
        # 形状: [num_wavelengths, field_size, field_size]
        phase_masks = torch.randn(self.num_wavelengths, self.field_size, self.field_size, device=self.device)
        
        # 🔥 改进的初始化策略
        for w_idx in range(self.num_wavelengths):
            # 为不同波长添加不同的初始偏置
            offset_x, offset_y = self.config.offsets[w_idx]
            
            # 创建朝向目标区域的初始相位梯度
            y_coords = torch.arange(self.field_size, device=self.device).float()
            x_coords = torch.arange(self.field_size, device=self.device).float()
            Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            center_x = self.field_size // 2
            center_y = self.field_size // 2
            
            # 添加朝向目标的线性相位梯度
            phase_gradient = (offset_x * (X - center_x) + offset_y * (Y - center_y)) * 0.001
            phase_masks[w_idx] += phase_gradient
        
        return phase_masks * 0.3  # 减小初始相位变化幅度
    
    def get_wavelength_dependent_phase(self, wavelength_masks, w_idx, wavelength):
        """🔥 改进：获取特定波长的相位掩膜"""
        # 直接使用该波长对应的独立相位掩膜
        base_phase = wavelength_masks[w_idx]
        
        # 🔥 可选：仍然保留色散效应
        if hasattr(self.config, 'use_dispersion') and self.config.use_dispersion:
            wavelength_scale_factor = self.reference_wavelength / wavelength
            return base_phase * wavelength_scale_factor
        else:
            return base_phase
    
    def forward(self, input_fields):
        """处理多个波长的输入场"""
        output_fields = []
        
        for w_idx, input_field in enumerate(input_fields):
            input_field = input_field.to(self.device)
            wavelength = self.wavelengths[w_idx]
            output_field = self._process_single_wavelength(input_field, wavelength, w_idx)
            output_fields.append(output_field)
        
        # 🔥 改进：可选的差分检测
        if self.use_differential_detection and len(output_fields) == 2:
            output_fields = self._apply_differential_detection(output_fields)
            
        return output_fields
    
    def _process_single_wavelength(self, input_field, wavelength, w_idx):
        """处理单个波长的场"""
        current_field = input_field.to(self.device)
        
        for i in range(self.num_layers):
            # 🔥 使用该波长独立的相位掩膜
            wavelength_dependent_phase = self.get_wavelength_dependent_phase(
                self.wavelength_dependent_masks[i], w_idx, wavelength
            )
            
            # 应用相位掩膜
            current_field = current_field * torch.exp(1j * wavelength_dependent_phase)
            
            # 传播到下一层
            if i < self.num_layers - 1:
                current_field = angular_spectrum_propagation(
                    current_field, 
                    self.propagation_distances[i], 
                    wavelength,
                    self.config.pixel_size
                )
        
        # 最后一次传播到检测平面
        output_field = angular_spectrum_propagation(
            current_field,
            self.propagation_distances[-1],
            wavelength,
            self.config.pixel_size
        )
        
        return output_field
    
    def _apply_differential_detection(self, output_fields):
        """🔥 新增：差分检测机制 - 参考文档"""
        if len(output_fields) != 2:
            return output_fields
        
        field1, field2 = output_fields[0], output_fields[1]
        intensity1 = torch.abs(field1)**2
        intensity2 = torch.abs(field2)**2
        
        # 计算差分信号
        diff_signal = intensity1 - intensity2
        
        # 增强分离效果
        enhanced_field1 = field1 * torch.sqrt(torch.abs(diff_signal) + 1e-10)
        enhanced_field2 = field2 * torch.sqrt(torch.abs(-diff_signal) + 1e-10)
        
        return [enhanced_field1, enhanced_field2]
    
    def compute_loss(self, output_fields):
        """使用改进的损失函数"""
        # 收集所有相位掩膜用于平滑损失
        all_phase_masks = []
        for layer_masks in self.wavelength_dependent_masks:
            for w_idx in range(self.num_wavelengths):
                all_phase_masks.append(layer_masks[w_idx])
        
        return self.criterion(output_fields, all_phase_masks)['total_loss']
    
    def get_detailed_loss(self, output_fields):
        """获取详细的损失信息"""
        # 收集所有相位掩膜
        all_phase_masks = []
        for layer_masks in self.wavelength_dependent_masks:
            for w_idx in range(self.num_wavelengths):
                all_phase_masks.append(layer_masks[w_idx])
        
        return self.criterion(output_fields, all_phase_masks)
    
    def get_all_fields(self, input_fields):
        """计算并返回每一层的场分布"""
        num_wavelengths = len(input_fields)
        all_fields = []
        
        for w_idx in range(num_wavelengths):
            wavelength = self.wavelengths[w_idx]
            field = input_fields[w_idx].to(self.device)
            
            fields_per_wavelength = []
            
            for l_idx in range(len(self.wavelength_dependent_masks)):
                wavelength_dependent_phase = self.get_wavelength_dependent_phase(
                    self.wavelength_dependent_masks[l_idx], w_idx, wavelength
                )
                
                field = field * torch.exp(1j * wavelength_dependent_phase)
                
                if l_idx < len(self.wavelength_dependent_masks) - 1:
                    field = angular_spectrum_propagation(
                        field, 
                        self.propagation_distances[l_idx], 
                        wavelength,
                        self.config.pixel_size
                    )
                
                fields_per_wavelength.append(field)
            
            all_fields.append(fields_per_wavelength)
        
        return all_fields
    
    def get_phase_masks_for_visualization(self):
        """获取用于可视化的相位掩膜"""
        phase_masks_vis = []
        
        for layer_idx in range(self.num_layers):
            layer_masks = []
            for w_idx in range(self.num_wavelengths):
                phase_mask = self.wavelength_dependent_masks[layer_idx][w_idx]
                layer_masks.append(phase_mask.detach().cpu())
            phase_masks_vis.append(layer_masks)
        
        return phase_masks_vis
    
    def print_model_info(self):
        """打印模型信息"""
        print("=" * 60)
        print("🔬 IMPROVED MULTI-WAVELENGTH MODEL INFO")
        print("=" * 60)
        print(f"🏗️  层数: {self.num_layers}")
        print(f"🌈 波长数: {self.num_wavelengths}")
        print(f"📏 场大小: {self.field_size}×{self.field_size}")
        print(f"🎯 独立相位掩膜: 是")
        print(f"🔄 差分检测: {'是' if self.use_differential_detection else '否'}")
        print(f"📊 损失函数组成: 效率+分离+串扰+集中+平滑")
        
        # 显示每个波长的目标位置
        print(f"\n🎯 各波长目标位置:")
        for w_idx, wavelength in enumerate(self.wavelengths):
            offset_x, offset_y = self.config.offsets[w_idx]
            center_x = self.field_size // 2 + offset_x
            center_y = self.field_size // 2 + offset_y
            print(f"  {wavelength*1e9:.0f}nm: 中心({center_x}, {center_y}), 偏移({offset_x:+d}, {offset_y:+d})")
        
        print("=" * 60)
