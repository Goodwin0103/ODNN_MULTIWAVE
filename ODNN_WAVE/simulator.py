import torch
from save_function import save_to_mat_MC
from light_propagation_simulation_qz import plot_propagated_field, propagation_multi
import numpy as np
from label_utils import evaluate_output, evaluate_all_regions
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt

class FocusingLoss(nn.Module):
    """专门用于聚焦的损失函数"""
    
    def __init__(self, config, evaluation_regions, focus_weight=10.0, spread_penalty=1.0):
        super().__init__()
        self.config = config
        self.evaluation_regions = evaluation_regions
        self.focus_weight = focus_weight
        self.spread_penalty = spread_penalty
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        计算聚焦损失
        
        参数:
            predictions: 模型预测 [batch, wavelength, height, width]
            targets: 目标标签 [batch, wavelength, height, width]
        """
        # 基础MSE损失
        base_loss = self.mse_loss(predictions, targets)
        
        # 聚焦损失：鼓励能量集中在目标区域
        focus_loss = 0.0
        spread_loss = 0.0
        
        batch_size = predictions.shape[0]
        
        for b in range(batch_size):
            for wl_idx in range(predictions.shape[1]):
                pred_intensity = predictions[b, wl_idx]
                target_intensity = targets[b, wl_idx]
                
                # 计算目标区域的能量集中度
                target_center = self._find_energy_center(target_intensity)
                if target_center is not None:
                    # 聚焦损失：鼓励预测在目标中心附近有高能量
                    focus_mask = self._create_focus_mask(
                        pred_intensity.shape, target_center, radius=self.config.focus_radius
                    )
                    focus_energy = (pred_intensity * focus_mask).sum()
                    total_energy = pred_intensity.sum()
                    
                    if total_energy > 1e-8:
                        focus_ratio = focus_energy / total_energy
                        focus_loss += (1.0 - focus_ratio) ** 2
                    
                    # 散射惩罚：惩罚能量过度分散
                    spread_loss += self._calculate_spread_penalty(pred_intensity, target_center)
        
        total_loss = base_loss + self.focus_weight * focus_loss + self.spread_penalty * spread_loss
        
        return total_loss
    
    def _find_energy_center(self, intensity):
        """找到能量中心"""
        if intensity.sum() < 1e-8:
            return None
        
        # 计算质心
        h, w = intensity.shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=intensity.device),
            torch.arange(w, device=intensity.device),
            indexing='ij'
        )
        
        total_intensity = intensity.sum()
        center_y = (intensity * y_coords).sum() / total_intensity
        center_x = (intensity * x_coords).sum() / total_intensity
        
        return (center_y.item(), center_x.item())
    
    def _create_focus_mask(self, shape, center, radius):
        """创建聚焦掩膜"""
        h, w = shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h),
            torch.arange(w),
            indexing='ij'
        )
        
        center_y, center_x = center
        distance = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        mask = (distance <= radius).float()
        
        return mask.to(next(iter(predictions.parameters())).device if hasattr(predictions, 'parameters') else 'cpu')
    
    def _calculate_spread_penalty(self, intensity, center):
        """计算散射惩罚"""
        h, w = intensity.shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=intensity.device),
            torch.arange(w, device=intensity.device),
            indexing='ij'
        )
        
        center_y, center_x = center
        distance_sq = (y_coords - center_y)**2 + (x_coords - center_x)**2
        
        # 加权距离平方和作为散射惩罚
        spread = (intensity * distance_sq).sum()
        total_intensity = intensity.sum()
        
        if total_intensity > 1e-8:
            return spread / total_intensity
        else:
            return torch.tensor(0.0, device=intensity.device)

def create_focusing_initial_mask(layer_size, wavelength, focal_length, pixel_size):
    """创建初始聚焦相位掩膜"""
    center = layer_size // 2
    y, x = np.ogrid[:layer_size, :layer_size]
    
    # 计算到中心的距离
    r_squared = ((x - center) * pixel_size) ** 2 + ((y - center) * pixel_size) ** 2
    
    # 计算聚焦相位
    k = 2 * np.pi / wavelength
    phase = -k * r_squared / (2 * focal_length)
    
    # 限制相位范围
    phase = phase % (2 * np.pi)
    
    return phase

class Simulator:
    """仿真器类"""

    def __init__(self, config, evaluation_regions=None):
        self.config = config
        self.visibility_value = 0.0
        self.training_losses = []
        self.evaluation_regions = evaluation_regions
        self.propagated_field = None
        self.focusing_loss = FocusingLoss(config, evaluation_regions) if evaluation_regions else None

    def _preprocess_field_for_simulation(self, field: torch.Tensor) -> torch.Tensor:
        """预处理场以适应仿真"""
        padding_size = (self.config.layer_size - self.config.field_size) // 2
        padding = (padding_size, padding_size, padding_size, padding_size)
        return torch.nn.functional.pad(field, padding)

    def simulate_propagation(self, phase_masks, input_field, process_all_modes=False, mode_specific_masks=None):
        """
        模拟光场在多层衍射网络中的传播过程
        
        参数:
            phase_masks: 相位掩膜列表
            input_field: 输入光场，可以是单个模式或多个模式
            process_all_modes: 是否处理所有模式
            mode_specific_masks: 模式特定的相位掩膜，格式为[mode_idx][layer_idx][wavelength_idx]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 打印输入字段的维度信息
        print(f"输入字段维度: {input_field.dim()}D, 形状: {input_field.shape}")
        
        # 检查是否为多模式输入
        if process_all_modes:
            # 确定模式维度和数量
            if input_field.dim() == 5:  # [batch, mode, wavelength, height, width]
                num_modes = input_field.shape[1]
                print(f"检测到5D输入 [batch, mode, wavelength, height, width]，共{num_modes}个模式")
                
                for mode_idx in range(num_modes):
                    print(f"\n处理模式 {mode_idx+1}/{num_modes}")
                    # 提取单个模式的数据 - 直接转换为3D格式
                    single_mode = input_field[0, mode_idx]  # [wavelength, height, width]
                    
                    # 使用模式特定的相位掩膜（如果有）
                    if mode_specific_masks is not None and mode_idx < len(mode_specific_masks):
                        print(f"  使用模式 {mode_idx+1} 的专用相位掩膜")
                        current_masks = mode_specific_masks[mode_idx]
                    else:
                        print(f"  使用通用相位掩膜")
                        current_masks = phase_masks
                    
                    self._simulate_single_mode(current_masks, single_mode, mode_suffix=f"_mode{mode_idx+1}")
            
            elif input_field.dim() == 4:  # [mode, wavelength, height, width]
                num_modes = input_field.shape[0]
                print(f"检测到4D输入 [mode, wavelength, height, width]，共{num_modes}个模式")
                
                for mode_idx in range(num_modes):
                    print(f"\n处理模式 {mode_idx+1}/{num_modes}")
                    single_mode = input_field[mode_idx]  # [wavelength, height, width]
                    
                    # 使用模式特定的相位掩膜（如果有）
                    if mode_specific_masks is not None and mode_idx < len(mode_specific_masks):
                        print(f"  使用模式 {mode_idx+1} 的专用相位掩膜")
                        current_masks = mode_specific_masks[mode_idx]
                    else:
                        print(f"  使用通用相位掩膜")
                        current_masks = phase_masks
                    
                    self._simulate_single_mode(current_masks, single_mode, mode_suffix=f"_mode{mode_idx+1}")
            
            else:
                print("输入不是多模式格式，按单模式处理")
                self._simulate_single_mode(phase_masks, input_field)
        
        else:
            # 单模式处理
            self._simulate_single_mode(phase_masks, input_field)
            
    def evaluate_simulation_result(self, field, mode_idx=None, wavelength_idx=None):
        """
        评估模拟结果在所有区域的能量分布
        
        参数:
            field: 传播后的场
            mode_idx: 输入模式索引 (可选)
            wavelength_idx: 波长索引 (可选)
        
        返回:
            各区域能量列表和最大能量区域索引
        """
        if self.evaluation_regions is None:
            print("警告: 未设置评估区域")
            return None, None
        
        # 计算场强度
        intensity = np.abs(field)**2
        
        # 计算所有区域的能量
        all_energies = evaluate_all_regions(intensity, self.evaluation_regions)
        
        # 找到能量最大的区域
        max_energy_idx = np.argmax(all_energies)
        
        # 计算预期索引和实际索引
        if mode_idx is not None and wavelength_idx is not None:
            num_wavelengths = len(self.config.wavelengths)
            expected_idx = mode_idx * num_wavelengths + wavelength_idx
            predicted_mode = max_energy_idx // num_wavelengths
            predicted_wl = max_energy_idx % num_wavelengths
            
            print(f"模式{mode_idx}波长{wavelength_idx}: "
                  f"最大能量在区域{max_energy_idx} (模式{predicted_mode},波长{predicted_wl}), "
                  f"正确: {max_energy_idx == expected_idx}")
        
        return all_energies, max_energy_idx

    def _simulate_single_mode(self, phase_masks, input_field, mode_suffix=""):
        """
        模拟光场在多层衍射网络中的传播过程
        
        参数:
            phase_masks: 相位掩膜列表
            input_field: 输入光场
            mode_suffix: 模式后缀，用于区分不同模式的保存结果
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 处理输入字段的维度
        input_dim = input_field.dim()
        print(f"输入字段维度: {input_dim}D, 形状: {input_field.shape}")
        
        # 处理不同维度的输入
        if input_dim == 2:  # [height, width]
            # 添加波长维度
            input_field = input_field.unsqueeze(0)  # [1, height, width]
        elif input_dim == 3:  # [wavelength, height, width]
            # 已经是正确的格式
            pass
        elif input_dim == 4:  # 可能是 [batch, wavelength, height, width] 或 [mode, wavelength, height, width]
            # 假设第一维是批次或模式，我们只取第一个
            input_field = input_field[0]  # [wavelength, height, width]
        elif input_dim == 5:  # [batch, mode, wavelength, height, width]
            # 取第一个批次和第一个模式
            input_field = input_field[0, 0]  # [wavelength, height, width]
        else:
            raise ValueError(f"输入张量维度错误: 支持 2D/3D/4D/5D, 但输入为 {input_dim}D")
        
        # 现在 input_field 应该是 [wavelength, height, width] 格式
        input_field_padded = self._preprocess_field_for_simulation(input_field)
        propagated = []

        for lam_idx, lam in enumerate(self.config.wavelengths):
            print(f'  λ = {lam*1e9:.0f} nm')
            Ei = input_field_padded[lam_idx:lam_idx+1].to(device=device, dtype=torch.complex64)
            Ei = Ei.unsqueeze(0)  # 添加批次维度，变为 [1, 1, height, width]
            layer_propagated = [Ei.clone()]

            for l, raw_mask_full in enumerate(phase_masks):
                raw_mask = raw_mask_full[lam_idx]
                
                # 修复：确保mask是二维张量并转换为PyTorch张量
                if isinstance(raw_mask, np.ndarray):
                    # 如果是NumPy数组，确保是二维的
                    if raw_mask.ndim == 1:
                        print(f"警告: 层 {l} 波长 {lam_idx} 的掩膜是一维的，尝试重塑")
                        # 尝试将一维掩膜重塑为二维方形掩膜
                        size = int(np.sqrt(raw_mask.size))
                        if size * size == raw_mask.size:
                            raw_mask = raw_mask.reshape(size, size)
                        else:
                            print(f"错误: 无法将掩膜重塑为方形，跳过此掩膜")
                            continue
                    mask = torch.from_numpy(raw_mask).to(Ei.device)
                else:
                    # 如果已经是PyTorch张量，确保是二维的
                    if raw_mask.dim() == 1:
                        print(f"警告: 层 {l} 波长 {lam_idx} 的掩膜是一维的，尝试重塑")
                        size = int(torch.sqrt(torch.tensor(raw_mask.numel())))
                        if size * size == raw_mask.numel():
                            mask = raw_mask.reshape(size, size).to(Ei.device)
                        else:
                            print(f"错误: 无法将掩膜重塑为方形，跳过此掩膜")
                            continue
                    else:
                        mask = raw_mask.to(Ei.device)
                
                # 添加批次和通道维度
                mask_expanded = mask.unsqueeze(0).unsqueeze(0)
                
                # 确保掩膜与输入形状匹配
                if mask_expanded.shape[-2:] != Ei.shape[-2:]:
                    print(f"警告: 掩膜形状 {mask_expanded.shape[-2:]} 与输入形状 {Ei.shape[-2:]} 不匹配，尝试调整")
                    # 可以添加调整代码，如插值或裁剪
                    # 此处简单跳过不匹配的掩膜
                    continue
                
                # 应用相位掩膜
                Ei = Ei * torch.exp(1j * mask_expanded)

                # 绘制传播过程
                plot_propagated_field(
                    Ei.squeeze(0),
                    z_start=0,
                    z_end=self.config.z_layers if l < len(phase_masks)-1 else self.config.z_prop,
                    z_step=self.config.z_step,
                    dx=self.config.pixel_size,
                    lam=float(lam)
                )

                if l < len(phase_masks) - 1:
                    Ei = propagation_multi(
                        Ei, z=self.config.z_layers,
                        wavelengths=[lam],
                        pixel_size=self.config.pixel_size,
                        device=Ei.device
                    )
                    layer_propagated.append(Ei.clone())

            Ef = propagation_multi(
                Ei, z=self.config.z_prop,
                wavelengths=[lam],
                pixel_size=self.config.pixel_size,
                device=Ei.device
            )
            layer_propagated.append(Ef)
            propagated.append(layer_propagated)
            print('  → 结束\n')

            # 评估聚焦质量
            self._evaluate_focusing_quality(Ef, lam_idx, mode_suffix)

            if self.config.flag_savemat:
                try:
                    save_to_mat_MC(
                        save_dir=self.config.save_dir,
                        mode_classification=f"MC_single_{lam*1e9:.0f}nm{mode_suffix}",  # 添加模式后缀
                        num_modes=self.config.num_modes,
                        test_dataset="TestSim",
                        visibility_value=self.visibility_value,
                        temp_model=[m[lam_idx] for m in phase_masks],
                        temp_E=input_field[lam_idx],
                        propagated_fields=layer_propagated,
                        distance_layers=self.config.z_layers,
                        pixel_size=self.config.pixel_size,
                        distance_propagation=self.config.z_prop,
                        training_loss=self.training_losses,
                        field_size=self.config.field_size,
                        focus_radius=self.config.focus_radius,
                        detectsize=self.config.detectsize,
                        wavelength=lam,
                        epochs=self.config.epochs
                    )
                except Exception as e:
                    print(f"保存.mat文件时出错: {e}")
                    print("跳过.mat文件保存，继续执行...")
        
        # 保存最终传播场用于评估
        self.propagated_field = Ef
        
        # 在光场传播完成后评估结果
        if self.evaluation_regions is not None:
            mode_idx = None
            
            # 尝试从模式后缀提取模式索引
            if mode_suffix and mode_suffix.startswith("_mode"):
                try:
                    mode_idx = int(mode_suffix.replace("_mode", "")) - 1
                except:
                    mode_idx = None
            
            print("\n评估传播结果:")
            for lam_idx, _ in enumerate(self.config.wavelengths):
                if Ef.shape[1] > lam_idx:  # 确保有足够的通道
                    # 获取当前波长的场
                    propagated_amp = np.abs(Ef.squeeze(0)[lam_idx].cpu().numpy())
                    
                    # 评估场在所有区域的能量分布
                    _, max_region = self.evaluate_simulation_result(
                        propagated_amp,
                        mode_idx=mode_idx,
                        wavelength_idx=lam_idx
                    )

    def _evaluate_focusing_quality(self, field, wavelength_idx, mode_suffix):
        """评估聚焦质量"""
        # 转换为numpy数组
        if isinstance(field, torch.Tensor):
            intensity = np.abs(field.squeeze().cpu().numpy())**2
        else:
            intensity = np.abs(field)**2
        
        # 如果是多波长，选择当前波长
        if intensity.ndim == 3:
            intensity = intensity[0]  # 取第一个波长
        
        # 计算聚焦指标
        total_energy = intensity.sum()
        if total_energy > 1e-8:
            # 计算质心
            h, w = intensity.shape
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            center_y = (intensity * y_coords).sum() / total_energy
            center_x = (intensity * x_coords).sum() / total_energy
            
            # 计算聚焦半径内的能量比例
            center = (int(center_y), int(center_x))
            focus_radius_pixels = int(self.config.focus_radius / self.config.pixel_size)
            
            # 创建圆形掩膜
            distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            focus_mask = distances <= focus_radius_pixels
            
            focus_energy = (intensity * focus_mask).sum()
            focus_ratio = focus_energy / total_energy
            
            # 计算峰值强度位置
            max_pos = np.unravel_index(np.argmax(intensity), intensity.shape)
            
            print(f"  聚焦质量{mode_suffix}:")
            print(f"    质心位置: ({center_y:.1f}, {center_x:.1f})")
            print(f"    峰值位置: {max_pos}")
            print(f"    聚焦比例: {focus_ratio:.4f}")
            print(f"    峰值强度: {intensity.max():.6f}")
            
            return focus_ratio
        else:
            print(f"  警告: 场能量过低{mode_suffix}")
            return 0.0

    def generate_mode_specific_masks(self, base_masks, num_modes):
        """
        为每个模式生成专用相位掩膜
        
        参数:
            base_masks: 基础相位掩膜列表
            num_modes: 模式数量
        
        返回:
            mode_specific_masks: 每个模式的特定掩膜
        """
        mode_specific_masks = []
        
        for mode_idx in range(num_modes):
            print(f"为模式 {mode_idx+1} 生成专用掩膜...")
            # 每个模式的掩膜基于基本掩膜，但添加聚焦优化
            mode_masks = []
            for layer_idx, layer_masks in enumerate(base_masks):
                # 复制层掩膜
                layer_copy = []
                for wl_idx, mask in enumerate(layer_masks):
                    # 创建聚焦增强掩膜
                    wavelength = self.config.wavelengths[wl_idx]
                    focal_length = self.config.z_layers * (layer_idx + 1) + self.config.z_prop
                    
                    # 生成聚焦相位掩膜
                    focusing_mask = create_focusing_initial_mask(
                        mask.shape[0], wavelength, focal_length, self.config.pixel_size
                    )
                    
                    # 结合原始掩膜和聚焦掩膜
                    if isinstance(mask, np.ndarray):
                        enhanced_mask = mask + 0.3 * focusing_mask
                    else:
                        enhanced_mask = mask.cpu().numpy() + 0.3 * focusing_mask
                    
                    # 添加模式特定的小扰动
                    mode_perturbation = np.random.normal(0, 0.05, enhanced_mask.shape)
                    enhanced_mask = enhanced_mask + mode_perturbation
                    
                    # 确保相位范围在[0, 2π]内
                    enhanced_mask = enhanced_mask % (2 * np.pi)
                    
                    layer_copy.append(enhanced_mask)
                mode_masks.append(layer_copy)
            mode_specific_masks.append(mode_masks)
            
        return mode_specific_masks

    def _check_and_adjust_masks(self, masks, mode_idx=None):
        """
        检查并调整掩膜结构，确保格式正确
        
        参数:
            masks: 相位掩膜
            mode_idx: 模式索引（用于调试信息）
        
        返回:
            调整后的掩膜
        """
        mode_info = f"模式 {mode_idx+1}" if mode_idx is not None else "通用"
        print(f"检查{mode_info}掩膜结构...")
        
        if not isinstance(masks, list):
            print(f"  {mode_info}掩膜不是列表，转换为列表")
            return [[masks]]
        
        if len(masks) == 0:
            print(f"  {mode_info}掩膜列表为空")
            return masks
        
        # 检查第一层
        if not isinstance(masks[0], list):
            print(f"  {mode_info}掩膜第一层不是列表，转换为[层数][波长数]格式")
            return [[m] for m in masks]
        
        # 检查是否有额外的模式维度
        if len(masks[0]) > 0 and isinstance(masks[0][0], list):
            print(f"  {mode_info}掩膜存在额外的模式维度，调整结构")
            adjusted_masks = []
            for layer_idx, layer_masks in enumerate(masks):
                if isinstance(layer_masks[0], list):
                    # 只取第一个模式的掩膜
                    adjusted_masks.append(layer_masks[0])
                else:
                    adjusted_masks.append(layer_masks)
            return adjusted_masks
        
        return masks

    def diagnose_focusing_issues(self, phase_masks):
        """诊断聚焦问题"""
        print("=== 聚焦问题诊断 ===")
        
        # 1. 检查相位掩膜的数值范围
        for layer_idx, layer_masks in enumerate(phase_masks):
            for wl_idx, mask in enumerate(layer_masks):
                if isinstance(mask, np.ndarray):
                    phase_range = mask.max() - mask.min()
                    phase_std = mask.std()
                    print(f"层{layer_idx+1} 波长{wl_idx+1}: 相位范围={phase_range:.3f}, 标准差={phase_std:.3f}")
        
        # 2. 检查传播距离是否合理
        total_distance = len(phase_masks) * self.config.z_layers + self.config.z_prop
        rayleigh_length = np.pi * (self.config.focus_radius * self.config.pixel_size)**2 / self.config.wavelengths[0]
        print(f"总传播距离: {total_distance*1e6:.1f} μm")
        print(f"瑞利长度: {rayleigh_length*1e6:.1f} μm")
        print(f"距离比: {total_distance/rayleigh_length:.2f}")
        
        # 3. 检查像素尺寸和采样
        fresnel_number = (self.config.layer_size * self.config.pixel_size)**2 / (4 * self.config.wavelengths[0] * total_distance)
        print(f"菲涅尔数: {fresnel_number:.3f}")
        
        if fresnel_number < 1:
            print("⚠️  菲涅尔数过小，可能需要增加层尺寸或减少传播距离")
        else:
            print("✅  菲涅尔数合理，采样足够")
