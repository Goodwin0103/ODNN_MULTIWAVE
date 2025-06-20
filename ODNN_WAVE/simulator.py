import torch
from save_function import save_to_mat_MC
from light_propagation_simulation_qz import plot_propagated_field, propagation_multi
import numpy as np
from label_utils import evaluate_output, evaluate_all_regions

class Simulator:
    """仿真器类"""

    def __init__(self, config, evaluation_regions=None):
        self.config = config
        self.visibility_value = 0.0
        self.training_losses = []
        self.evaluation_regions = evaluation_regions
        self.propagated_field = None

    def _preprocess_field_for_simulation(self, field: torch.Tensor) -> torch.Tensor:
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
            # 每个模式的掩膜基于基本掩膜，但可以添加小扰动
            mode_masks = []
            for layer_idx, layer_masks in enumerate(base_masks):
                # 复制层掩膜
                layer_copy = []
                for wl_idx, mask in enumerate(layer_masks):
                    # 添加小的随机扰动使得每个模式的掩膜略有不同
                    perturbed_mask = mask + np.random.normal(0, 0.05, mask.shape)
                    # 确保相位范围在[0, 2π]内
                    perturbed_mask = perturbed_mask % (2 * np.pi)
                    layer_copy.append(perturbed_mask)
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
