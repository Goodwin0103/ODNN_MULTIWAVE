import torch
from save_function import save_to_mat_MC
from light_propagation_simulation_qz import plot_propagated_field, propagation_multi

class Simulator:
    """仿真器类"""

    def __init__(self, config):
        self.config = config
        self.visibility_value = 0.0
        self.training_losses = []

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

        for lam_idx, lam in enumerate(self.config.wavelengths):
            print(f'  λ = {lam*1e9:.0f} nm')
            Ei = input_field_padded[lam_idx:lam_idx+1].to(device=device, dtype=torch.complex64)
            Ei = Ei.unsqueeze(0)  # 添加批次维度，变为 [1, 1, height, width]
            propagated = [Ei.clone()]

            for l, raw_mask_full in enumerate(phase_masks):
                raw_mask = raw_mask_full[lam_idx]
                mask = torch.from_numpy(raw_mask).to(Ei.device)
                Ei = Ei * torch.exp(1j * mask[None, None, :, :]) 

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
                    propagated.append(Ei.clone())

            Ef = propagation_multi(
                Ei, z=self.config.z_prop,
                wavelengths=[lam],
                pixel_size=self.config.pixel_size,
                device=Ei.device
            )
            propagated.append(Ef)
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
                        propagated_fields=propagated,
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

    def generate_mode_specific_masks(self, base_masks, num_modes):
        """
        为每个模式生成专用的相位掩膜
        
        参数:
            base_masks: 基础相位掩膜
            num_modes: 模式数量
        
        返回:
            mode_specific_masks: 每个模式的专用相位掩膜
        """
        import numpy as np
        
        mode_specific_masks = []
        
        for mode_idx in range(num_modes):
            # 复制基础掩膜
            mode_masks = []
            
            for layer_idx, layer_mask in enumerate(base_masks):
                # 复制每一层的掩膜
                wavelength_masks = []
                
                for wl_idx, wl_mask in enumerate(layer_mask):
                    # 确保 wl_mask 是 numpy 数组
                    if not isinstance(wl_mask, np.ndarray):
                        wl_mask = np.array(wl_mask)
                    
                    # 为不同模式稍微调整掩膜
                    if mode_idx == 0:  # 第一个模式使用原始掩膜
                        adjusted_mask = wl_mask.copy()
                    elif mode_idx == 1:  # 第二个模式增强相位对比度
                        # 增强相位对比度
                        adjusted_mask = wl_mask.copy()
                        # 找出掩膜的中值
                        median_val = np.median(adjusted_mask)
                        # 增强对比度：大于中值的值增大，小于中值的值减小
                        adjusted_mask = median_val + (adjusted_mask - median_val) * 1.2
                    else:  # 其他模式进行其他调整
                        # 可以根据需要添加其他模式的调整策略
                        adjusted_mask = wl_mask.copy()
                        # 例如，添加小的随机扰动
                        adjusted_mask += np.random.normal(0, 0.1, adjusted_mask.shape)
                    
                    wavelength_masks.append(adjusted_mask)
                
                mode_masks.append(wavelength_masks)
            
            mode_specific_masks.append(mode_masks)
            print(f"已为模式 {mode_idx+1} 生成专用相位掩膜")
        
        return mode_specific_masks
