class Simulator:
    """完整的仿真器类"""

    def __init__(self, config, evaluation_regions=None):
        self.config = config
        self.visibility_value = 0.0
        self.training_losses = []
        self.evaluation_regions = evaluation_regions
        self.propagated_field = None
        self.focusing_loss = FocusingLoss(config, evaluation_regions) if evaluation_regions else None
        self.simulation_results = {}  # 存储仿真结果
        self.focusing_metrics = {}    # 存储聚焦指标

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
        
        # 初始化结果存储
        mode_key = f"mode{mode_suffix}" if mode_suffix else "single_mode"
        self.simulation_results[mode_key] = {}
        self.focusing_metrics[mode_key] = {}

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
            focus_quality = self._evaluate_focusing_quality(Ef, lam_idx, mode_suffix)
            
            # 存储结果
            wavelength_key = f"wavelength_{lam*1e9:.0f}nm"
            self.simulation_results[mode_key][wavelength_key] = {
                'final_field': Ef.clone(),
                'layer_propagated': layer_propagated,
                'focus_quality': focus_quality
            }

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
            
            # 计算半高全宽 (FWHM)
            max_intensity = intensity.max()
            half_max = max_intensity / 2
            fwhm_mask = intensity >= half_max
            fwhm_area = np.sum(fwhm_mask)
            fwhm_radius = np.sqrt(fwhm_area / np.pi) * self.config.pixel_size
            
            # 计算聚焦效率
            theoretical_diffraction_limit = 1.22 * self.config.wavelengths[wavelength_idx] / (2 * 0.5)  # 假设NA=0.5
            focusing_efficiency = theoretical_diffraction_limit / (fwhm_radius * 2) if fwhm_radius > 0 else 0
            
            print(f"  聚焦质量{mode_suffix}:")
            print(f"    质心位置: ({center_y:.1f}, {center_x:.1f})")
            print(f"    峰值位置: {max_pos}")
            print(f"    聚焦比例: {focus_ratio:.4f}")
            print(f"    峰值强度: {intensity.max():.6f}")
            print(f"    FWHM半径: {fwhm_radius*1e6:.2f} μm")
            print(f"    聚焦效率: {focusing_efficiency:.4f}")
            
            # 存储聚焦指标
            mode_key = f"mode{mode_suffix}" if mode_suffix else "single_mode"
            wavelength_key = f"wavelength_{self.config.wavelengths[wavelength_idx]*1e9:.0f}nm"
            
            if mode_key not in self.focusing_metrics:
                self.focusing_metrics[mode_key] = {}
            
            self.focusing_metrics[mode_key][wavelength_key] = {
                'focus_ratio': focus_ratio,
                'peak_intensity': intensity.max(),
                'centroid': (center_y, center_x),
                'peak_position': max_pos,
                'fwhm_radius': fwhm_radius,
                'focusing_efficiency': focusing_efficiency
            }
            
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

    def optimize_masks_for_focusing(self, initial_masks, target_positions=None, iterations=100):
        """
        使用梯度下降优化相位掩膜以改善聚焦性能
        
        参数:
            initial_masks: 初始相位掩膜
            target_positions: 目标聚焦位置列表
            iterations: 优化迭代次数
        
        返回:
            优化后的相位掩膜
        """
        print("开始相位掩膜聚焦优化...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将掩膜转换为可训练参数
        optimizable_masks = []
        for layer_masks in initial_masks:
            layer_params = []
            for mask in layer_masks:
                if isinstance(mask, np.ndarray):
                    mask_tensor = torch.from_numpy(mask).float().to(device)
                else:
                    mask_tensor = mask.float().to(device)
                mask_tensor.requires_grad_(True)
                layer_params.append(mask_tensor)
            optimizable_masks.append(layer_params)
        
        # 设置优化器
        all_params = []
        for layer_params in optimizable_masks:
            all_params.extend(layer_params)
        
        optimizer = torch.optim.Adam(all_params, lr=0.01)
        
        # 优化循环
        for iteration in range(iterations):
            optimizer.zero_grad()
            
            # 计算聚焦损失
            total_loss = 0.0
            
            for wl_idx, wavelength in enumerate(self.config.wavelengths):
                # 创建测试输入场
                test_field = torch.ones((1, 1, self.config.layer_size, self.config.layer_size), 
                                      dtype=torch.complex64, device=device)
                
                # 通过网络传播
                current_field = test_field
                for layer_idx, layer_params in enumerate(optimizable_masks):
                    # 应用相位掩膜
                    phase_mask = layer_params[wl_idx]
                    current_field = current_field * torch.exp(1j * phase_mask.unsqueeze(0).unsqueeze(0))
                    
                    # 传播到下一层
                    if layer_idx < len(optimizable_masks) - 1:
                        current_field = propagation_multi(
                            current_field, z=self.config.z_layers,
                            wavelengths=[wavelength],
                            pixel_size=self.config.pixel_size,
                            device=device
                        )
                
                # 最终传播
                final_field = propagation_multi(
                    current_field, z=self.config.z_prop,
                    wavelengths=[wavelength],
                    pixel_size=self.config.pixel_size,
                    device=device
                )
                
                # 计算聚焦损失
                intensity = torch.abs(final_field)**2
                
                # 计算质心
                h, w = intensity.shape[-2:]
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(h, device=device, dtype=torch.float32),
                    torch.arange(w, device=device, dtype=torch.float32),
                    indexing='ij'
                )
                
                total_intensity = intensity.sum()
                if total_intensity > 1e-8:
                    center_y = (intensity.squeeze() * y_coords).sum() / total_intensity
                    center_x = (intensity.squeeze() * x_coords).sum() / total_intensity
                    
                    # 目标位置（图像中心）
                    target_y = h // 2
                    target_x = w // 2
                    
                    # 位置损失
                    position_loss = (center_y - target_y)**2 + (center_x - target_x)**2
                    
                    # 聚焦损失（鼓励能量集中）
                    focus_radius = self.config.focus_radius / self.config.pixel_size
                    distances = torch.sqrt((y_coords - target_y)**2 + (x_coords - target_x)**2)
                    focus_mask = (distances <= focus_radius).float()
                    
                    focus_energy = (intensity.squeeze() * focus_mask).sum()
                    focus_loss = 1.0 - (focus_energy / total_intensity)
                    
                    total_loss += position_loss + 10.0 * focus_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            if iteration % 20 == 0:
                print(f"优化迭代 {iteration}/{iterations}, 损失: {total_loss.item():.6f}")
        
        # 转换回numpy格式
        optimized_masks = []
        for layer_params in optimizable_masks:
            layer_masks = []
            for mask_tensor in layer_params:
                layer_masks.append(mask_tensor.detach().cpu().numpy())
            optimized_masks.append(layer_masks)
        
        print("相位掩膜优化完成")
        return optimized_masks

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
        
        # 4. 建议
        print("\n=== 建议 ===")
        phase_range = 0
        for layer_masks in phase_masks:
            for mask in layer_masks:
                if isinstance(mask, np.ndarray):
                    phase_range = max(phase_range, mask.max() - mask.min())
        
        if phase_range < np.pi:
            print("- 相位范围较小，考虑增加相位调制深度")
        if total_distance > 10 * rayleigh_length:
            
