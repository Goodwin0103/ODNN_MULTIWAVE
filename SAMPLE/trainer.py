import torch
import numpy as np
import time

class SimpleTrainer:
    def __init__(self, config, data_generator):
        self.config = config
        self.data_generator = data_generator
        self.device = config.device
        
    def train_model(self, model, num_epochs=None):
        """训练单个模型"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        # 确保模型在正确设备上
        model = model.to(self.device)
        model.train()
        
        # 初始化优化器，使用更合适的学习率
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 记录训练损失
        losses = []
        
        # 初始化最佳损失和最佳模型状态
        best_loss = float('inf')
        best_model_state = None
        
        # 计时
        start_time = time.time()
        
        # 获取训练数据 - 确保在正确设备上
        input_fields = self.data_generator.generate_input_fields()
        if isinstance(input_fields, list):
            input_fields = [field.to(self.device) for field in input_fields]
        else:
            input_fields = input_fields.to(self.device)
        
        print(f"Training on device: {self.device}")
        print(f"Model device: {next(model.parameters()).device}")
        if isinstance(input_fields, list):
            print(f"Input fields devices: {[field.device for field in input_fields]}")
        else:
            print(f"Input fields device: {input_fields.device}")
        
        # 训练循环
        for epoch in range(num_epochs):
            # 前向传播
            optimizer.zero_grad()
            
            # 确保输入数据在正确设备上
            if isinstance(input_fields, list):
                input_fields_device = [field.to(self.device) for field in input_fields]
            else:
                input_fields_device = input_fields.to(self.device)
            
            output_fields = model(input_fields_device)
            
            # 计算损失
            loss = model.compute_loss(output_fields)
            
            # 反向传播和优化
            loss.backward()
            
            # 打印梯度信息（调试用）
            if epoch == 0 or epoch % 100 == 99:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            if grad_norm < 1e-8:
                                print(f"Warning: {name} has very small gradient: {grad_norm}")
            
            optimizer.step()
            
            # 记录损失
            current_loss = loss.item()
            losses.append(current_loss)
            
            # 更新最佳模型
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            # 打印训练进度
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {current_loss:.6f}")
        
        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds.")

        # 计算最终可见度
        visibility = self.calculate_visibility(model, input_fields_device)
        print(f"Model with {model.num_layers} layers - Final loss: {best_loss:.6f}, Visibility: {visibility:.4f}")

        return {
            'model': model,
            'losses': losses,
            'visibility': visibility,
            'training_time': training_time
        }

    def train_multiple_models(self, num_layer_options):
        """训练多个不同层数的模型"""
        from model import SimpleMultiWavelengthModel
        
        # 存储结果
        results = {
            'models': [],
            'losses': [],
            'phase_masks': [],
            'energy_normalized': [],
            'visibility': []
        }
        
        # 生成输入数据 - 确保在正确设备上
        input_fields = self.data_generator.generate_input_data()
        if isinstance(input_fields, list):
            input_fields = [field.to(self.device) for field in input_fields]
        else:
            input_fields = input_fields.to(self.device)
        
        # 训练每个模型
        for num_layers in num_layer_options:
            print(f"\n训练 {num_layers} 层模型...")
            
            # 创建模型
            model = SimpleMultiWavelengthModel(
                config=self.config,
                num_layers=num_layers
            ).to(self.device)
            
            # 训练模型
            train_result = self.train_model(model)
            model = train_result['model']
            loss_history = train_result['losses']
            
            # 计算最终输出场
            with torch.no_grad():
                if isinstance(input_fields, list):
                    input_fields_device = [field.to(self.device) for field in input_fields]
                else:
                    input_fields_device = input_fields.to(self.device)
                
                output_fields = model(input_fields_device)
            
            # 计算归一化能量分布
            energy_normalized = []
            for w_idx, field in enumerate(output_fields):
                energy = torch.abs(field)**2
                energy_norm = energy / torch.max(energy)
                energy_normalized.append(energy_norm)
            
            # 计算可见度
            visibility = self.calculate_visibility(model, input_fields_device)
            
            # 存储结果
            results['models'].append(model)
            results['losses'].append(loss_history)
            results['phase_masks'].append(model.phase_masks)
            results['energy_normalized'].append(energy_normalized)
            results['visibility'].append(visibility)
            
            print(f"Model with {num_layers} layers - Final loss: {loss_history[-1]:.6f}, Visibility: {visibility:.4f}")
            
        return results
    
    def calculate_visibility(self, model, input_fields):
        """计算模型的可见度"""
        # 确保模型在评估模式
        model.eval()
        
        with torch.no_grad():
            # 确保输入在正确设备上
            if isinstance(input_fields, list):
                input_fields_device = [field.to(self.device) for field in input_fields]
            else:
                input_fields_device = input_fields.to(self.device)
            
            # 获取输出场
            output_fields = model(input_fields_device)
        
        # 计算每个波长在对应检测区域的平均强度
        intensities = []
        
        for w_idx, field in enumerate(output_fields):
            # 计算强度
            intensity = torch.abs(field)**2
            
            # 获取检测区域
            offset_x, offset_y = self.config.offsets[w_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            half_size = self.config.detectsize // 2  # 修改这里
            x_start = center_x - half_size
            x_end = center_x + half_size
            y_start = center_y - half_size
            y_end = center_y + half_size
            
            # 计算检测区域内的平均强度
            detect_intensity = torch.mean(intensity[y_start:y_end, x_start:x_end])
            intensities.append(detect_intensity.item())
        
        # 计算可见度
        if len(intensities) > 1:
            I_max = max(intensities)
            I_min = min(intensities)
            visibility = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) > 0 else 0
        else:
            visibility = 0.0
            
        # 恢复训练模式
        model.train()
        
        return visibility


    def calculate_wavelength_separation(self, model):
        """
        计算波长分离性能指标
        
        Args:
            model: 训练好的模型
            
        Returns:
            separation_metrics: 包含各种分离指标的字典
        """
        # 确保模型在评估模式
        model.eval()
        
        # 获取输入场
        input_fields = self.data_generator.generate_input_fields()
        if isinstance(input_fields, list):
            input_fields = [field.to(self.device) for field in input_fields]
        else:
            input_fields = input_fields.to(self.device)
        
        # 获取输出场
        with torch.no_grad():
            output_fields = model(input_fields)
        
        separation_metrics = {}
        
        for w_idx, wavelength in enumerate(self.config.wavelengths):
            field = output_fields[w_idx]
            intensity = torch.abs(field)**2
            
            # 计算总能量
            total_energy = torch.sum(intensity).item()
            
            # 获取目标检测区域
            offset_x, offset_y = self.config.offsets[w_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            half_size = self.config.detect_size // 2
            x_start = max(0, center_x - half_size)
            x_end = min(self.config.field_size, center_x + half_size)
            y_start = max(0, center_y - half_size)
            y_end = min(self.config.field_size, center_y + half_size)
            
            # 计算目标区域内的能量
            target_energy = torch.sum(intensity[y_start:y_end, x_start:x_end]).item()
            
            # 计算效率（目标区域能量占总能量的比例）
            efficiency = target_energy / total_energy if total_energy > 0 else 0
            
            # 计算串扰（其他波长目标区域的能量）
            crosstalk_energies = []
            for other_w_idx in range(len(self.config.wavelengths)):
                if other_w_idx != w_idx:
                    other_offset_x, other_offset_y = self.config.offsets[other_w_idx]
                    other_center_x = self.config.field_size // 2 + other_offset_x
                    other_center_y = self.config.field_size // 2 + other_offset_y
                    
                    other_x_start = max(0, other_center_x - half_size)
                    other_x_end = min(self.config.field_size, other_center_x + half_size)
                    other_y_start = max(0, other_center_y - half_size)
                    other_y_end = min(self.config.field_size, other_center_y + half_size)
                    
                    other_energy = torch.sum(intensity[other_y_start:other_y_end, other_x_start:other_x_end]).item()
                    crosstalk_energies.append(other_energy / total_energy if total_energy > 0 else 0)
            
            # 计算平均串扰
            avg_crosstalk = np.mean(crosstalk_energies) if crosstalk_energies else 0
            
            # 计算信噪比（目标区域能量与串扰的比值）
            snr = efficiency / (avg_crosstalk + 1e-10)
            
            # 计算对比度
            max_intensity = torch.max(intensity).item()
            target_avg_intensity = target_energy / ((x_end - x_start) * (y_end - y_start))
            
            # 计算背景强度（除目标区域外的平均强度）
            background_mask = torch.ones_like(intensity, dtype=torch.bool)
            background_mask[y_start:y_end, x_start:x_end] = False
            background_intensity = torch.mean(intensity[background_mask]).item() if torch.any(background_mask) else 0
            
            contrast = (target_avg_intensity - background_intensity) / (target_avg_intensity + background_intensity + 1e-10)
            
            # 存储指标
            separation_metrics[f"wavelength_{wavelength*1e9:.0f}nm"] = {
                'efficiency': efficiency,
                'avg_crosstalk': avg_crosstalk,
                'snr': snr,
                'contrast': contrast,
                'target_energy': target_energy,
                'total_energy': total_energy
            }
        
        # 计算整体分离性能
        all_efficiencies = [metrics['efficiency'] for metrics in separation_metrics.values()]
        all_crosstalks = [metrics['avg_crosstalk'] for metrics in separation_metrics.values()]
        
        separation_metrics['overall'] = {
            'avg_efficiency': np.mean(all_efficiencies),
            'avg_crosstalk': np.mean(all_crosstalks),
            'efficiency_std': np.std(all_efficiencies),
            'crosstalk_std': np.std(all_crosstalks),
            'separation_ratio': np.mean(all_efficiencies) / (np.mean(all_crosstalks) + 1e-10)
        }
        
        # 恢复训练模式
        model.train()
        
        return separation_metrics
