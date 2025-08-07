import torch
import time
import numpy as np

class ImprovedTrainer:
    def __init__(self, config, data_generator):
        self.config = config
        self.data_generator = data_generator
        self.device = config.device
        
    def train_model(self, model, num_epochs=None):
        """🔥 改进的训练方法 - 参考文档优化策略"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        model = model.to(self.device)
        model.train()
        
        # 🔥 改进1：使用AdamW优化器 + 学习率调度
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 🔥 改进2：余弦退火学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        # 🔥 改进3：学习率预热
        warmup_epochs = min(50, num_epochs // 10)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        
        # 训练历史记录
        training_history = {
            'total_loss': [],
            'efficiency_loss': [],
            'separation_loss': [],
            'crosstalk_loss': [],
            'concentration_loss': [],
            'smoothing_loss': [],
            'efficiencies': [],
            'learning_rates': []
        }
        
        best_loss = float('inf')
        best_model_state = None
        patience = 100  # 早停耐心值
        patience_counter = 0
        
        start_time = time.time()
        
        # 获取训练数据
        input_fields = self.data_generator.generate_input_fields()
        if isinstance(input_fields, list):
            input_fields = [field.to(self.device) for field in input_fields]
        else:
            input_fields = input_fields.to(self.device)
        
        print(f"🚀 开始改进训练，使用多目标损失函数...")
        print(f"📊 损失函数组成: 效率 + 分离 + 串扰 + 集中 + 平滑")
        print(f"🔧 优化器: AdamW + 余弦退火 + 预热")
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            if isinstance(input_fields, list):
                input_fields_device = [field.to(self.device) for field in input_fields]
            else:
                input_fields_device = input_fields.to(self.device)
            
            output_fields = model(input_fields_device)
            
            # 🔥 使用详细损失信息
            loss_dict = model.get_detailed_loss(output_fields)
            total_loss = loss_dict['total_loss']
            
            total_loss.backward()
            
            # 🔥 改进4：梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 🔥 改进5：学习率调度
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step()
            
            # 记录训练历史
            current_loss = total_loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            
            training_history['total_loss'].append(current_loss)
            training_history['efficiency_loss'].append(loss_dict['efficiency_loss'].item())
            training_history['separation_loss'].append(loss_dict['separation_loss'].item())
            training_history['crosstalk_loss'].append(loss_dict['crosstalk_loss'].item())
            training_history['concentration_loss'].append(loss_dict['concentration_loss'].item())
            training_history['smoothing_loss'].append(loss_dict['smoothing_loss'].item())
            training_history['learning_rates'].append(current_lr)
            
            # 计算当前效率
            current_efficiencies = self._calculate_efficiencies(output_fields)
            training_history['efficiencies'].append(current_efficiencies)
            
            # 🔥 改进6：早停机制
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 打印训练进度
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1:3d}/{num_epochs}] | "
                      f"Total: {current_loss:.6f} | "
                      f"Eff: {loss_dict['efficiency_loss'].item():.4f} | "
                      f"Sep: {loss_dict['separation_loss'].item():.4f} | "
                      f"Cross: {loss_dict['crosstalk_loss'].item():.4f} | "
                      f"Conc: {loss_dict['concentration_loss'].item():.4f} | "
                      f"Smooth: {loss_dict['smoothing_loss'].item():.4f} | "
                      f"LR: {current_lr:.2e}")
                
                # 显示各波长效率
                for wl_idx, eff in enumerate(current_efficiencies):
                    wl_nm = self.config.wavelengths[wl_idx] * 1e9
                    print(f"    {wl_nm:.0f}nm: {eff:.4f} ({eff*100:.1f}%)")
                print()
            
            # 早停检查
            if patience_counter >= patience:
                print(f"🛑 早停触发! 在第 {epoch+1} 轮停止训练")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        final_efficiencies = self._calculate_efficiencies(model(input_fields_device))
        
        print(f"✅ 改进训练完成! 用时: {training_time:.2f}s")
        print(f"📊 最终效率:")
        avg_efficiency = 0
        for wl_idx, eff in enumerate(final_efficiencies):
            wl_nm = self.config.wavelengths[wl_idx] * 1e9
            print(f"    {wl_nm:.0f}nm: {eff:.4f} ({eff*100:.1f}%)")
            avg_efficiency += eff
        
        avg_efficiency /= len(final_efficiencies)
        print(f"📈 平均效率: {avg_efficiency:.4f} ({avg_efficiency*100:.1f}%)")
        
        return {
            'model': model,
            'training_history': training_history,
            'final_efficiencies': final_efficiencies,
            'training_time': training_time,
            'best_loss': best_loss,
            'avg_efficiency': avg_efficiency
        }
    
    def _calculate_efficiencies(self, output_fields):
        """计算各波长的效率"""
        efficiencies = []
        
        for w_idx, field in enumerate(output_fields):
            intensity = torch.abs(field)**2
            total_energy = torch.sum(intensity)
            
            # 获取目标区域
            offset_x, offset_y = self.config.offsets[w_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            half_size = self.config.detect_size // 2
            x_start = max(0, center_x - half_size)
            x_end = min(self.config.field_size, center_x + half_size)
            y_start = max(0, center_y - half_size)
            y_end = min(self.config.field_size, center_y + half_size)
            
            target_energy = torch.sum(intensity[y_start:y_end, x_start:x_end])
            efficiency = (target_energy / (total_energy + 1e-10)).item()
            efficiencies.append(efficiency)
        
        return efficiencies
    
    def calculate_separation_metrics(self, model, input_fields):
        """🔥 新增：计算详细的分离性能指标"""
        model.eval()
        with torch.no_grad():
            if isinstance(input_fields, list):
                input_fields_device = [field.to(self.device) for field in input_fields]
            else:
                input_fields_device = input_fields.to(self.device)
            
            output_fields = model(input_fields_device)
        
        metrics = {}
        total_efficiency = 0
        total_crosstalk = 0
        
        for w_idx, wavelength in enumerate(self.config.wavelengths):
            field = output_fields[w_idx]
            intensity = torch.abs(field)**2
            total_power = torch.sum(intensity).item()
            
            # 获取目标区域
            offset_x, offset_y = self.config.offsets[w_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            half_size = self.config.detect_size // 2
            x_start = max(0, center_x - half_size)
            x_end = min(self.config.field_size, center_x + half_size)
            y_start = max(0, center_y - half_size)
            y_end = min(self.config.field_size, center_y + half_size)
            
            # 目标区域功率
            target_power = torch.sum(intensity[y_start:y_end, x_start:x_end]).item()
            efficiency = target_power / (total_power + 1e-10)
            
            # 计算串扰
            crosstalk_powers = []
            for other_w_idx in range(len(self.config.wavelengths)):
                if other_w_idx != w_idx:
                    other_offset_x, other_offset_y = self.config.offsets[other_w_idx]
                    other_center_x = self.config.field_size // 2 + other_offset_x
                    other_center_y = self.config.field_size // 2 + other_offset_y
                    
                    other_x_start = max(0, other_center_x - half_size)
                    other_x_end = min(self.config.field_size, other_center_x + half_size)
                    other_y_start = max(0, other_center_y - half_size)
                    other_y_end = min(self.config.field_size, other_center_y + half_size)
                    
                    crosstalk_power = torch.sum(intensity[other_y_start:other_y_end, other_x_start:other_x_end]).item()
                    crosstalk_powers.append(crosstalk_power / (total_power + 1e-10))
            
            avg_crosstalk = np.mean(crosstalk_powers) if crosstalk_powers else 0
            
            # 计算信噪比和对比度
            background_power = total_power - target_power - sum([cp * total_power for cp in crosstalk_powers])
            snr = target_power / (background_power / (self.config.field_size**2 - self.config.detect_size**2) + 1e-10)
            contrast = (target_power - avg_crosstalk * total_power) / (target_power + avg_crosstalk * total_power + 1e-10)
            
            wavelength_nm = int(wavelength * 1e9)
            metrics[f"wavelength_{wavelength_nm}nm"] = {
                'efficiency': efficiency,
                'avg_crosstalk': avg_crosstalk,
                'snr': snr,
                'contrast': contrast,
                'extinction_ratio': efficiency / (avg_crosstalk + 1e-10)
            }
            
            total_efficiency += efficiency
            total_crosstalk += avg_crosstalk
        
        # 整体指标
        metrics['overall'] = {
            'avg_efficiency': total_efficiency / len(self.config.wavelengths),
            'avg_crosstalk': total_crosstalk / len(self.config.wavelengths),
            'separation_ratio': (total_efficiency / len(self.config.wavelengths)) / (total_crosstalk / len(self.config.wavelengths) + 1e-10)
        }
        
        return metrics
            
