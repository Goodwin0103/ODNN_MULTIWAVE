import torch
import numpy as np
import time

class SimpleTrainer:
    def __init__(self, config, data_generator):
        self.config = config
        self.data_generator = data_generator
        
    # 修改 train_model 方法
    def train_model(self, model, num_epochs=None):
        """训练单个模型"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        # 初始化优化器，使用更合适的学习率
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 记录训练损失
        losses = []
        
        # 初始化最佳损失和最佳模型状态
        best_loss = float('inf')
        best_model_state = None
        
        # 计时
        start_time = time.time()
        
        # 获取训练数据
        input_fields = self.data_generator.generate_input_fields()
        
        # 训练循环
        for epoch in range(num_epochs):
            # 前向传播
            optimizer.zero_grad()
            output_fields = model(input_fields)
            
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
        
        # 计算训练时间
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds.")
        
        # 计算最终可见度
        visibility = self.calculate_visibility(model)
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
            'visibility': []  # 添加可见度列表
        }
        
        # 生成输入数据
        input_fields = self.data_generator.generate_input_data()
        
        # 训练每个模型
        for num_layers in num_layer_options:
            # 创建模型 - 修改为使用config对象
            model = SimpleMultiWavelengthModel(
                config=self.config,
                num_layers=num_layers
            )
            
            # 训练模型
            model, loss_history = self.train_model(model)
            
            # 计算最终输出场
            output_fields = model(input_fields)
            
            # 计算归一化能量分布
            energy_normalized = []
            for w_idx, field in enumerate(output_fields):
                energy = torch.abs(field)**2
                energy_norm = energy / torch.max(energy)
                energy_normalized.append(energy_norm)
            
            # 计算可见度
            visibility = self.calculate_visibility(model, input_fields)
            
            # 存储结果
            results['models'].append(model)
            results['losses'].append(loss_history)
            results['phase_masks'].append(model.phase_masks)
            results['energy_normalized'].append(energy_normalized)
            results['visibility'].append(visibility)  # 存储可见度
            
            print(f"Model with {num_layers} layers - Final loss: {loss_history[-1]:.6f}, Visibility: {visibility:.4f}")
            
        return results
    
    def calculate_visibility(self, model, input_fields):
        """计算模型的可见度"""
        # 获取输出场
        output_fields = model(input_fields)
        
        # 计算每个波长在对应检测区域的平均强度
        intensities = []
        
        for w_idx, field in enumerate(output_fields):
            # 计算强度
            intensity = torch.abs(field)**2
            
            # 获取检测区域
            offset_x, offset_y = self.config.offsets[w_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            half_size = self.config.detectsize // 2
            x_start = center_x - half_size
            x_end = center_x + half_size
            y_start = center_y - half_size
            y_end = center_y + half_size
            
            # 计算检测区域内的平均强度
            detect_intensity = torch.mean(intensity[y_start:y_end, x_start:x_end])
            intensities.append(detect_intensity.item())
        
        # 计算可见度
        # 可见度定义为: (I_max - I_min) / (I_max + I_min)
        if len(intensities) > 1:
            I_max = max(intensities)
            I_min = min(intensities)
            visibility = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) > 0 else 0
        else:
            # 如果只有一个波长，可见度设为0
            visibility = 0.0
            
        return visibility
