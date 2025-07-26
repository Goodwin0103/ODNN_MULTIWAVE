import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from label_utils import create_evaluation_regions_mode_wavelength, evaluate_output, evaluate_all_regions

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(self, config, data_generator, model_class, evaluation_regions=None):
        self.config = config
        self.data_generator = data_generator
        self.model_class = model_class
        
        # 使用提供的评估区域或创建新的区域
        if evaluation_regions is not None:
            self.evaluation_regions = evaluation_regions
            print(f"使用外部提供的评估区域: {len(evaluation_regions)}个区域")
        else:
            # 使用新的创建方法
            self.evaluation_regions = create_evaluation_regions_mode_wavelength(
                self.config.layer_size,
                self.config.layer_size,
                self.config.focus_radius,
                detectsize=self.config.detectsize
            )
            print(f"创建评估区域: {len(self.evaluation_regions)}个区域")

    def train_model(self, num_layers):
        train_loader = self.data_generator.create_dataloader()
        model = self.model_class(self.config, num_layers).to(device)
        losses = self._train_loop(model, train_loader)
        evaluation_results = self._evaluate_model(model, train_loader)
        return {
            'models': model,
            'losses': losses,
            'phase_masks': self._extract_phase_masks(model),
            'weights_pred': evaluation_results['weights_pred'],
            'visibility': evaluation_results['visibility']
        }

    def _train_loop(self, model, train_loader):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay)
        losses = []
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0
            for images, labels in train_loader:
                images = images.to(device, dtype=torch.complex64)
                labels = labels.to(device)  # 保持标签原样
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # 动态适应标签通道数
                label_channels = labels.shape[1]
                loss = criterion(outputs[:, :label_channels], labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{self.config.epochs}], Loss: {avg_loss:.18f}')
        
        return losses

    def _extract_phase_masks(self, model):
        phase_masks = []
        for layer in model.layers:
            # 获取单个相位掩膜
            phase = layer.phase.detach().cpu().numpy()
            phase = phase % (2 * np.pi)
            
            wavelength_masks = []
            for _ in range(len(self.config.wavelengths)):
                wavelength_masks.append(phase)
            phase_masks.append(wavelength_masks)
        return phase_masks

    def _evaluate_model(self, model, test_loader):
        model.eval()
        all_weights_pred = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, dtype=torch.complex64)
                predictions = model(images)
                B, C, H, W = predictions.shape
                
                # 确保C等于波长数量
                if C != len(self.config.wavelengths):
                    print(f"警告: 预测通道数({C})与波长数量({len(self.config.wavelengths)})不匹配")
                
                # 处理每个波长的预测
                weights_batch = []
                for c in range(min(C, len(self.config.wavelengths))):
                    chan = predictions[:, c]  # 形状: [批次, H, W]
                    energies = []
                    
                    # 使用掩码数组计算区域能量
                    for region_idx, region_mask in enumerate(self.evaluation_regions):
                        # 确保region_mask是tensor且在正确设备上
                        if isinstance(region_mask, np.ndarray):
                            region_mask = torch.from_numpy(region_mask).to(device)
                        elif isinstance(region_mask, torch.Tensor):
                            region_mask = region_mask.to(device)
                        
                        # 计算区域内的能量总和
                        # chan: [批次, H, W], region_mask: [H, W]
                        region_energy = (chan * region_mask.float()).sum(dim=(-2, -1))
                        energies.append(region_energy)
                    
                    energies = torch.stack(energies, dim=1)  # [批次, 区域数]
                    weights_batch.append(energies)
                
                # 重新排列维度: [波长, 批次, 区域数]
                weights_batch = torch.stack(weights_batch, dim=0)
                all_weights_pred.append(weights_batch.cpu())
        
        # 合并批次维度
        weights_pred = torch.cat(all_weights_pred, dim=1).numpy()
        
        # 计算可见度
        visibility = self._calculate_visibility(weights_pred)
        
        return {'weights_pred': weights_pred, 'visibility': visibility}


    def _calculate_visibility(self, weights):
        """返回每个模式的可见度"""
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()
        
        num_wavelengths, num_batches, num_regions = weights.shape
        num_modes = self.config.num_modes
        mode_visibilities = []
        
        # 假设区域按模式-波长组合排列
        for mode_idx in range(num_modes):
            # 计算该模式在所有批次中的平均可见度
            mode_vis_across_batches = []
            
            for batch_idx in range(num_batches):
                # 计算该模式在所有区域的能量分布
                mode_energies = []
                
                # 找出该模式对应的区域索引 (对所有波长)
                for wl_idx in range(num_wavelengths):
                    # 检查对应当前模式和波长的所有区域能量
                    region_indices = []
                    for r in range(num_modes * num_wavelengths):
                        r_mode = r // num_wavelengths
                        r_wl = r % num_wavelengths
                        if r_mode == mode_idx:
                            region_indices.append(r)
                    
                    # 收集区域能量
                    for region_idx in region_indices:
                        if region_idx < num_regions:
                            mode_energies.append(weights[wl_idx, batch_idx, region_idx])
                
                # 计算该批次该模式的可见度
                if mode_energies:
                    I_max = np.max(mode_energies)
                    I_min = np.min(mode_energies)
                    
                    if I_max + I_min > 1e-12:
                        visibility = (I_max - I_min) / (I_max + I_min)
                    else:
                        visibility = 0.0
                        
                    mode_vis_across_batches.append(visibility)
            
            # 计算该模式所有批次的平均可见度
            if mode_vis_across_batches:
                mode_visibilities.append(np.mean(mode_vis_across_batches))
            else:
                mode_visibilities.append(0.0)
        
        return mode_visibilities

    def train_multiple_models(self, num_layer_options):
        results = {'models': [], 'losses': [], 'phase_masks': [], 'weights_pred': [], 'visibility': []}
        
        # 初始化按模式组织的可见度列表
        visibility_by_mode = []
        
        for num_layers in num_layer_options:
            model_result = self.train_model(num_layers)
            
            # 收集每个层数下的模型结果
            for k in results:
                if k != 'visibility':
                    results[k].append(model_result[k])
            
            # 处理可见度数据 - 假设model_result['visibility']是一个包含每个模式可见度的列表
            mode_visibilities = model_result['visibility']
            results['visibility'].append(mode_visibilities)
            
        return results


# 在 trainer.py 文件中添加以下类

class ImprovedMultiWavelengthTrainer:
    """改进的多波长训练器"""
    
    def __init__(self, model, train_loader, criterion, optimizer, device, 
                 evaluation_regions, wavelengths, num_modes):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.evaluation_regions = evaluation_regions
        self.wavelengths = wavelengths
        self.num_modes = num_modes
        
        # 训练历史
        self.losses = []
        self.detailed_losses = {'mse': [], 'separation': [], 'focus': [], 'total': []}
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )
        
        # 早停机制
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = 100
        
        print(f"📚 改进训练器初始化完成")
        print(f"  - 波长数: {len(wavelengths)}")
        print(f"  - 模式数: {num_modes}")
        print(f"  - 评估区域数: {len(evaluation_regions)}")
    
    def train(self, epochs):
        """改进的训练循环"""
        print(f"🚀 开始训练 {epochs} 轮...")
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_losses = {'mse': 0, 'separation': 0, 'focus': 0, 'total': 0}
            batch_count = 0
            
            for batch_input, batch_target in self.train_loader:
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                output = self.model(batch_input)
                
                # 计算损失
                if hasattr(self.criterion, '__call__') and len(self.criterion(output, batch_target)) == 2:
                    loss, loss_components = self.criterion(output, batch_target)
                    
                    # 记录详细损失
                    for key, value in loss_components.items():
                        epoch_losses[key] += value
                else:
                    loss = self.criterion(output, batch_target)
                    epoch_losses['total'] += loss.item()
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                batch_count += 1
            
            # 计算平均损失
            for key in epoch_losses:
                avg_loss = epoch_losses[key] / batch_count
                epoch_losses[key] = avg_loss
                self.detailed_losses[key].append(avg_loss)
            
            avg_total_loss = epoch_losses['total']
            self.losses.append(avg_total_loss)
            
            # 学习率调度
            self.scheduler.step(avg_total_loss)
            
            # 早停检查
            if avg_total_loss < self.best_loss:
                self.best_loss = avg_total_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.early_stop_patience:
                print(f"⏹️ 早停触发在第 {epoch} 轮")
                break
            
            # 打印进度
            if epoch % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch:3d}/{epochs}] - '
                      f'总损失: {avg_total_loss:.6f}, '
                      f'MSE: {epoch_losses.get("mse", 0):.6f}, '
                      f'分离: {epoch_losses.get("separation", 0):.6f}, '
                      f'聚焦: {epoch_losses.get("focus", 0):.6f}, '
                      f'学习率: {current_lr:.2e}')
        
        print(f"✅ 训练完成! 最佳损失: {self.best_loss:.6f}")
        return self.losses
    
    def evaluate_energy_distribution(self):
        """评估能量分布"""
        print("📊 评估能量分布...")
        
        self.model.eval()
        with torch.no_grad():
            # 获取样本数据
            sample_input, _ = next(iter(self.train_loader))
            sample_input = sample_input.to(self.device)
            
            # 前向传播
            output = self.model(sample_input)
            
            # 计算区域能量
            weights = torch.zeros(len(self.wavelengths), self.num_modes, len(self.evaluation_regions))
            
            for wl_idx in range(len(self.wavelengths)):
                for mode_idx in range(self.num_modes):
                    intensity = torch.abs(output[:, wl_idx, mode_idx])**2
                    
                    for region_idx, region_mask in enumerate(self.evaluation_regions):
                        region_mask = torch.tensor(region_mask, device=self.device, dtype=torch.float32)
                        region_energy = torch.sum(intensity * region_mask, dim=(1, 2))
                        weights[wl_idx, mode_idx, region_idx] = torch.mean(region_energy)
            
            return weights.cpu().numpy()
