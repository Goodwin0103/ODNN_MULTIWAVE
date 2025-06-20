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
                    chan = predictions[:, c]
                    energies = []
                    # 使用所有评估区域
                    for region_idx, region in enumerate(self.evaluation_regions):
                        xs, xe, ys, ye = region
                        region_sum = chan[:, ys:ye, xs:xe].sum(dim=(-2, -1))
                        energies.append(region_sum)
                    energies = torch.stack(energies, dim=1)
                    weights_batch.append(energies)
                
                # 重新排列维度: [波长, 批次, 评估区域]
                weights_batch = torch.stack(weights_batch, dim=0)
                all_weights_pred.append(weights_batch.cpu())
        
        # 合并批次维度
        weights_pred = torch.cat(all_weights_pred, dim=1).numpy()
        
        # 计算可见度 - 考虑多模式多波长
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
