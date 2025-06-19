import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(self, config, data_generator, model_class):
        self.config = config
        self.data_generator = data_generator
        self.model_class = model_class
        self.evaluation_regions = self._create_evaluation_regions()

    def _create_evaluation_regions(self):
        center_x = self.config.layer_size // 2
        center_y = self.config.layer_size // 2
        
        regions = []
        for offset_x, offset_y in self.config.offsets:
            region = (
                center_x - self.config.detectsize // 2 + offset_x,
                center_x + self.config.detectsize // 2 + offset_x,
                center_y - self.config.detectsize // 2 + offset_y,
                center_y + self.config.detectsize // 2 + offset_y
            )
            regions.append(region)
        
        return regions

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
                
                # [修改点12] 确保C等于波长数量
                if C != len(self.config.wavelengths):
                    print(f"警告: 预测通道数({C})与波长数量({len(self.config.wavelengths)})不匹配")
                
                # 处理每个波长的预测
                weights_batch = []
                for c in range(min(C, len(self.config.wavelengths))):
                    chan = predictions[:, c]
                    energies = []
                    for (xs, xe, ys, ye) in self.evaluation_regions:
                        region_sum = chan[:, ys:ye, xs:xe].sum(dim=(-2, -1))
                        energies.append(region_sum)
                    energies = torch.stack(energies, dim=1)
                    weights_batch.append(energies)
                
                # [修改点13] 重新排列维度: [波长, 批次, 探测器]
                weights_batch = torch.stack(weights_batch, dim=0)
                all_weights_pred.append(weights_batch.cpu())
        
        # [修改点14] 合并批次维度
        weights_pred = torch.cat(all_weights_pred, dim=1).numpy()
        
        # [修改点15] 计算可见度 - 考虑所有波长
        visibility = self._calculate_visibility(weights_pred)
        
        return {'weights_pred': weights_pred, 'visibility': visibility}


    def _calculate_visibility(self, weights):
        """返回每个模式的可见度"""
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()
        
        N, C, N_det = weights.shape
        mode_visibilities = []
        
        for n in range(N):
            mode_vis = []
            for c in range(C):
                intensities = weights[n, c, :]
                I_max = np.max(intensities)
                I_min = np.min(intensities)
                
                if I_max + I_min > 1e-12:
                    visibility = (I_max - I_min) / (I_max + I_min)
                else:
                    visibility = 0.0
                
                mode_vis.append(visibility)
            mode_visibilities.append(np.mean(mode_vis))  # 每个模式的平均可见度
        
        return mode_visibilities  # 返回数组而不是单一值



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
            
            # 初始化按模式组织的列表
            if len(visibility_by_mode) == 0:
                visibility_by_mode = [[] for _ in range(len(mode_visibilities))]
            
            # 按模式组织可见度数据
            for mode_idx, mode_vis in enumerate(mode_visibilities):
                visibility_by_mode[mode_idx].append(mode_vis)
        
        # 将按模式组织的可见度数据添加到结果中
        results['visibility'] = visibility_by_mode
        
        return results

    def set_detector_regions(self, regions):
        """
        设置检测区域
        
        参数:
            regions: 检测区域列表，每个区域为(x_start, x_end, y_start, y_end)的元组
        """
        self.detector_regions = regions
        print(f"已设置{len(regions)}个检测区域")
        
        # 如果有评估器，也更新评估器的检测区域
        if hasattr(self, 'evaluator'):
            self.evaluator.detector_regions = regions
            print("已同步更新评估器的检测区域")
        
        return True

    def train_multi_mode_multi_wavelength_model(self, model, train_loader):
        """训练多模式多波长模型"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay)
        losses = []
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0
            
            for images, labels in train_loader:
                # 将数据移动到设备
                images = images.to(device, dtype=torch.complex64)
                labels = labels.to(device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(images)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 更新学习率
            scheduler.step()
            
            # 记录平均损失
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{self.config.epochs}], Loss: {avg_loss:.6f}')
        
        return losses

    def evaluate_multi_mode_multi_wavelength_model(self, model, test_loader):
        """评估多模式多波长模型"""
        model.eval()
        all_weights_pred = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device, dtype=torch.complex64)
                predictions = model(images)
                
                # 提取每个模式和波长的预测权重
                B, M, W, H, W = predictions.shape
                
                # 对每个模式和波长计算检测区域的能量
                for m in range(M):
                    for w in range(W):
                        weights = []
                        
                        # 计算每个检测区域的能量
                        for region in self.evaluation_regions:
                            x_start, x_end, y_start, y_end = region
                            region_energy = torch.mean(predictions[:, m, w, y_start:y_end, x_start:x_end])
                            weights.append(region_energy.item())
                        
                        all_weights_pred.append(weights)
        
        # 重新组织预测权重为[模式数, 波长数, 检测区域数]的形状
        weights_pred = np.array(all_weights_pred).reshape(M, W, -1)
        
        # 计算可见度
        visibility = self._calculate_visibility(weights_pred)
        
        return {'weights_pred': weights_pred, 'visibility': visibility}

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import time

class Trainer:
    def __init__(self, config, data_generator):
        self.config = config
        self.data_generator = data_generator
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.evaluation_regions = self._create_evaluation_regions()
        print(f"使用设备: {self.device}")

    def _create_evaluation_regions(self):
        """创建评估区域"""
        regions = []
        
        # 为每个模式和波长创建评估区域
        for mode_idx in range(self.config.num_modes):
            mode_regions = []
            
            for wl_idx, (offset_x, offset_y) in enumerate(self.config.offsets):
                # 计算中心点
                center_x = self.config.layer_size // 2 + offset_x
                center_y = self.config.layer_size // 2 + offset_y
                
                # 计算区域边界
                half_size = self.config.detectsize // 2
                region = (
                    center_x - half_size,  # x_start
                    center_x + half_size,  # x_end
                    center_y - half_size,  # y_start
                    center_y + half_size   # y_end
                )
                
                mode_regions.append(region)
            
            regions.append(mode_regions)
        
        return regions

    def train_model(self, model, train_loader, epochs=None):
        """训练模型"""
        if epochs is None:
            epochs = self.config.epochs
            
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay)
        
        losses = []
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for images, labels in train_loader:
                # 将数据移动到设备
                images = images.to(self.device, dtype=torch.complex64)
                labels = labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(images)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 更新学习率
            scheduler.step()
            
            # 记录平均损失
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            # 打印进度
            if epoch % 100 == 0 or epoch == epochs - 1:
                elapsed = time.time() - start_time
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Time: {elapsed:.2f}s')
                start_time = time.time()
        
        # 保存训练损失
        self.data_generator.training_losses = losses
        
        return losses

    def evaluate_model(self, model, test_loader):
        """评估模型"""
        model.eval()
        all_weights_pred = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device, dtype=torch.complex64)
                predictions = model(images)
                
                # 提取每个模式和波长的预测权重
                B, C, H, W = predictions.shape
                
                # 对每个模式和波长计算检测区域的能量
                for b in range(B):  # 批次(模式)
                    mode_weights = []
                    
                    for c in range(C):  # 通道(波长)
                        # 获取当前模式和波长的预测
                        pred_bc = predictions[b, c]
                        
                        # 获取对应的评估区域
                        region = self.evaluation_regions[b][c]
                        x_start, x_end, y_start, y_end = region
                        
                        # 计算区域内的平均能量
                        region_energy = torch.mean(pred_bc[y_start:y_end, x_start:x_end])
                        mode_weights.append(region_energy.item())
                    
                    all_weights_pred.append(mode_weights)
        
        # 重新组织预测权重为[模式数, 波长数]的形状
        weights_pred = np.array(all_weights_pred).reshape(self.config.num_modes, len(self.config.wavelengths))
        
        # 计算可见度
        visibility = self._calculate_visibility(weights_pred)
        
        # 保存可见度
        self.data_generator.visibility_value = np.mean(visibility)
        
        return {
            'weights_pred': weights_pred,
            'visibility': visibility
        }

    def _calculate_visibility(self, weights):
        """计算可见度"""
        # 输入形状: [模式数, 波长数]
        # 输出: 每个模式的可见度列表
        
        num_modes, num_wavelengths = weights.shape
        visibility = []
        
        for m in range(num_modes):
            # 获取当前模式的所有波长权重
            mode_weights = weights[m]
            
            # 计算最大值和最小值
            max_val = np.max(mode_weights)
            min_val = np.min(mode_weights)
            
            # 计算可见度
            if max_val + min_val > 0:
                vis = (max_val - min_val) / (max_val + min_val)
            else:
                vis = 0
            
            visibility.append(vis)
        
        return visibility

    def train_multiple_models(self, model_class, num_layer_options):
        """训练多个不同层数的模型"""
        results = {
            'models': [],
            'losses': [],
            'phase_masks': [],
            'weights_pred': [],
            'visibility': []
        }
        
        for num_layers in num_layer_options:
            print(f"\n训练 {num_layers} 层的模型...\n")
            
            # 创建模型
            model = model_class(self.config, num_layers)
            
            # 创建数据加载器
            train_loader = self.data_generator.create_dataloader()
            
            # 训练模型
            losses = self.train_model(model, train_loader)
            
            # 评估模型
            evaluation_results = self.evaluate_model(model, train_loader)
            
            # 提取相位掩膜
            phase_masks = model.get_all_phase_masks()
            
            # 保存结果
            results['models'].append(model)
            results['losses'].append(losses)
            results['phase_masks'].append(phase_masks)
            results['weights_pred'].append(evaluation_results['weights_pred'])
            results['visibility'].append(evaluation_results['visibility'])
            
            print(f"模型训练完成，可见度: {np.mean(evaluation_results['visibility']):.4f}")
        
        return results
