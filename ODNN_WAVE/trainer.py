# trainer.py - 修改导入和相关代码

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import os
from simulator import Simulator
# 修改导入语句
from label_utils import (
    create_evaluation_regions_by_wavelength,  # 新函数
    create_evaluation_regions_mode_wavelength,  # 兼容性函数
    evaluate_output, 
    evaluate_all_regions
)

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
            # 🔧 使用新的按波长分列的方法，并传入模式数量
            self.evaluation_regions = create_evaluation_regions_by_wavelength(
                self.config.layer_size,
                self.config.layer_size,
                self.config.focus_radius,
                detectsize=getattr(self.config, 'detectsize', 20),  # 🔧 安全获取detectsize
                 offsets=getattr(self.config, 'offsets', [(0,0) for _ in range(len(self.config.wavelengths))]),
                num_modes=self.config.num_modes  # 🔧 添加模式数量参数
            )
            print(f"创建按波长分列的评估区域: {len(self.evaluation_regions)}个区域")

    def train_model(self, num_layers):
        train_loader = self.data_generator.create_dataloader()
        model = self.model_class(self.config, num_layers).to(device)
        losses = self._train_loop(model, train_loader)
        evaluation_results = self._evaluate_model(model, train_loader)
        
        # 保存训练结果
        self._save_training_results(model, losses, num_layers)
        
        return {
            'models': model,
            'losses': losses,
            'phase_masks': self._extract_phase_masks(model),
            'weights_pred': evaluation_results['weights_pred'],
            'visibility': evaluation_results['visibility']
        }

    def _save_training_results(self, model, losses, num_layers):
        """保存训练结果"""
        # 创建保存目录
        save_dir = os.path.join(self.config.save_dir, "trained_models")
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存完整模型
        model_path = os.path.join(save_dir, f"trained_model_{num_layers}layers.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_layers': num_layers,
                'model_class': self.model_class.__name__
            },
            'train_losses': losses,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }, model_path)
        print(f"✓ 完整模型已保存到: {model_path}")
        
        # 保存相位掩码（用于仿真）
        masks_path = os.path.join(save_dir, f"trained_phase_masks_{num_layers}layers.npz")
        model.save_trained_masks(masks_path)
        
        # 保存训练损失曲线
        loss_path = os.path.join(save_dir, f"training_losses_{num_layers}layers.npy")
        np.save(loss_path, losses)
        print(f"✓ 训练损失已保存到: {loss_path}")
        
        # 保存相位掩码可视化
        vis_dir = os.path.join(save_dir, f"phase_mask_visualization_{num_layers}layers")
        model.print_phase_masks(save_path=vis_dir)
        
        return model_path, masks_path

    def _train_loop(self, model, train_loader):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay)
        losses = []
        
        print(f"开始训练 - 设备: {device}")
        print(f"训练参数: epochs={self.config.epochs}, lr={self.config.learning_rate}")
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
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
        
        print("✓ 训练完成!")
        return losses

    def _extract_phase_masks(self, model):
        """提取相位掩码用于返回"""
        if hasattr(model, 'get_phase_masks_for_simulation'):
            return model.get_phase_masks_for_simulation()
        else:
            # 兼容旧版本
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
                    
                    # 🔧 修复：根据波长数量调整区域使用方式
                    if len(self.config.wavelengths) == 1:
                        # 单波长：直接使用所有区域（按模式顺序）
                        print(f"单波长评估: 使用前{self.config.num_modes}个区域")
                        for mode_idx in range(min(self.config.num_modes, len(self.evaluation_regions))):
                            region = self.evaluation_regions[mode_idx]
                            xs, xe, ys, ye = region
                            region_sum = chan[:, ys:ye, xs:xe].sum(dim=(-2, -1))
                            energies.append(region_sum)
                            
                    else:
                        # 多波长：按波长分组使用区域
                        start_idx = c * self.config.num_modes
                        end_idx = start_idx + self.config.num_modes
                        print(f"多波长评估: 波长{c+1}使用区域{start_idx}-{end_idx-1}")
                        
                        for region_idx in range(start_idx, min(end_idx, len(self.evaluation_regions))):
                            region = self.evaluation_regions[region_idx]
                            xs, xe, ys, ye = region
                            region_sum = chan[:, ys:ye, xs:xe].sum(dim=(-2, -1))
                            energies.append(region_sum)
                    
                    if energies:  # 🔧 确保energies不为空
                        energies = torch.stack(energies, dim=1)
                        weights_batch.append(energies)
                    else:
                        print(f"警告: 波长{c+1}没有找到有效的评估区域")
                        # 创建零张量作为占位符
                        zero_energies = torch.zeros(B, self.config.num_modes, device=chan.device)
                        weights_batch.append(zero_energies)
                
                # 重新排列维度: [波长, 批次, 评估区域]
                if weights_batch:
                    weights_batch = torch.stack(weights_batch, dim=0)
                    all_weights_pred.append(weights_batch.cpu())
        
        # 合并批次维度
        if all_weights_pred:
            weights_pred = torch.cat(all_weights_pred, dim=1).numpy()
            
            # 计算可见度 - 使用修复版本
            visibility = self._calculate_visibility_fixed(weights_pred)
            
            return {'weights_pred': weights_pred, 'visibility': visibility}
        else:
            print("警告: 没有生成有效的预测结果")
            return {'weights_pred': np.array([]), 'visibility': []}

    def _calculate_visibility_fixed(self, weights):
        """
        修复版本：正确映射区域到模式和波长
        """
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()
        
        # weights shape: [num_wavelengths, num_batches, num_regions_per_wavelength]
        if weights.size == 0:  # 🔧 处理空数组
            print("警告: 权重数组为空，返回空的可见度列表")
            return []
            
        num_wavelengths, num_batches, num_regions_per_wavelength = weights.shape
        num_modes = self.config.num_modes
        
        print(f"计算可见度: {num_wavelengths}波长, {num_batches}批次, 每波长{num_regions_per_wavelength}区域, {num_modes}模式")
        
        all_visibilities = []
        
        # 🔧 单波长和多波长的不同处理
        if num_wavelengths == 1:
            # 单波长情况：区域直接对应模式
            print("单波长可见度计算")
            for mode_idx in range(min(num_modes, num_regions_per_wavelength)):
                mode_vis_across_batches = []
                
                for batch_idx in range(num_batches):
                    # 获取该模式的能量
                    energy = weights[0, batch_idx, mode_idx]  # 单波长，索引为0
                    
                    # 🔧 改进的可见度计算
                    if isinstance(energy, (np.ndarray, torch.Tensor)):
                        energy = float(energy)
                    
                    # 确保能量值在合理范围内
                    visibility = max(0.0, min(1.0, float(energy)))
                    mode_vis_across_batches.append(visibility)
                
                avg_visibility = np.mean(mode_vis_across_batches) if mode_vis_across_batches else 0.0
                all_visibilities.append(avg_visibility)
                
                print(f"  模式{mode_idx+1}: 平均可见度={avg_visibility:.6f}")
        else:
            # 多波长情况：按原有逻辑处理
            print("多波长可见度计算")
            for wl_idx in range(num_wavelengths):
                wavelength = self.config.wavelengths[wl_idx]
                print(f"处理波长 {wavelength*1e9:.0f}nm (索引{wl_idx})")
                
                for mode_idx in range(min(num_modes, num_regions_per_wavelength)):
                    mode_vis_across_batches = []
                    
                    for batch_idx in range(num_batches):
                        # 获取该波长该模式的能量
                        energy = weights[wl_idx, batch_idx, mode_idx]
                        
                        # 🔧 改进的可见度计算
                        if isinstance(energy, (np.ndarray, torch.Tensor)):
                            energy = float(energy)
                        
                        visibility = max(0.0, min(1.0, float(energy)))
                        mode_vis_across_batches.append(visibility)
                    
                    avg_visibility = np.mean(mode_vis_across_batches) if mode_vis_across_batches else 0.0
                    all_visibilities.append(avg_visibility)
                    
                    print(f"    模式{mode_idx+1}: 平均可见度={avg_visibility:.6f}")
        
        print(f"✓ 计算完成，共{len(all_visibilities)}个可见度值")
        return all_visibilities

    def train_multiple_models(self, num_layer_options):
        results = {'models': [], 'losses': [], 'phase_masks': [], 'weights_pred': [], 'visibility': []}
        
        for num_layers in num_layer_options:
            print(f"\n{'='*50}")
            print(f"开始训练 {num_layers} 层模型...")
            print(f"{'='*50}")
            
            model_result = self.train_model(num_layers)
            
            # 收集每个层数下的模型结果
            for k in results:
                results[k].append(model_result[k])
            
            print(f"✓ {num_layers}层模型训练完成，可见度数量: {len(model_result['visibility'])}")
            
        return results

    @staticmethod
    def load_trained_model(model_path, model_class, config):
        """加载训练好的完整模型"""
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # 获取模型配置
            model_config = checkpoint.get('model_config', {})
            num_layers = model_config.get('num_layers', 3)
            
            # 创建模型实例
            model = model_class(config, num_layers).to(device)
            
            # 加载模型参数
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"✓ 成功加载训练好的模型: {model_path}")
            print(f"  模型类型: {model_config.get('model_class', 'Unknown')}")
            print(f"  层数: {num_layers}")
            
            return model, checkpoint.get('train_losses', [])
            
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            return None, None

class EnhancedTrainer(Trainer):
    """增强的训练器，支持 Zero Padding 模型"""
    
    def __init__(self, config, data_generator, model_class, evaluation_regions=None):
        # 调用父类构造函数
        super().__init__(config, data_generator, model_class, evaluation_regions)
        print("✓ 初始化增强训练器")

    def create_model_with_padding(self, num_layers):
        """创建支持Zero Padding的模型"""
        # 检查模型类是否支持padding参数
        try:
            model = self.model_class(
                self.config, 
                num_layers,
                use_zero_padding=True  # 启用Zero Padding
            ).to(device)
            print(f"✓ 创建了支持Zero Padding的{num_layers}层模型")
            return model
        except TypeError:
            # 如果模型不支持padding参数，使用标准创建方式
            print("⚠️ 模型不支持Zero Padding，使用标准模式")
            return self.model_class(self.config, num_layers).to(device)

    def train_model_with_checkpoints(self, num_layers, checkpoint_interval=500):
        """
        带检查点的训练方法，支持中断后继续训练
        """
        train_loader = self.data_generator.create_dataloader()
        model = self.create_model_with_padding(num_layers)
        
        # 检查是否存在检查点
        checkpoint_path = f"checkpoint_{num_layers}layers.pth"
        start_epoch = 0
        losses = []
        
        if os.path.exists(checkpoint_path):
            print(f"发现检查点文件: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            losses = checkpoint['losses']
            print(f"从第 {start_epoch} 轮继续训练")
        
        # 训练循环
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay)
        
        # 如果从检查点恢复，也恢复优化器状态
        if os.path.exists(checkpoint_path):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"开始训练 - 设备: {device}")
        print(f"训练参数: epochs={self.config.epochs}, lr={self.config.learning_rate}")
        
        for epoch in range(start_epoch, self.config.epochs):
            model.train()
            epoch_loss = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device, dtype=torch.complex64)
                labels = labels.to(device)
                
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
            
            # 定期保存检查点
            if epoch % checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses': losses,
                    'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
                }, checkpoint_path)
                print(f"检查点已保存: epoch {epoch}")
            
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{self.config.epochs}], Loss: {avg_loss:.18f}')
        
        # 删除检查点文件（训练完成）
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("训练完成，检查点文件已删除")
        
        print("✓ 训练完成!")
        return model, losses

    def evaluate_with_enhanced_metrics(self, model, test_loader):
        """增强的评估方法，包含更多指标"""
        model.eval()
        all_weights_pred = []
        all_outputs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, dtype=torch.complex64)
                predictions = model(images)
                all_outputs.append(predictions.cpu())
                
                B, C, H, W = predictions.shape
                
                # 处理每个波长的预测
                weights_batch = []
                for c in range(min(C, len(self.config.wavelengths))):
                    chan = predictions[:, c]
                    energies = []
                    
                    if len(self.config.wavelengths) == 1:
                        # 单波长：直接使用所有区域
                        for region_idx, region in enumerate(self.evaluation_regions):
                            xs, xe, ys, ye = region
                            region_sum = chan[:, ys:ye, xs:xe].sum(dim=(-2, -1))
                            energies.append(region_sum)
                    else:
                        # 多波长：按波长分组使用区域
                        start_idx = c * self.config.num_modes
                        end_idx = start_idx + self.config.num_modes
                        for region_idx in range(start_idx, min(end_idx, len(self.evaluation_regions))):
                            region = self.evaluation_regions[region_idx]
                            xs, xe, ys, ye = region
                            region_sum = chan[:, ys:ye, xs:xe].sum(dim=(-2, -1))
                            energies.append(region_sum)
                    
                    energies = torch.stack(energies, dim=1)
                    weights_batch.append(energies)
                
                weights_batch = torch.stack(weights_batch, dim=0)
                all_weights_pred.append(weights_batch.cpu())
        
        # 合并所有批次
        weights_pred = torch.cat(all_weights_pred, dim=1).numpy()
        all_outputs = torch.cat(all_outputs, dim=0)
        
        # 计算增强指标
        visibility = self._calculate_visibility_fixed(weights_pred)
        snr_metrics = self._calculate_snr_metrics(all_outputs)
        efficiency_metrics = self._calculate_efficiency_metrics(weights_pred)
        
        return {
            'weights_pred': weights_pred,
            'visibility': visibility,
            'snr_metrics': snr_metrics,
            'efficiency_metrics': efficiency_metrics,
            'outputs': all_outputs
        }

    def _calculate_snr_metrics(self, outputs):
        """计算信噪比指标"""
        snr_metrics = {}
        
        for wl_idx in range(outputs.shape[1]):  # 遍历波长
            wl_output = outputs[:, wl_idx]  # [batch, H, W]
            
            # 计算每个评估区域的SNR
            region_snrs = []
            for region_idx, region in enumerate(self.evaluation_regions):
                xs, xe, ys, ye = region
                
                # 信号：区域内的平均强度
                signal = wl_output[:, ys:ye, xs:xe].mean()
                
                # 噪声：区域外的标准差
                mask = torch.ones_like(wl_output[0], dtype=torch.bool)
                mask[ys:ye, xs:xe] = False
                noise_std = wl_output[:, mask].std()
                
                # SNR计算
                snr = 20 * torch.log10(signal / (noise_std + 1e-8))
                region_snrs.append(snr.item())
            
            snr_metrics[f'wavelength_{wl_idx+1}'] = region_snrs
        
        return snr_metrics

    def _calculate_efficiency_metrics(self, weights_pred):
        """计算效率指标"""
        efficiency_metrics = {}
        
        # 计算每个模式的聚焦效率
        for wl_idx in range(weights_pred.shape[0]):
            for mode_idx in range(weights_pred.shape[2]):
                # 目标模式的能量
                target_energy = weights_pred[wl_idx, :, mode_idx].mean()
                
                # 总能量
                total_energy = weights_pred[wl_idx, :, :].sum(axis=1).mean()
                
                # 效率 = 目标能量 / 总能量
                efficiency = target_energy / (total_energy + 1e-8)
                
                key = f'wl{wl_idx+1}_mode{mode_idx+1}_efficiency'
                efficiency_metrics[key] = efficiency
        
        return efficiency_metrics

