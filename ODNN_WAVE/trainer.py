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
            # 🔧 使用新的按波长分列的方法
            self.evaluation_regions = create_evaluation_regions_by_wavelength(
                self.config.layer_size,
                self.config.layer_size,
                self.config.focus_radius,
                detectsize=self.config.detectsize,
                offsets=getattr(self.config, 'offsets', None)  # 安全获取偏移参数
            )
            print(f"创建按波长分列的评估区域: {len(self.evaluation_regions)}个区域")
    
    # ... 其余代码保持不变 ...


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
        
        # 计算可见度 - 修复为每个波长每个模式的可见度
        visibility = self._calculate_visibility_fixed(weights_pred)
        
        return {'weights_pred': weights_pred, 'visibility': visibility}

    def _calculate_visibility_fixed(self, weights):
        """
        修复版本：正确映射区域到模式和波长
        """
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()
        
        # weights shape: [num_wavelengths, num_batches, num_regions]
        num_wavelengths, num_batches, num_regions = weights.shape
        num_modes = self.config.num_modes
        
        print(f"计算可见度: {num_wavelengths}波长, {num_batches}批次, {num_regions}区域, {num_modes}模式")
        
        # 验证区域数量
        expected_regions = num_wavelengths * num_modes
        if num_regions != expected_regions:
            print(f"⚠️ 区域数量不匹配: 期望{expected_regions}, 实际{num_regions}")
        
        all_visibilities = []
        
        # 🔧 关键修复：按照区域创建的顺序来计算可见度
        # 区域创建顺序：wl0_mode0, wl0_mode1, wl0_mode2, wl1_mode0, wl1_mode1, wl1_mode2
        for wl_idx in range(num_wavelengths):
            wavelength = self.config.wavelengths[wl_idx]
            print(f"处理波长 {wavelength*1e9:.0f}nm (索引{wl_idx})")
            
            for mode_idx in range(num_modes):
                print(f"  处理模式 {mode_idx+1}")
                
                # 🔧 正确计算区域索引
                # 区域索引 = wl_idx * num_modes + mode_idx
                region_idx = wl_idx * num_modes + mode_idx
                
                if region_idx < num_regions:
                    # 计算该区域在所有批次中的可见度
                    mode_vis_across_batches = []
                    
                    for batch_idx in range(num_batches):
                        # 获取该波长该区域的能量
                        energy = weights[wl_idx, batch_idx, region_idx]
                        
                        # 简单的可见度计算（可以根据需要改进）
                        # 这里假设单个检测器的情况，实际应用中可能需要多个检测器
                        visibility = min(1.0, max(0.0, float(energy)))
                        mode_vis_across_batches.append(visibility)
                    
                    avg_visibility = np.mean(mode_vis_across_batches) if mode_vis_across_batches else 0.0
                    all_visibilities.append(avg_visibility)
                    
                    print(f"    区域{region_idx}: 平均可见度={avg_visibility:.6f}")
                else:
                    all_visibilities.append(0.0)
                    print(f"    区域{region_idx}: 超出范围，可见度=0.0")
        
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

    def evaluate_model_with_cross_matrix(self, model, test_inputs, layer_count):
        """
        使用交叉矩阵和SNR评估模型
        """
        model.eval()
        with torch.no_grad():
            # 获取相位掩膜
            phase_masks = []
            for layer in model.layers:
                if hasattr(layer, 'phase_mask'):
                    phase_masks.append(layer.phase_mask.detach())
            
            # 创建模拟器
            simulator = Simulator(
                self.config.H, self.config.W, self.config.dx,
                self.config.wavelengths, self.config.propagation_distance,
                phase_masks, self.config.target_positions
            )
            
            # 运行模拟
            outputs = []
            for mode_idx in range(self.config.num_modes):
                mode_input = test_inputs[mode_idx:mode_idx+1]  # [1, num_wl, H, W]
                mode_output = simulator(mode_input)  # [1, num_wl, H, W]
                outputs.append(mode_output[0])  # [num_wl, H, W]
            
            outputs = torch.stack(outputs)  # [num_modes, num_wl, H, W]
            
            # 计算交叉矩阵和SNR
            cross_matrix, snr_matrix, focus_metrics = calculate_cross_matrix_and_snr(
                outputs, self.config.target_positions, radius=self.config.focus_radius
            )
            
            # 打印分析结果
            separation_quality, avg_snrs = print_cross_matrix_analysis(
                cross_matrix, snr_matrix, focus_metrics, self.config.wavelengths
            )
            
            return {
                'cross_matrix': cross_matrix,
                'snr_matrix': snr_matrix,
                'focus_metrics': focus_metrics,
                'separation_quality': separation_quality,
                'avg_snrs': avg_snrs,
                'outputs': outputs
            }

    def train_single_configuration(self, num_layers):
        """
        修改后的训练函数，使用新的评估方法
        """
        # ... 原有的训练代码 ...
        
        # 训练完成后的评估
        print(f"\n🔍 评估 {num_layers} 层模型...")
        evaluation_results = self.evaluate_model_with_cross_matrix(model, test_inputs, num_layers)
        
        # 保存结果
        results = {
            'num_layers': num_layers,
            'cross_matrix': evaluation_results['cross_matrix'],
            'snr_matrix': evaluation_results['snr_matrix'],
            'separation_quality': evaluation_results['separation_quality'],
            'avg_snrs': evaluation_results['avg_snrs'],
            'focus_metrics': evaluation_results['focus_metrics']
        }
        
        return model, results

