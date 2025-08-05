import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
import os
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ComplexMSELoss(nn.Module):
    """
    🔧 复数MSE损失函数 - 解决 "mse_cuda" not implemented for 'ComplexFloat' 问题
    """
    def __init__(self):
        super(ComplexMSELoss, self).__init__()
    
    def forward(self, input_complex, target_complex):
        """
        计算复数张量的MSE损失
        input_complex, target_complex: 复数张量
        """
        if target_complex is None:
            # 自监督模式：保持能量守恒
            input_energy = torch.sum(torch.abs(input_complex)**2, dim=(-2, -1))
            target_energy = input_energy  # 能量应该保持不变
            return torch.mean((input_energy - target_energy)**2)
        
        # 分别计算实部和虚部的MSE
        real_loss = F.mse_loss(input_complex.real, target_complex.real)
        imag_loss = F.mse_loss(input_complex.imag, target_complex.imag)
        
        return real_loss + imag_loss

class ComplexPhysicsConstrainedLoss(nn.Module):
    """
    🔧 复数物理约束损失函数
    """
    def __init__(self, energy_weight=1.0, smoothness_weight=0.1):
        super(ComplexPhysicsConstrainedLoss, self).__init__()
        self.energy_weight = energy_weight
        self.smoothness_weight = smoothness_weight
        self.mse_loss = ComplexMSELoss()
    
    def forward(self, predictions, targets, input_energy=None):
        """
        计算物理约束损失
        """
        # 基础MSE损失
        mse_loss = self.mse_loss(predictions, targets)
        
        # 能量守恒约束
        pred_energy = torch.sum(torch.abs(predictions)**2, dim=(-2, -1))
        if input_energy is not None:
            energy_loss = F.mse_loss(pred_energy, input_energy)
        else:
            energy_loss = torch.tensor(0.0, device=predictions.device)
        
        # 平滑性约束（相位梯度）
        pred_phase = torch.angle(predictions)
        grad_x = torch.diff(pred_phase, dim=-1)
        grad_y = torch.diff(pred_phase, dim=-2)
        smoothness_loss = torch.mean(grad_x**2) + torch.mean(grad_y**2)
        
        total_loss = (mse_loss + 
                     self.energy_weight * energy_loss + 
                     self.smoothness_weight * smoothness_loss)
        
        return total_loss

class MultiModeMultiWavelengthTrainer:
    """
    🆕 多模式多波长衍射神经网络训练器
    修复版本 - 解决复数损失函数问题
    """
    
    def __init__(self, config, data_generator, model_class=None, evaluation_regions=None):
        self.config = config
        self.data_generator = data_generator
        self.model_class = model_class
        
        # 设置评估区域
        if evaluation_regions is not None:
            self.evaluation_regions = evaluation_regions
        else:
            self.evaluation_regions = self._create_evaluation_regions()
        
        # 多模式相关配置
        self.num_modes = getattr(config, 'num_modes', 1)
        self.num_wavelengths = len(config.wavelengths) if hasattr(config, 'wavelengths') else 1
        
        print(f"🚀 初始化多模式多波长训练器:")
        print(f"   模式数: {self.num_modes}")
        print(f"   波长数: {self.num_wavelengths}")
        print(f"   检测区域数: {len(self.evaluation_regions)}")
        print(f"   数据字段大小: {getattr(config, 'field_size', 'Unknown')}")
        print(f"   模型层大小: {getattr(config, 'layer_size', 'Unknown')}")
        
        # 初始化损失函数 - 使用支持复数的损失函数
        self.criterion = self._create_loss_function()

    def _create_loss_function(self):
        """创建支持复数的损失函数"""
        try:
            # 尝试使用物理约束损失
            return ComplexPhysicsConstrainedLoss(
                energy_weight=getattr(self.config, 'energy_weight', 1.0),
                smoothness_weight=getattr(self.config, 'smoothness_weight', 0.1)
            )
        except Exception as e:
            print(f"⚠️  创建物理约束损失失败: {e}，使用复数MSE损失")
            return ComplexMSELoss()

    def _create_evaluation_regions(self):
        """创建评估区域 - 处理嵌套的 3x3 网格结构"""
        print("🔧 检查并修复 config.offsets...")
        
        # 检查 offsets 是否存在
        if not hasattr(self.config, 'offsets'):
            print("❌ config 中没有 offsets 属性，创建默认值")
            self.config.offsets = [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20)]
            return self._create_regions_from_offsets()
        
        print(f"原始 offsets: {self.config.offsets}")
        print(f"offsets 类型: {type(self.config.offsets)}")
        
        # 检测并处理嵌套结构
        if isinstance(self.config.offsets, list) and len(self.config.offsets) > 0:
            first_element = self.config.offsets[0]
            
            # 如果第一个元素也是列表，说明是嵌套结构（如 3x3 网格）
            if isinstance(first_element, list):
                print("🔍 检测到嵌套的 offsets 结构，正在展平...")
                
                flattened_offsets = []
                for i, row in enumerate(self.config.offsets):
                    print(f"  处理第 {i} 行: {row}")
                    for j, item in enumerate(row):
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            try:
                                offset_x, offset_y = float(item[0]), float(item[1])
                                flattened_offsets.append((offset_x, offset_y))
                                print(f"    位置 ({i},{j}): ({offset_x}, {offset_y})")
                            except (ValueError, TypeError) as e:
                                print(f"    ❌ 解析位置 ({i},{j}) 出错: {e}，使用默认值 (0, 0)")
                                flattened_offsets.append((0.0, 0.0))
                        else:
                            print(f"    ⚠️  位置 ({i},{j}) 格式异常: {item}，使用默认值 (0, 0)")
                            flattened_offsets.append((0.0, 0.0))
                
                self.config.offsets = flattened_offsets
                print(f"✅ 展平后的 offsets ({len(flattened_offsets)} 个): {self.config.offsets}")
                
            else:
                # 如果不是嵌套结构，按原来的方式处理
                print("🔍 检测到平坦的 offsets 结构")
                fixed_offsets = []
                for i, item in enumerate(self.config.offsets):
                    try:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            fixed_offsets.append((float(item[0]), float(item[1])))
                        elif isinstance(item, (int, float)):
                            fixed_offsets.append((float(item), 0.0))
                        else:
                            print(f"    ⚠️  无法解析元素 {i}: {item}")
                            fixed_offsets.append((0.0, 0.0))
                    except Exception as e:
                        print(f"    ❌ 解析元素 {i} 出错: {e}")
                        fixed_offsets.append((0.0, 0.0))
                
                self.config.offsets = fixed_offsets
        
        return self._create_regions_from_offsets()

    def _create_regions_from_offsets(self):
        """根据 offsets 创建检测区域"""
        center_x = self.config.layer_size // 2
        center_y = self.config.layer_size // 2
        
        regions = []
        
        print(f"🏗️  创建评估区域:")
        print(f"   图像中心: ({center_x}, {center_y})")
        print(f"   检测区域大小: {self.config.detectsize}×{self.config.detectsize}")
        print(f"   总共 {len(self.config.offsets)} 个区域")
        
        for i, (offset_x, offset_y) in enumerate(self.config.offsets):
            # 计算区域边界
            x_start = center_x - self.config.detectsize // 2 + int(offset_x)
            x_end = center_x + self.config.detectsize // 2 + int(offset_x)
            y_start = center_y - self.config.detectsize // 2 + int(offset_y)
            y_end = center_y + self.config.detectsize // 2 + int(offset_y)
            
            # 确保区域在图像边界内
            x_start = max(0, min(x_start, self.config.layer_size))
            x_end = max(0, min(x_end, self.config.layer_size))
            y_start = max(0, min(y_start, self.config.layer_size))
            y_end = max(0, min(y_end, self.config.layer_size))
            
            region = (x_start, x_end, y_start, y_end)
            regions.append(region)
            
            # 计算在 3x3 网格中的位置（如果是9个区域）
            if len(self.config.offsets) == 9:
                grid_row = i // 3
                grid_col = i % 3
                print(f"   区域 {i} (网格位置 [{grid_row},{grid_col}]): 偏移({offset_x:+.0f},{offset_y:+.0f}) -> 边界({x_start},{x_end},{y_start},{y_end})")
            else:
                print(f"   区域 {i}: 偏移({offset_x:+.0f},{offset_y:+.0f}) -> 边界({x_start},{x_end},{y_start},{y_end})")
        
        print(f"✅ 成功创建 {len(regions)} 个评估区域")
        return regions

    def _resize_input_if_needed(self, images):
        """🔧 调整输入尺寸以匹配模型期望"""
        # 获取当前尺寸
        current_size = images.shape[-1]  # 假设是正方形
        target_size = self.config.layer_size
        
        if current_size != target_size:
            # print(f"🔧 调整输入尺寸: {current_size}×{current_size} -> {target_size}×{target_size}")
            
            # 保存原始形状
            original_shape = images.shape
            
            # 重塑为 [B*M*W, H, W] 进行插值
            B, M, W, H, _ = original_shape
            images_reshaped = images.view(B * M * W, 1, H, current_size)
            
            # 分别处理实部和虚部
            real_part = images_reshaped.real
            imag_part = images_reshaped.imag
            
            # 使用双线性插值调整尺寸
            real_resized = F.interpolate(real_part, size=(target_size, target_size), 
                                       mode='bilinear', align_corners=False)
            imag_resized = F.interpolate(imag_part, size=(target_size, target_size), 
                                       mode='bilinear', align_corners=False)
            
            # 重新组合复数
            images_resized = torch.complex(real_resized, imag_resized)
            
            # 恢复原始形状
            images = images_resized.view(B, M, W, target_size, target_size)
            
            # print(f"✅ 尺寸调整完成: {images.shape}")
        
        return images

    def train_model(self, num_layers=None):
        """
        训练单个模型 - 适配多模式多波长，修复复数损失问题
        
        参数:
            num_layers: 层数（如果为None，使用config中的默认值）
        """
        if num_layers is not None:
            # 临时修改config中的层数
            original_num_layers = self.config.num_layers
            self.config.num_layers = num_layers
        
        print(f"\n🎯 开始训练 {self.config.num_layers} 层模型...")
        
        try:
            # 创建数据加载器和模型
            train_loader = self.data_generator.create_dataloader()
            
            # 使用修复后的模型类
            if self.model_class is None:
                from model import MultiModeMultiWavelengthModel
                self.model_class = MultiModeMultiWavelengthModel
            
            model = self.model_class(
                self.config, 
                self.data_generator, 
                evaluation_regions=self.evaluation_regions
            ).to(device)
            
            print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 训练循环
            losses = self._train_loop(model, train_loader)
            
            # 评估模型
            evaluation_results = self._evaluate_model(model, train_loader)
            
            # 恢复原始层数配置
            if num_layers is not None:
                self.config.num_layers = original_num_layers
            
            return {
                'model': model,
                'losses': losses,
                'phase_masks': self._extract_phase_masks(model),
                'weights_pred': evaluation_results['weights_pred'],
                'visibility': evaluation_results['visibility'],
                'layer_statistics': evaluation_results.get('layer_statistics', None)
            }
            
        except Exception as e:
            print(f"❌ 训练过程中出错: {e}")
            print(f"   错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # 恢复原始层数配置
            if num_layers is not None:
                self.config.num_layers = original_num_layers
            
            # 返回空结果
            return {
                'model': None,
                'losses': [],
                'phase_masks': None,
                'weights_pred': None,
                'visibility': 0.0,
                'layer_statistics': None
            }

    def _train_loop(self, model, train_loader):
        """训练循环 - 使用支持复数的损失函数"""
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay)
        losses = []
        
        print(f"🔄 开始训练，共 {self.config.epochs} 轮...")
        print(f"🔧 使用损失函数: {type(self.criterion).__name__}")
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                try:
                    # 处理不同的数据格式
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        images, labels = batch_data
                    else:
                        images = batch_data
                        labels = None
                    
                    # 确保输入是正确的格式
                    if len(images.shape) == 4:
                        B, C, H, W = images.shape
                        if C == self.num_wavelengths:
                            images = images.unsqueeze(1)
                        else:
                            images = images.view(B, self.num_modes, self.num_wavelengths, H, W)
                    
                    # 调整输入尺寸以匹配模型
                    images = self._resize_input_if_needed(images)
                    images = images.to(device, dtype=torch.complex64)
                    
                    if labels is not None:
                        labels = labels.to(device, dtype=torch.complex64)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    
                    # 🔧 使用支持复数的损失函数
                    if isinstance(self.criterion, ComplexPhysicsConstrainedLoss):
                        # 物理约束损失
                        input_energy = torch.sum(torch.abs(images)**2, dim=(-2, -1))
                        loss = self.criterion(outputs, labels, input_energy)
                    else:
                        # 简单复数MSE损失
                        loss = self.criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # 只在第一个epoch显示详细信息
                    if epoch == 0 and batch_idx < 3:
                        print(f"   批次 {batch_idx}: 损失 = {loss.item():.6f}")
                    
                except Exception as e:
                    print(f"❌ 批次 {batch_idx} 处理出错: {e}")
                    continue
            
            scheduler.step()
            avg_loss = epoch_loss / max(batch_count, 1)
            losses.append(avg_loss)
            
            if epoch % 50 == 0 or epoch == self.config.epochs - 1:
                print(f'   Epoch [{epoch+1}/{self.config.epochs}], Loss: {avg_loss:.8f}')
        
        print("✅ 训练完成!")
        return losses

    def _extract_phase_masks(self, model):
        """提取相位掩膜 - 适配多波长"""
        if model is None:
            return None
            
        phase_masks = []
        
        try:
            for layer_idx, layer in enumerate(model.layers):
                # 获取该层每个波长的有效相位掩膜
                effective_masks = model.get_effective_phase_masks_for_layer(layer)
                phase_masks.append(effective_masks)
                
                print(f"📋 提取第 {layer_idx+1} 层相位掩膜:")
                for wl_idx, mask in enumerate(effective_masks):
                    wl_nm = self.config.wavelengths[wl_idx] * 1e9
                    print(f"   波长 {wl_nm:.1f}nm: 形状 {mask.shape}, 范围 [{mask.min():.4f}, {mask.max():.4f}]")
        except Exception as e:
            print(f"❌ 提取相位掩膜时出错: {e}")
            return None
        
        return phase_masks

    def _evaluate_model(self, model, test_loader):
        """评估模型 - 适配多模式多波长输出"""
        if model is None:
            return {'weights_pred': None, 'visibility': 0.0, 'layer_statistics': None}
            
        model.eval()
        all_weights_pred = []
        layer_stats = None
        
        print("🔍 开始模型评估...")
        
        try:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    # 处理数据格式
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        images, labels = batch_data
                    else:
                        images = batch_data
                        labels = None
                    
                    # 确保输入格式正确
                    if len(images.shape) == 4:
                        B, C, H, W = images.shape
                        if C == self.num_wavelengths:
                            images = images.unsqueeze(1)
                        else:
                            images = images.view(B, self.num_modes, self.num_wavelengths, H, W)
                    
                    # 调整尺寸
                    images = self._resize_input_if_needed(images)
                    images = images.to(device, dtype=torch.complex64)
                    predictions = model(images)
                    
                    # 分析第一个批次的层统计信息
                    if batch_idx == 0:
                        layer_stats = model.analyze_layer_statistics(images)
                    
                    # 计算检测区域的权重
                    weights_batch = self._compute_detection_weights(predictions)
                    all_weights_pred.append(weights_batch.cpu())
                    
                    if batch_idx == 0:  # 只处理第一个批次进行演示
                        break
            
            weights_pred = torch.cat(all_weights_pred, dim=0).numpy()
            visibility = self._calculate_visibility(weights_pred)
            
            print(f"✅ 评估完成，可见度: {visibility:.6f}")
            
            return {
                'weights_pred': weights_pred, 
                'visibility': visibility,
                'layer_statistics': layer_stats
            }
            
        except Exception as e:
            print(f"❌ 模型评估时出错: {e}")
            return {'weights_pred': None, 'visibility': 0.0, 'layer_statistics': None}

    def _compute_detection_weights(self, predictions):
        """计算检测区域权重 - 适配多模式多波长"""
        # predictions shape: [B, num_modes, num_wavelengths, H, W]
        B, num_modes, num_wavelengths, H, W = predictions.shape
        N_det = len(self.evaluation_regions)
        
        # 为每个批次、模式、波长计算检测权重
        all_weights = []
        
        for b in range(B):
            batch_weights = []
            for m in range(num_modes):
                mode_weights = []
                for c in range(num_wavelengths):
                    # 获取当前模式和波长的预测
                    pred = predictions[b, m, c]  # [H, W]
                    
                    # 计算每个检测区域的能量（使用强度而不是复数）
                    energies = []
                    for (xs, xe, ys, ye) in self.evaluation_regions:
                        region_intensity = torch.abs(pred[ys:ye, xs:xe])**2
                        region_sum = region_intensity.sum()
                        energies.append(region_sum)
                    
                    mode_weights.append(torch.stack(energies))
                batch_weights.append(torch.stack(mode_weights))
            all_weights.append(torch.stack(batch_weights))
        
        # 形状: [B, num_modes, num_wavelengths, N_det]
        return torch.stack(all_weights)

    def _calculate_visibility(self, weights):
        """计算可见度 - 保持与原代码一致，处理None值"""
        if weights is None:
            return 0.0
            
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()
        
        # weights shape: [N, num_modes, num_wavelengths, N_det]
        N = weights.shape[0]
        visibilities = []
        
        # 对所有维度计算可见度
        for n in range(N):
            for m in range(weights.shape[1]):  # 模式
                for c in range(weights.shape[2]):  # 波长
                    intensities = weights[n, m, c, :]  # 检测区域
                    I_max = np.max(intensities)
                    I_min = np.min(intensities)
                    
                    if I_max + I_min > 1e-12:
                        visibility = (I_max - I_min) / (I_max + I_min)
                    else:
                        visibility = 0.0
                    
                    visibilities.append(visibility)
        
        return float(np.mean(visibilities))

    def train_multiple_models(self, num_layer_options):
        """训练多个不同层数的模型 - 保持与原代码接口一致"""
        results = {
            'models': [], 
            'losses': [], 
            'phase_masks': [], 
            'weights_pred': [], 
            'visibility': [],
            'layer_statistics': []
        }
        
        print(f"🎯 开始训练多个模型，层数选项: {num_layer_options}")
        
        for i, num_layers in enumerate(num_layer_options):
            print(f"\n{'='*50}")
            print(f"🔄 训练第 {i+1}/{len(num_layer_options)} 个模型 ({num_layers} 层)")
            print(f"{'='*50}")
            
            try:
                model_result = self.train_model(num_layers)
                
                for k in results:
                    if k in model_result:
                        results[k].append(model_result[k])
                    else:
                        results[k].append(None)
                        
            except Exception as e:
                print(f"❌ 训练 {num_layers} 层模型时出错: {e}")
                # 添加默认结果以保持列表长度一致
                results['models'].append(None)
                results['losses'].append([])
                results['phase_masks'].append(None)
                results['weights_pred'].append(None)
                results['visibility'].append(0.0)
                results['layer_statistics'].append(None)
        
        print(f"\n🎉 所有模型训练完成!")
        self._print_summary(results, num_layer_options)
        
        return results

    def _print_summary(self, results, num_layer_options):
        """打印训练总结"""
        print(f"\n📊 训练总结:")
        print(f"{'层数':<6} {'最终损失':<12} {'可见度':<10}")
        print("-" * 30)
        
        for i, num_layers in enumerate(num_layer_options):
            final_loss = results['losses'][i][-1] if results['losses'][i] else 'N/A'
            visibility = results['visibility'][i] if results['visibility'][i] else 'N/A'
            
            if isinstance(final_loss, float):
                final_loss = f"{final_loss:.6f}"
            if isinstance(visibility, float):
                visibility = f"{visibility:.6f}"
                
            print(f"{num_layers:<6} {final_loss:<12} {visibility:<10}")

    def set_detector_regions(self, regions):
        """设置检测区域 - 保持与原代码一致"""
        self.evaluation_regions = regions
        print(f"✅ 已设置 {len(regions)} 个检测区域")
        return True
