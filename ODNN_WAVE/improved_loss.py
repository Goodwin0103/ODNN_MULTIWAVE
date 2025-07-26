# 创建新文件: improved_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedMultiWavelengthLoss:
    """
    改进的多波长损失函数
    解决波长间能量不平衡和模式混淆问题
    """
    
    def __init__(self, wavelengths, num_modes, 
                 wavelength_weights=None, 
                 separation_weight=0.1, 
                 focus_weight=0.2):
        """
        Args:
            wavelengths: 波长列表
            num_modes: 模式数量
            wavelength_weights: 波长权重字典
            separation_weight: 模式分离损失权重
            focus_weight: 聚焦损失权重
        """
        self.wavelengths = wavelengths
        self.num_modes = num_modes
        self.separation_weight = separation_weight
        self.focus_weight = focus_weight
        
        # 默认波长权重 - 基于观察到的问题设置
        if wavelength_weights is None:
            self.wavelength_weights = {}
            for wl in wavelengths:
                wl_nm = int(wl * 1e9)
                if wl_nm == 450:
                    self.wavelength_weights[wl_nm] = 1.0    # 450nm正常
                elif wl_nm == 550:
                    self.wavelength_weights[wl_nm] = 0.4   # 550nm降权
                elif wl_nm == 650:
                    self.wavelength_weights[wl_nm] = 2.0   # 650nm增权
                else:
                    self.wavelength_weights[wl_nm] = 1.0
        else:
            self.wavelength_weights = wavelength_weights
            
        print(f"💡 损失函数权重设置: {self.wavelength_weights}")
    
    def __call__(self, predictions, targets):
        """计算总损失"""
        # 1. 带权重的MSE损失
        weighted_mse = self._weighted_mse_loss(predictions, targets)
        
        # 2. 模式分离损失
        separation_loss = self._mode_separation_loss(predictions)
        
        # 3. 聚焦损失
        focus_loss = self._focus_loss(predictions, targets)
        
        # 总损失
        total_loss = (weighted_mse + 
                     self.separation_weight * separation_loss + 
                     self.focus_weight * focus_loss)
        
        return total_loss, {
            'mse': weighted_mse.item(),
            'separation': separation_loss.item(),
            'focus': focus_loss.item(),
            'total': total_loss.item()
        }
    
    def _weighted_mse_loss(self, predictions, targets):
        """带波长权重的MSE损失"""
        total_loss = 0
        total_weight = 0
        
        for wl_idx, wavelength in enumerate(self.wavelengths):
            wl_nm = int(wavelength * 1e9)
            weight = self.wavelength_weights.get(wl_nm, 1.0)
            
            # predictions: [batch, wavelength, mode, height, width]
            wl_pred = predictions[:, wl_idx]  # [batch, mode, height, width]
            wl_target = targets[:, wl_idx]
            
            wl_loss = F.mse_loss(wl_pred, wl_target, reduction='mean')
            total_loss += weight * wl_loss
            total_weight += weight
        
        return total_loss / total_weight
    
    def _mode_separation_loss(self, predictions):
        """模式分离损失"""
        if self.num_modes <= 1:
            return torch.tensor(0.0, device=predictions.device)
        
        separation_loss = 0
        count = 0
        
        for wl_idx in range(len(self.wavelengths)):
            for mode_i in range(self.num_modes):
                for mode_j in range(mode_i + 1, self.num_modes):
                    # 同一波长下不同模式应该有不同响应
                    pred_i = predictions[:, wl_idx, mode_i].flatten(1)  # [batch, -1]
                    pred_j = predictions[:, wl_idx, mode_j].flatten(1)
                    
                    # 计算相似度
                    similarity = F.cosine_similarity(pred_i, pred_j, dim=1)
                    separation_loss += torch.mean(torch.abs(similarity))
                    count += 1
        
        return separation_loss / count if count > 0 else torch.tensor(0.0)
    
    def _focus_loss(self, predictions, targets):
        """聚焦损失 - 鼓励能量集中"""
        focus_loss = 0
        
        for wl_idx in range(len(self.wavelengths)):
            for mode_idx in range(self.num_modes):
                pred = predictions[:, wl_idx, mode_idx]  # [batch, height, width]
                target = targets[:, wl_idx, mode_idx]
                
                # 计算预测分布的熵
                pred_flat = pred.flatten(1) + 1e-8  # [batch, -1]
                pred_prob = F.softmax(pred_flat, dim=1)
                pred_entropy = -torch.sum(pred_prob * torch.log(pred_prob), dim=1)
                
                # 计算目标分布的熵
                target_flat = target.flatten(1) + 1e-8
                target_prob = F.softmax(target_flat, dim=1)
                target_entropy = -torch.sum(target_prob * torch.log(target_prob), dim=1)
                
                # 熵匹配损失
                focus_loss += F.mse_loss(pred_entropy, target_entropy)
        
        return focus_loss / (len(self.wavelengths) * self.num_modes)
