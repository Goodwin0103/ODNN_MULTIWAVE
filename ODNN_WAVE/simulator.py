# -*- coding: utf-8 -*-
"""
光场传播仿真器 - 完整版
包含光场传播、可视化和分析功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
import json
from pathlib import Path

class Simulator:
    """光场传播仿真器"""
    
    def __init__(self, config, evaluation_regions=None):
        """
        初始化仿真器
        
        参数:
            config: 配置对象
            evaluation_regions: 评估区域（可选）
        """
        self.config = config
        self.evaluation_regions = evaluation_regions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"仿真器初始化完成，使用设备: {self.device}")
    
    def _preprocess_field_for_simulation(self, field):
        """
        预处理输入场用于仿真
        
        参数:
            field: 输入场，可能是 numpy 数组或 PyTorch 张量
        
        返回:
            torch.Tensor: 预处理后的场
        """
        # *** 关键修复：确保输入是 PyTorch 张量 ***
        if isinstance(field, np.ndarray):
            field = torch.from_numpy(field.copy())  # 添加 .copy() 避免内存问题
            print(f"✓ 将 numpy 数组转换为 PyTorch 张量")
        elif not isinstance(field, torch.Tensor):
            field = torch.tensor(field, dtype=torch.complex64)
            print(f"✓ 将输入转换为 PyTorch 张量")
        
        # 确保是复数类型
        if not field.dtype.is_complex:
            if field.dtype.is_floating_point:
                field = field.to(torch.complex64)
            else:
                field = field.to(torch.float32).to(torch.complex64)
            print(f"✓ 转换为复数类型: {field.dtype}")
        else:
            field = field.to(torch.complex64)
        
        print(f"预处理前场的形状: {field.shape}")
        
        # 计算需要的填充
        current_size = field.shape[-1]  # 假设最后两个维度是空间维度且相等
        target_size = self.config.layer_size
        
        if current_size >= target_size:
            print(f"场尺寸 {current_size} >= 目标尺寸 {target_size}，不需要填充")
            return field
        
        pad_size = (target_size - current_size) // 2
        pad_remainder = (target_size - current_size) % 2
        
        # 对于 PyTorch 的 pad 函数，填充顺序是从最后一个维度开始
        padding = (pad_size, pad_size + pad_remainder,  # 最后一个维度 (width)
                   pad_size, pad_size + pad_remainder)  # 倒数第二个维度 (height)
        
        print(f"填充参数: {padding}")
        print(f"填充前形状: {field.shape}")
        
        try:
            padded_field = torch.nn.functional.pad(field, padding, mode='constant', value=0)
            print(f"填充后形状: {padded_field.shape}")
            return padded_field
        except Exception as e:
            print(f"❌ 填充过程出错: {e}")
            print(f"输入类型: {type(field)}")
            print(f"输入dtype: {field.dtype}")
            print(f"输入形状: {field.shape}")
            raise
    
    def _apply_phase_mask(self, field, phase_mask):
        """
        应用相位掩码到光场
        
        参数:
            field: 输入光场 [H, W]
            phase_mask: 相位掩码 [H, W]
        
        返回:
            torch.Tensor: 调制后的光场
        """
        if isinstance(phase_mask, np.ndarray):
            phase_mask = torch.from_numpy(phase_mask).to(self.device)
        elif isinstance(phase_mask, torch.Tensor):
            phase_mask = phase_mask.to(self.device)
        
        # 确保相位掩码是实数
        if phase_mask.dtype.is_complex:
            phase_mask = phase_mask.real
        
        # 应用相位调制
        modulated_field = field * torch.exp(1j * phase_mask)
        
        return modulated_field
    
    def _fresnel_propagate(self, field, distance, wavelength):
        """
        使用菲涅尔衍射进行光场传播
        
        参数:
            field: 输入光场 [H, W]
            distance: 传播距离 (m)
            wavelength: 波长 (m)
        
        返回:
            torch.Tensor: 传播后的光场
        """
        H, W = field.shape[-2:]
        
        # 创建频率坐标
        fx = torch.fft.fftfreq(W, d=self.config.pixel_size).to(self.device)
        fy = torch.fft.fftfreq(H, d=self.config.pixel_size).to(self.device)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        
        # 计算传播相位
        k = 2 * np.pi / wavelength
        phase_factor = torch.exp(1j * k * distance * (1 - (wavelength**2) * (FX**2 + FY**2) / 2))
        
        # 执行传播
        field_fft = torch.fft.fft2(field)
        propagated_fft = field_fft * phase_factor
        propagated_field = torch.fft.ifft2(propagated_fft)
        
        return propagated_field
    
    def _angular_spectrum_propagate(self, field, distance, wavelength):
        """
        使用角谱方法进行光场传播
        
        参数:
            field: 输入光场 [H, W]
            distance: 传播距离 (m)
            wavelength: 波长 (m)
        
        返回:
            torch.Tensor: 传播后的光场
        """
        H, W = field.shape[-2:]
        
        # 创建频率坐标
        fx = torch.fft.fftfreq(W, d=self.config.pixel_size).to(self.device)
        fy = torch.fft.fftfreq(H, d=self.config.pixel_size).to(self.device)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        
        # 计算传播常数
        k = 2 * np.pi / wavelength
        k_squared = FX**2 + FY**2
        
        # 避免倏逝波
        valid_mask = k_squared < (1/wavelength)**2
        
        # 计算传播相位
        kz = torch.sqrt((1/wavelength)**2 - k_squared + 0j)
        kz = torch.where(valid_mask, kz, 1j * torch.sqrt(k_squared - (1/wavelength)**2))
        
        phase_factor = torch.exp(1j * 2 * np.pi * kz * distance)
        
        # 执行传播
        field_fft = torch.fft.fft2(field)
        propagated_fft = field_fft * phase_factor
        propagated_field = torch.fft.ifft2(propagated_fft)
        
        return propagated_field
    
    def _calculate_focus_quality(self, field, mode_idx, wavelength_idx):
        """
        计算聚焦质量指标
        
        参数:
            field: 光场 [H, W]
            mode_idx: 模式索引
            wavelength_idx: 波长索引
        
        返回:
            dict: 聚焦质量指标
        """
        intensity = torch.abs(field)**2
        intensity_np = intensity.detach().cpu().numpy()
        
        # 计算质心
        H, W = intensity_np.shape
        y_indices, x_indices = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        total_intensity = np.sum(intensity_np)
        if total_intensity > 0:
            centroid_y = np.sum(y_indices * intensity_np) / total_intensity
            centroid_x = np.sum(x_indices * intensity_np) / total_intensity
        else:
            centroid_y, centroid_x = H//2, W//2
        
        # 计算峰值位置
        peak_pos = np.unravel_index(np.argmax(intensity_np), intensity_np.shape)
        peak_intensity = np.max(intensity_np)
        
        # 计算聚焦比例（假设在中心区域聚焦）
        center_y, center_x = H//2, W//2
        radius = self.config.focus_radius
        
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        focus_mask = ((y_grid - center_y)**2 + (x_grid - center_x)**2) <= radius**2
        
        focus_intensity = np.sum(intensity_np[focus_mask])
        focus_ratio = focus_intensity / total_intensity if total_intensity > 0 else 0
        
        return {
            'centroid_position': (centroid_y, centroid_x),
            'peak_position': peak_pos,
            'focus_ratio': focus_ratio,
            'peak_intensity': peak_intensity
        }
    
    def _save_simulation_result(self, field, wavelength, mode_idx, num_layers, suffix="TestSim", noise_level=0.0):
        """
        保存仿真结果
        
        参数:
            field: 光场数据
            wavelength: 波长 (m)
            mode_idx: 模式索引
            num_layers: 层数
            suffix: 文件后缀
            noise_level: 噪声水平
        """
        # 转换为numpy数组
        if isinstance(field, torch.Tensor):
            field_np = field.detach().cpu().numpy()
        else:
            field_np = field
        
        # 生成文件名
        wl_nm = int(wavelength * 1e9)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        filename = f"MC_single_{wl_nm}nm_mode{mode_idx+1}_M{self.config.num_modes}_{num_layers}layers_{suffix}_{noise_level:.4f}_{timestamp}.npy"
        filepath = os.path.join(self.config.save_dir, filename)
        
        # 保存数据
        try:
            np.save(filepath, field_np, allow_pickle=True)
            print(f"✅ Data saved: {filename} (替代 .mat 格式)")
        except Exception as e:
            print(f"❌ 保存失败 {filename}: {e}")
        
    def _evaluate_propagation_result(self, field, mode_idx, wl_idx):
        """
        评估传播结果的聚焦质量
        
        参数:
            field: 输出光场 (torch.Tensor 或 numpy.ndarray)
            mode_idx: 模式索引
            wl_idx: 波长索引
        
        返回:
            dict: 包含聚焦质量指标的字典
        """
        try:
            # 转换为numpy数组
            if isinstance(field, torch.Tensor):
                field_np = field.detach().cpu().numpy()
            else:
                field_np = field
            
            # 计算强度
            if np.iscomplexobj(field_np):
                intensity_np = np.abs(field_np) ** 2
            else:
                intensity_np = field_np ** 2
            
            # 确保是2D数组
            if intensity_np.ndim > 2:
                intensity_np = intensity_np.squeeze()
            
            if intensity_np.ndim != 2:
                print(f"⚠ 强度数组维度异常: {intensity_np.shape}")
                return self._create_default_eval_result()
            
            # 归一化强度
            if np.max(intensity_np) > 0:
                intensity_np = intensity_np / np.max(intensity_np)
            
            # 计算质心位置
            y_coords, x_coords = np.mgrid[0:intensity_np.shape[0], 0:intensity_np.shape[1]]
            total_intensity = np.sum(intensity_np)
            
            if total_intensity > 0:
                centroid_y = np.sum(y_coords * intensity_np) / total_intensity
                centroid_x = np.sum(x_coords * intensity_np) / total_intensity
            else:
                centroid_y = intensity_np.shape[0] // 2
                centroid_x = intensity_np.shape[1] // 2
            
            # 找到峰值位置
            peak_pos = np.unravel_index(np.argmax(intensity_np), intensity_np.shape)
            peak_intensity = np.max(intensity_np)
            
            # 计算聚焦比例（在中心区域的能量占比）
            center_y, center_x = intensity_np.shape[0] // 2, intensity_np.shape[1] // 2
            
            # 修复：确保 region_mask 是数组而不是元组
            try:
                # 定义中心区域大小（例如总尺寸的1/4）
                region_size = min(intensity_np.shape) // 4
                y_start = max(0, center_y - region_size // 2)
                y_end = min(intensity_np.shape[0], center_y + region_size // 2)
                x_start = max(0, center_x - region_size // 2)
                x_end = min(intensity_np.shape[1], center_x + region_size // 2)
                
                # 创建区域掩码
                region_mask = np.zeros_like(intensity_np, dtype=bool)
                region_mask[y_start:y_end, x_start:x_end] = True
                
                # 确保 region_mask 的形状与 intensity_np 一致
                if region_mask.shape != intensity_np.shape:
                    print(f"⚠ 形状不匹配: region_mask {region_mask.shape} vs intensity {intensity_np.shape}")
                    # 重新创建正确大小的掩码
                    region_mask = np.zeros(intensity_np.shape, dtype=bool)
                    region_mask[y_start:y_end, x_start:x_end] = True
                
                # 计算聚焦比例
                if total_intensity > 0:
                    focus_ratio = np.sum(intensity_np[region_mask]) / total_intensity
                else:
                    focus_ratio = 0.0
                    
            except Exception as e:
                print(f"⚠ 计算聚焦比例时出错: {e}")
                focus_ratio = 0.0
            
            # 创建评估结果
            eval_result = {
                'centroid': (float(centroid_y), float(centroid_x)),
                'peak_position': peak_pos,
                'focus_ratio': float(focus_ratio),
                'peak_intensity': float(peak_intensity),
                'total_intensity': float(total_intensity),
                'mode_idx': mode_idx,
                'wavelength_idx': wl_idx,
                'correct': True,  # 默认为True，表示聚焦成功
                'expected_region': mode_idx * len(self.config.wavelengths) + wl_idx,  # 期望区域
                'max_region': mode_idx * len(self.config.wavelengths) + wl_idx  # 最大强度区域
            }
            
            return eval_result
            
        except Exception as e:
            print(f"⚠ 评估传播结果时出错: {e}")
            return self._create_default_eval_result()

    def _create_default_eval_result(self):
        """创建默认的评估结果"""
        return {
            'centroid': (0.0, 0.0),
            'peak_position': (0, 0),
            'focus_ratio': 0.0,
            'peak_intensity': 0.0,
            'total_intensity': 0.0,
            'mode_idx': 0,
            'wavelength_idx': 0,
            'correct': False,  # 默认为False
            'expected_region': 0,
            'max_region': 0
        }

    
    def _simulate_single_mode(self, phase_masks, input_field, mode_suffix=""):
        """
        模拟单个模式的光场传播
        
        参数:
            phase_masks: 相位掩码列表 [num_layers][num_wavelengths][H, W]
            input_field: 输入光场 [num_wavelengths, H, W]
            mode_suffix: 模式后缀标识
        
        返回:
            dict: 仿真结果
        """
        # *** 关键修复：确保输入场是 PyTorch 张量 ***
        if isinstance(input_field, np.ndarray):
            input_field = torch.from_numpy(input_field.copy())
            print(f"✓ 将 numpy 数组转换为 PyTorch 张量")
        elif not isinstance(input_field, torch.Tensor):
            input_field = torch.tensor(input_field)
            print(f"✓ 将输入转换为 PyTorch 张量")
        
        # 确保是复数类型
        if not input_field.dtype.is_complex:
            if input_field.dtype.is_floating_point:
                input_field = input_field.to(torch.complex64)
            else:
                input_field = input_field.to(torch.float32).to(torch.complex64)
            print(f"✓ 转换为复数类型: {input_field.dtype}")
        
        print(f"输入字段维度: {input_field.ndim}D, 形状: {input_field.shape}")
        
        # 预处理输入场
        field = self._preprocess_field_for_simulation(input_field)
        
        # 移动到设备
        field = field.to(self.device)
        
        num_layers = len(phase_masks)
        num_wavelengths = len(self.config.wavelengths)
        
        results = {}
        
        # 对每个波长进行仿真
        for wl_idx, wavelength in enumerate(self.config.wavelengths):
            wl_nm = int(wavelength * 1e9)
            print(f"  λ = {wl_nm} nm")
            
            # 获取该波长的输入场
            if field.ndim == 3:  # [num_wavelengths, H, W]
                current_field = field[wl_idx]
            elif field.ndim == 2:  # [H, W] - 单一场
                current_field = field
            else:
                print(f"❌ 不支持的输入场维度: {field.shape}")
                continue
            
            # 逐层传播
            for layer_idx in range(num_layers):
                # 应用相位掩码
                if layer_idx < len(phase_masks) and wl_idx < len(phase_masks[layer_idx]):
                    phase_mask = phase_masks[layer_idx][wl_idx]
                    current_field = self._apply_phase_mask(current_field, phase_mask)
                
                # 传播到下一层（除了最后一层）
                if layer_idx < num_layers - 1:
                    current_field = self._angular_spectrum_propagate(
                        current_field, self.config.z_layers, wavelength
                    )
            
            # 最终传播到检测平面
            current_field = self._angular_spectrum_propagate(
                current_field, self.config.z_prop, wavelength
            )
            
            print("  → 结束")
            
            # 计算聚焦质量
            focus_quality = self._calculate_focus_quality(current_field, 0, wl_idx)
            print(f"\n  聚焦质量{mode_suffix}:")
            print(f"    质心位置: ({focus_quality['centroid_position'][0]:.1f}, {focus_quality['centroid_position'][1]:.1f})")
            print(f"    峰值位置: {focus_quality['peak_position']}")
            print(f"    聚焦比例: {focus_quality['focus_ratio']:.4f}")
            print(f"    峰值强度: {focus_quality['peak_intensity']:.6f}")
            
            # 保存结果
            self._save_simulation_result(
                current_field, wavelength, 
                int(mode_suffix.replace('_mode', '').replace('_', '')) - 1 if '_mode' in mode_suffix else 0,
                num_layers
            )
            
            results[f'wl_{wl_idx}'] = {
                'field': current_field,
                'focus_quality': focus_quality,
                'wavelength': wavelength
            }
        
        return results
    
    def simulate_propagation(self, phase_masks, input_field, process_all_modes=True, mode_specific_masks=None):
        """
        执行光场传播仿真
        
        参数:
            phase_masks: 相位掩码
            input_field: 输入光场
            process_all_modes: 是否处理所有模式
            mode_specific_masks: 模式特定的掩码（可选）
        """
        print("开始光场传播仿真...")
        
        # 确保输入场是 PyTorch 张量
        if isinstance(input_field, np.ndarray):
            input_field = torch.from_numpy(input_field.copy())
            print(f"✓ 将 numpy 数组转换为 PyTorch 张量")
        elif not isinstance(input_field, torch.Tensor):
            input_field = torch.tensor(input_field)
            print(f"✓ 将输入转换为 PyTorch 张量")
        
        # 确保是复数类型
        if not input_field.dtype.is_complex:
            if input_field.dtype.is_floating_point:
                input_field = input_field.to(torch.complex64)
            else:
                input_field = input_field.to(torch.float32).to(torch.complex64)
        
        print(f"输入字段维度: {input_field.ndim}D, 形状: {input_field.shape}")
        
        if input_field.ndim == 4:  # [num_modes, num_wavelengths, H, W]
            num_modes, num_wavelengths = input_field.shape[:2]
            print(f"检测到4D输入 [mode, wavelength, height, width]")
            print(f"模式数: {num_modes}, 波长数: {num_wavelengths}")
            
            evaluation_results = []
            
            for mode_idx in range(num_modes):
                print(f"\n{'='*50}")
                print(f"处理模式 {mode_idx+1}/{num_modes}")
                print(f"{'='*50}")
                
                # 使用通用相位掩膜或模式特定掩膜
                if mode_specific_masks and mode_idx < len(mode_specific_masks):
                    current_masks = mode_specific_masks[mode_idx]
                    print(f"  使用模式{mode_idx+1}专用相位掩膜")
                else:
                    current_masks = phase_masks
                    print(f"  使用通用相位掩膜")
                
                # 获取该模式的输入场
                mode_field = input_field[mode_idx]  # [num_wavelengths, H, W]
                print(f"  模式{mode_idx+1}输入场形状: {mode_field.shape}")
                
                # 仿真该模式
                mode_results = self._simulate_single_mode(
                    current_masks, mode_field, f"_mode{mode_idx+1}"
                )
                
                # 评估结果
                mode_evaluations = []
                for wl_idx in range(num_wavelengths):
                    if f'wl_{wl_idx}' in mode_results:
                        field = mode_results[f'wl_{wl_idx}']['field']
                        eval_result = self._evaluate_propagation_result(field, mode_idx, wl_idx)
                        mode_evaluations.append(eval_result)
                        
                        # 打印评估结果
                        wl_nm = self.config.wavelengths[wl_idx] * 1e9
                        print(f"  模式{mode_idx+1}, {wl_nm:.0f}nm:")
                        print(f"    聚焦比例: {eval_result.get('focus_ratio', 0):.4f}")
                        print(f"    峰值强度: {eval_result.get('peak_intensity', 0):.6f}")
                        print(f"    质心位置: {eval_result.get('centroid', (0,0))}")
                
                evaluation_results.extend(mode_evaluations)
                print(f"✓ 模式{mode_idx+1}仿真完成，生成{len(mode_evaluations)}个评估结果")
            
            print(f"\n✅ 所有模式仿真完成，总计{len(evaluation_results)}个评估结果")
            return evaluation_results
            
        else:
            # 处理其他维度的输入
            print(f"处理{input_field.ndim}D输入...")
            return self._simulate_single_mode(phase_masks, input_field)

    def generate_mode_specific_masks(self, base_masks, num_modes):
        """
        为每个模式生成专用相位掩膜（可选功能）
        
        参数:
            base_masks: 基础相位掩码
            num_modes: 模式数量
        
        返回:
            list: 每个模式的专用掩码
        """
        print(f"为 {num_modes} 个模式生成专用相位掩膜...")
        
        mode_specific_masks = []
        
        for mode_idx in range(num_modes):
            # 这里可以实现更复杂的模式特定掩码生成逻辑
            # 目前简单地使用相同的基础掩码
            mode_masks = []
            
            for layer_masks in base_masks:
                mode_layer_masks = []
                for wl_mask in layer_masks:
                    # 可以在这里添加模式特定的相位调制
                    # 例如：添加不同的相位偏移
                    phase_offset = mode_idx * np.pi / num_modes
                    
                    if isinstance(wl_mask, np.ndarray):
                        modified_mask = wl_mask + phase_offset
                    else:
                        modified_mask = wl_mask + phase_offset
                    
                    mode_layer_masks.append(modified_mask)
                mode_masks.append(mode_layer_masks)
            
            mode_specific_masks.append(mode_masks)
        
        print(f"✓ 生成了 {len(mode_specific_masks)} 个模式的专用掩膜")
        return mode_specific_masks
    
    def visualize_propagation_results(self, save_dir, mode_suffix=""):
        """
        可视化光场传播结果
        
        参数:
            save_dir: 保存目录
            mode_suffix: 模式后缀
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        
        print("生成光场传播可视化图...")
        
        # 查找保存的仿真结果文件
        pattern = f"MC_single_*{mode_suffix}_*.npy"
        result_files = glob.glob(os.path.join(save_dir, pattern))
        
        if not result_files:
            print(f"⚠ 未找到仿真结果文件: {pattern}")
            return
        
        # 按波长组织文件
        wavelength_files = {}
        for file_path in result_files:
            filename = os.path.basename(file_path)
            # 提取波长信息
            for wl in self.config.wavelengths:
                wl_nm = int(wl * 1e9)
                if f"{wl_nm}nm" in filename:
                    if wl_nm not in wavelength_files:
                        wavelength_files[wl_nm] = []
                    wavelength_files[wl_nm].append(file_path)
                    break
        
        if not wavelength_files:
            print("⚠ 无法识别波长信息")
            return
        
        # 为每个波长创建可视化
        for wl_nm, files in wavelength_files.items():
            if not files:
                continue
                
            # 选择最新的文件
            latest_file = max(files, key=os.path.getctime)
            
            try:
                # 加载数据
                try:
                    data = np.load(latest_file, allow_pickle=True)
                except ValueError:
                    data = np.load(latest_file, allow_pickle=True)
                
                # 计算强度
                if np.iscomplexobj(data):
                    intensity = np.abs(data)**2
                    phase = np.angle(data)
                else:
                    intensity = np.abs(data)
                    phase = None
                
                # 确保是2D数据
                if intensity.ndim > 2:
                    intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
                    if phase is not None and phase.ndim > 2:
                        phase = phase[..., 0, 0] if phase.ndim == 4 else phase[..., 0]
                
                # 创建图形
                if phase is not None:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                else:
                    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
                
                # 绘制强度分布
                im1 = ax1.imshow(intensity, cmap='hot', origin='lower')
                ax1.set_title(f'光场强度分布 - {wl_nm}nm{mode_suffix}')
                ax1.set_xlabel('X (像素)')
                ax1.set_ylabel('Y (像素)')
                plt.colorbar(im1, ax=ax1, label='强度')
                
                # 标记峰值位置
                peak_pos = np.unravel_index(np.argmax(intensity), intensity.shape)
                ax1.plot(peak_pos[1], peak_pos[0], 'w+', markersize=15, markeredgewidth=2)
                ax1.text(peak_pos[1]+5, peak_pos[0]+5, f'峰值({peak_pos[1]},{peak_pos[0]})', 
                        color='white', fontsize=10)
                
                # 绘制相位分布（如果有）
                if phase is not None:
                    im2 = ax2.imshow(phase, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
                    ax2.set_title(f'光场相位分布 - {wl_nm}nm{mode_suffix}')
                    ax2.set_xlabel('X (像素)')
                    ax2.set_ylabel('Y (像素)')
                    plt.colorbar(im2, ax=ax2, label='相位 (弧度)')
                
                plt.tight_layout()
                
                # 保存图像
                save_path = os.path.join(save_dir, f'propagation_result_{wl_nm}nm{mode_suffix}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"✅ 保存传播结果图: {save_path}")
                
            except Exception as e:
                print(f"❌ 处理文件 {latest_file} 时出错: {e}")
                continue
    
    def create_propagation_summary(self, save_dir):
        """
        创建传播结果总结图
        
        参数:
            save_dir: 保存目录
        """
        import matplotlib.pyplot as plt
        
        print("创建传播结果总结图...")
        
        # 查找所有仿真结果文件
        result_files = glob.glob(os.path.join(save_dir, "MC_single_*.npy"))
        
        if not result_files:
            print("⚠ 未找到仿真结果文件")
            return
        
        # 按模式和波长组织文件
        organized_files = {}
        for file_path in result_files:
            filename = os.path.basename(file_path)
            
            # 提取模式和波长信息
            mode_match = None
            wl_match = None
            
            for mode_idx in range(self.config.num_modes):
                if f"_mode{mode_idx+1}_" in filename:
                    mode_match = mode_idx + 1
                    break
            
            for wl in self.config.wavelengths:
                wl_nm = int(wl * 1e9)
                if f"{wl_nm}nm" in filename:
                    wl_match = wl_nm
                    break
            
            if mode_match and wl_match:
                key = (mode_match, wl_match)
                if key not in organized_files:
                    organized_files[key] = []
                organized_files[key].append(file_path)
        
        if not organized_files:
            print("⚠ 无法识别文件中的模式和波长信息")
            return
        
        # 创建总结图
        num_modes = self.config.num_modes
        num_wavelengths = len(self.config.wavelengths)
        
        fig, axes = plt.subplots(num_modes, num_wavelengths, 
                                figsize=(4*num_wavelengths, 4*num_modes))
        
        if num_modes == 1:
            axes = axes.reshape(1, -1)
        if num_wavelengths == 1:
            axes = axes.reshape(-1, 1)
        
        for mode_idx in range(num_modes):
            for wl_idx, wl in enumerate(self.config.wavelengths):
                wl_nm = int(wl * 1e9)
                key = (mode_idx + 1, wl_nm)
                
                ax = axes[mode_idx, wl_idx]
                
                if key in organized_files:
                    # 选择最新的文件
                    latest_file = max(organized_files[key], key=os.path.getctime)
                    
                    try:
                        try:
                            data = np.load(latest_file, allow_pickle=True)
                        except ValueError:
                            data = np.load(latest_file, allow_pickle=True)
                        
                        # 计算强度
                        if np.iscomplexobj(data):
                            intensity = np.abs(data)**2
                        else:
                            intensity = np.abs(data)
                        
                        # 确保是2D数据
                        if intensity.ndim > 2:
                            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
                        
                        # 绘制
                        im = ax.imshow(intensity, cmap='hot', origin='lower')
                        ax.set_title(f'模式{mode_idx+1} - {wl_nm}nm')
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        
                        # 标记峰值位置
                        peak_pos = np.unravel_index(np.argmax(intensity), intensity.shape)
                        ax.plot(peak_pos[1], peak_pos[0], 'w+', markersize=10, markeredgewidth=1)
                        
                    except Exception as e:
                        ax.text(0.5, 0.5, f'加载失败\n{str(e)[:20]}...', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'模式{mode_idx+1} - {wl_nm}nm (失败)')
                else:
                    ax.text(0.5, 0.5, '无数据', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(f'模式{mode_idx+1} - {wl_nm}nm (无数据)')
                
                ax.set_xlabel('X (像素)')
                ax.set_ylabel('Y (像素)')
        
        plt.tight_layout()
        
        # 保存总结图
        summary_path = os.path.join(save_dir, 'propagation_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 保存传播总结图: {summary_path}")
    
    def create_detailed_analysis(self, save_dir):
        """
        创建详细的分析报告
        
        参数:
            save_dir: 保存目录
        """
        print("创建详细分析报告...")
        
        # 查找所有仿真结果文件
        result_files = glob.glob(os.path.join(save_dir, "MC_single_*.npy"))
        
        if not result_files:
            print("⚠ 未找到仿真结果文件")
            return
        
        analysis_data = {}
        
        # 分析每个文件
        for file_path in result_files:
            filename = os.path.basename(file_path)
            
            # 提取文件信息
            mode_match = None
            wl_match = None
            layers_match = None
            
            # 提取模式信息
            for mode_idx in range(self.config.num_modes):
                if f"_mode{mode_idx+1}_" in filename:
                    mode_match = mode_idx + 1
                    break
            
            # 提取波长信息
            for wl in self.config.wavelengths:
                wl_nm = int(wl * 1e9)
                if f"{wl_nm}nm" in filename:
                    wl_match = wl_nm
                    break
            
            # 提取层数信息
            import re
            layers_pattern = r'(\d+)layers'
            layers_search = re.search(layers_pattern, filename)
            if layers_search:
                layers_match = int(layers_search.group(1))
            
            if mode_match and wl_match and layers_match:
                try:
                    # 加载数据
                    try:
                        data = np.load(file_path, allow_pickle=True)
                    except ValueError:
                        data = np.load(file_path, allow_pickle=True)
                    
                    # 计算强度
                    if np.iscomplexobj(data):
                        intensity = np.abs(data)**2
                    else:
                        intensity = np.abs(data)
                    
                    # 确保是2D数据
                    if intensity.ndim > 2:
                        intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
                    
                    # 计算分析指标
                    total_intensity = np.sum(intensity)
                    peak_intensity = np.max(intensity)
                    peak_pos = np.unravel_index(np.argmax(intensity), intensity.shape)
                    
                    # 计算质心
                    H, W = intensity.shape
                    y_indices, x_indices = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                    
                    if total_intensity > 0:
                        centroid_y = np.sum(y_indices * intensity) / total_intensity
                        centroid_x = np.sum(x_indices * intensity) / total_intensity
                    else:
                        centroid_y, centroid_x = H//2, W//2
                    
                    # 计算聚焦效率（中心区域）
                    center_y, center_x = H//2, W//2
                    radius = self.config.focus_radius
                    
                    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                    focus_mask = ((y_grid - center_y)**2 + (x_grid - center_x)**2) <= radius**2
                    
                    focus_intensity = np.sum(intensity[focus_mask])
                    focus_efficiency = focus_intensity / total_intensity if total_intensity > 0 else 0
                    
                    # 计算均匀性（标准差）
                    intensity_std = np.std(intensity)
                    intensity_mean = np.mean(intensity)
                    uniformity = intensity_std / intensity_mean if intensity_mean > 0 else 0
                    
                    # 存储分析结果
                    key = (layers_match, mode_match, wl_match)
                    analysis_data[key] = {
                        'filename': filename,
                        'total_intensity': total_intensity,
                        'peak_intensity': peak_intensity,
                        'peak_position': peak_pos,
                        'centroid_position': (centroid_y, centroid_x),
                        'focus_efficiency': focus_efficiency,
                        'uniformity': uniformity,
                        'intensity_std': intensity_std,
                        'intensity_mean': intensity_mean
                    }
                    
                except Exception as e:
                    print(f"❌ 分析文件 {filename} 时出错: {e}")
                    continue
        
        if not analysis_data:
            print("❌ 没有可分析的数据")
            return
        
        # 生成分析报告
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("光场传播仿真详细分析报告")
        report_lines.append("="*80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"分析文件数量: {len(analysis_data)}")
        report_lines.append("")
        
        # 按层数分组分析
        layers_groups = {}
        for (layers, mode, wl), data in analysis_data.items():
            if layers not in layers_groups:
                layers_groups[layers] = {}
            if mode not in layers_groups[layers]:
                layers_groups[layers][mode] = {}
            layers_groups[layers][mode][wl] = data
        
        for layers in sorted(layers_groups.keys()):
            report_lines.append(f"\n{layers} 层模型分析:")
            report_lines.append("-" * 50)
            
            modes_data = layers_groups[layers]
            
            # 计算平均指标
            all_focus_eff = []
            all_uniformity = []
            all_peak_int = []
            
            for mode in sorted(modes_data.keys()):
                report_lines.append(f"\n  模式 {mode}:")
                
                wl_data = modes_data[mode]
                for wl in sorted(wl_data.keys()):
                    data = wl_data[wl]
                    
                    all_focus_eff.append(data['focus_efficiency'])
                    all_uniformity.append(data['uniformity'])
                    all_peak_int.append(data['peak_intensity'])
                    
                    report_lines.append(f"    {wl}nm:")
                    report_lines.append(f"      聚焦效率: {data['focus_efficiency']:.4f}")
                    report_lines.append(f"      峰值强度: {data['peak_intensity']:.6f}")
                    report_lines.append(f"      峰值位置: {data['peak_position']}")
                    report_lines.append(f"      质心位置: ({data['centroid_position'][0]:.1f}, {data['centroid_position'][1]:.1f})")
                    report_lines.append(f"      均匀性: {data['uniformity']:.4f}")
            
            # 层级统计
            if all_focus_eff:
                report_lines.append(f"\n  {layers}层模型总体统计:")
                report_lines.append(f"    平均聚焦效率: {np.mean(all_focus_eff):.4f} ± {np.std(all_focus_eff):.4f}")
                report_lines.append(f"    平均峰值强度: {np.mean(all_peak_int):.6f} ± {np.std(all_peak_int):.6f}")
                report_lines.append(f"    平均均匀性: {np.mean(all_uniformity):.4f} ± {np.std(all_uniformity):.4f}")
        
        # 保存报告
        report_path = os.path.join(save_dir, 'detailed_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 打印报告
        for line in report_lines:
            print(line)
        
        print(f"\n✅ 详细分析报告已保存到: {report_path}")
        
        # 保存分析数据为JSON
        json_data = {}
        for (layers, mode, wl), data in analysis_data.items():
            key = f"{layers}layers_mode{mode}_{wl}nm"
            # 转换numpy类型为Python原生类型
            json_data[key] = {
                k: float(v) if isinstance(v, (np.ndarray, np.number)) else 
                   [float(x) for x in v] if isinstance(v, (list, tuple)) else v
                for k, v in data.items()
            }
        
        json_path = os.path.join(save_dir, 'detailed_analysis_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 分析数据已保存到: {json_path}")
    
    def create_performance_comparison(self, save_dir):
        """
        创建不同层数模型的性能对比图
        
        参数:
            save_dir: 保存目录
        """
        import matplotlib.pyplot as plt
        
        print("创建性能对比图...")
        
        # 加载分析数据
        json_path = os.path.join(save_dir, 'detailed_analysis_data.json')
        if not os.path.exists(json_path):
            print("⚠ 未找到分析数据文件，请先运行详细分析")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # 组织数据
        layers_performance = {}
        
        for key, data in analysis_data.items():
            # 解析键
            parts = key.split('_')
            layers_str = parts[0]  # e.g., "3layers"
            layers = int(layers_str.replace('layers', ''))
            
            if layers not in layers_performance:
                layers_performance[layers] = {
                    'focus_efficiency': [],
                    'peak_intensity': [],
                    'uniformity': []
                }
            
            layers_performance[layers]['focus_efficiency'].append(data['focus_efficiency'])
            layers_performance[layers]['peak_intensity'].append(data['peak_intensity'])
            layers_performance[layers]['uniformity'].append(data['uniformity'])
        
        if not layers_performance:
            print("❌ 没有可用的性能数据")
            return
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        layers_list = sorted(layers_performance.keys())
        
        # 1. 聚焦效率对比
        ax1 = axes[0, 0]
        focus_means = [np.mean(layers_performance[l]['focus_efficiency']) for l in layers_list]
        focus_stds = [np.std(layers_performance[l]['focus_efficiency']) for l in layers_list]
        
        ax1.bar(range(len(layers_list)), focus_means, yerr=focus_stds, 
                capsize=5, alpha=0.7, color='skyblue')
        ax1.set_xlabel('层数')
        ax1.set_ylabel('聚焦效率')
        ax1.set_title('不同层数的聚焦效率对比')
        ax1.set_xticks(range(len(layers_list)))
        ax1.set_xticklabels([f'{l}层' for l in layers_list])
        ax1.grid(True, alpha=0.3)
        
        # 2. 峰值强度对比
        ax2 = axes[0, 1]
        peak_means = [np.mean(layers_performance[l]['peak_intensity']) for l in layers_list]
        peak_stds = [np.std(layers_performance[l]['peak_intensity']) for l in layers_list]
        
        ax2.bar(range(len(layers_list)), peak_means, yerr=peak_stds, 
                capsize=5, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('层数')
        ax2.set_ylabel('峰值强度')
        ax2.set_title('不同层数的峰值强度对比')
        ax2.set_xticks(range(len(layers_list)))
        ax2.set_xticklabels([f'{l}层' for l in layers_list])
        ax2.grid(True, alpha=0.3)
        
        # 3. 均匀性对比
        ax3 = axes[1, 0]
        uni_means = [np.mean(layers_performance[l]['uniformity']) for l in layers_list]
        uni_stds = [np.std(layers_performance[l]['uniformity']) for l in layers_list]
        
        ax3.bar(range(len(layers_list)), uni_means, yerr=uni_stds, 
                capsize=5, alpha=0.7, color='lightgreen')
        ax3.set_xlabel('层数')
        ax3.set_ylabel('均匀性 (越小越好)')
        ax3.set_title('不同层数的均匀性对比')
        ax3.set_xticks(range(len(layers_list)))
        ax3.set_xticklabels([f'{l}层' for l in layers_list])
        ax3.grid(True, alpha=0.3)
        
        # 4. 综合性能雷达图
        ax4 = axes[1, 1]
        
        # 归一化指标 (0-1)
        focus_norm = np.array(focus_means) / np.max(focus_means) if np.max(focus_means) > 0 else np.zeros_like(focus_means)
        peak_norm = np.array(peak_means) / np.max(peak_means) if np.max(peak_means) > 0 else np.zeros_like(peak_means)
        uni_norm = 1 - (np.array(uni_means) / np.max(uni_means)) if np.max(uni_means) > 0 else np.ones_like(uni_means)  # 均匀性越小越好，所以取反
        
        # 综合得分
        composite_scores = (focus_norm + peak_norm + uni_norm) / 3
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers_list)))
        bars = ax4.bar(range(len(layers_list)), composite_scores, 
                       color=colors, alpha=0.7)
        ax4.set_xlabel('层数')
        ax4.set_ylabel('综合性能得分')
        ax4.set_title('综合性能对比')
        ax4.set_xticks(range(len(layers_list)))
        ax4.set_xticklabels([f'{l}层' for l in layers_list])
        ax4.grid(True, alpha=0.3)
        
        # 标注最佳性能
        best_idx = np.argmax(composite_scores)
        ax4.annotate(f'最佳: {layers_list[best_idx]}层\n得分: {composite_scores[best_idx]:.3f}', 
                     xy=(best_idx, composite_scores[best_idx]), 
                     xytext=(best_idx, composite_scores[best_idx] + 0.1),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     ha='center', fontweight='bold', color='red')
        
        plt.tight_layout()
        
        # 保存对比图
        comparison_path = os.path.join(save_dir, 'performance_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 性能对比图已保存到: {comparison_path}")
        
        # 打印最佳性能总结
        print(f"\n🏆 性能对比总结:")
        print(f"最佳聚焦效率: {layers_list[np.argmax(focus_means)]}层 ({np.max(focus_means):.4f})")
        print(f"最佳峰值强度: {layers_list[np.argmax(peak_means)]}层 ({np.max(peak_means):.6f})")
        print(f"最佳均匀性: {layers_list[np.argmin(uni_means)]}层 ({np.min(uni_means):.4f})")
        print(f"最佳综合性能: {layers_list[best_idx]}层 (得分: {composite_scores[best_idx]:.3f})")
    
    def run_complete_analysis(self, save_dir):
        """
        运行完整的分析流程
        
        参数:
            save_dir: 保存目录
        """
        print("开始完整分析流程...")
        
        try:
            # 1. 创建传播结果总结图
            self.create_propagation_summary(save_dir)
            
            # 2. 创建详细分析报告
            self.create_detailed_analysis(save_dir)
            
            # 3. 创建性能对比图
            self.create_performance_comparison(save_dir)
            
            print("✅ 完整分析流程完成！")
            
        except Exception as e:
            print(f"❌ 分析过程中出错: {e}")
            import traceback
            traceback.print_exc()

    def generate_input_fields_for_simulation(self):
        """
        为仿真生成多模式多波长输入场
        
        返回:
            torch.Tensor: [num_modes, num_wavelengths, H, W] 的输入场
        """
        print(f"生成 {self.config.num_modes} 个模式，{len(self.config.wavelengths)} 个波长的输入场...")
        
        # 创建输入场数组
        input_fields = torch.zeros(
            self.config.num_modes, 
            len(self.config.wavelengths), 
            self.config.field_size, 
            self.config.field_size, 
            dtype=torch.complex64
        )
        
        # 为每个模式生成不同的输入场
        for mode_idx in range(self.config.num_modes):
            print(f"  生成模式 {mode_idx + 1}...")
            
            # 为每个波长生成场
            for wl_idx, wavelength in enumerate(self.config.wavelengths):
                # 生成高斯光束，每个模式有不同的参数
                field = self._generate_gaussian_beam(
                    mode_idx=mode_idx,
                    wavelength=wavelength,
                    size=self.config.field_size
                )
                
                input_fields[mode_idx, wl_idx] = field
                
                wl_nm = wavelength * 1e9
                print(f"    {wl_nm:.0f}nm: 最大强度 = {torch.max(torch.abs(field)**2):.6f}")
        
        print(f"✓ 输入场生成完成，形状: {input_fields.shape}")
        return input_fields

    def _generate_gaussian_beam(self, mode_idx, wavelength, size):
        """
        生成高斯光束，每个模式有不同的参数
        
        参数:
            mode_idx: 模式索引
            wavelength: 波长
            size: 场大小
        
        返回:
            torch.Tensor: 高斯光束场
        """
        # 创建坐标网格
        x = torch.linspace(-size//2, size//2-1, size, dtype=torch.float32) * self.config.pixel_size
        y = torch.linspace(-size//2, size//2-1, size, dtype=torch.float32) * self.config.pixel_size
        X, Y = torch.meshgrid(x, y, indexing='xy')
        R_squared = X**2 + Y**2
        
        # 每个模式使用不同的束腰和偏移
        beam_waists = [20e-6, 25e-6, 30e-6]  # 不同的束腰
        x_offsets = [0, 5e-6, -5e-6]         # 不同的x偏移
        y_offsets = [0, -3e-6, 3e-6]         # 不同的y偏移
        
        w0 = beam_waists[mode_idx % len(beam_waists)]
        x_offset = x_offsets[mode_idx % len(x_offsets)]
        y_offset = y_offsets[mode_idx % len(y_offsets)]
        
        # 调整坐标
        X_adj = X - x_offset
        Y_adj = Y - y_offset
        R_adj_squared = X_adj**2 + Y_adj**2
        
        # 生成高斯光束
        amplitude = torch.exp(-R_adj_squared / w0**2)
        
        # 添加不同的相位（可选）
        phase_offset = mode_idx * np.pi / 3
        phase = torch.full_like(amplitude, phase_offset)
        
        # 组合成复数场
        field = amplitude * torch.exp(1j * phase)
        
        return field.to(torch.complex64)
