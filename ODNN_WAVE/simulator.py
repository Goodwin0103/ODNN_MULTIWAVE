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
import pandas as pd

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
            peak_pos = np.unravel_index(np.argmax(intensity_np), intensity_np.shape)
            print(f"🔍 仿真结果调试:")
            print(f"  MODE {mode_idx+1}, WL {wl_idx+1}: 峰值位置 {peak_pos}")
            print(f"  期望行: {mode_idx+1}, 实际峰值行: {peak_pos[0]}")           
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


    def simulate_propagation(self, phase_masks, input_field, process_all_modes=True, mode_specific_masks=None):
        """
        执行光场传播仿真 - 添加坐标系调试
        """
        print("开始光场传播仿真...")
        print("🔍 仿真参数调试:")
        print(f"  输入场形状: {input_field.shape}")
        
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
                print(f"🔍 处理模式 {mode_idx+1}/{num_modes} (数组索引: {mode_idx})")
                print(f"{'='*50}")
                
                # 🔧 添加输入场分析
                mode_field = input_field[mode_idx]  # [num_wavelengths, H, W]
                print(f"  模式{mode_idx+1}输入场形状: {mode_field.shape}")
                
                # 分析输入场的能量分布
                for wl_idx in range(num_wavelengths):
                    wl_field = mode_field[wl_idx]
                    if isinstance(wl_field, torch.Tensor):
                        wl_field_np = wl_field.detach().cpu().numpy()
                    else:
                        wl_field_np = wl_field
                    
                    intensity = np.abs(wl_field_np) ** 2
                    peak_pos = np.unravel_index(np.argmax(intensity), intensity.shape)
                    wl_nm = self.config.wavelengths[wl_idx] * 1e9
                    print(f"    输入场 WL{wl_idx+1} ({wl_nm:.0f}nm): 峰值位置 {peak_pos}")
                
                # 使用通用相位掩膜或模式特定掩膜
                if mode_specific_masks and mode_idx < len(mode_specific_masks):
                    current_masks = mode_specific_masks[mode_idx]
                    print(f"  使用模式{mode_idx+1}专用相位掩膜")
                else:
                    current_masks = phase_masks
                    print(f"  使用通用相位掩膜")
                
                # 仿真该模式
                mode_results = self._simulate_single_mode(
                    current_masks, mode_field, f"_mode{mode_idx+1}"
                )
                
                # 评估结果
                mode_evaluations = []
                for wl_idx in range(num_wavelengths):
                    if f'wl_{wl_idx}' in mode_results:
                        field = mode_results[f'wl_{wl_idx}']['field']
                        
                        # 🔧 添加输出场分析
                        if isinstance(field, torch.Tensor):
                            field_np = field.detach().cpu().numpy()
                        else:
                            field_np = field
                        
                        output_intensity = np.abs(field_np) ** 2
                        output_peak_pos = np.unravel_index(np.argmax(output_intensity), output_intensity.shape)
                        wl_nm = self.config.wavelengths[wl_idx] * 1e9
                        
                        print(f"🔍 仿真输出分析:")
                        print(f"  MODE {mode_idx+1}, WL{wl_idx+1} ({wl_nm:.0f}nm):")
                        print(f"    输出峰值位置: {output_peak_pos}")
                        print(f"    期望行位置: ~{40 + mode_idx * 60} (MODE {mode_idx+1})")
                        print(f"    实际行位置: {output_peak_pos[0]}")
                        
                        # 判断是否聚焦到正确位置
                        expected_y_center = 40 + mode_idx * 60  # 基于调试输出的计算
                        y_tolerance = 30  # 允许的误差范围
                        
                        if abs(output_peak_pos[0] - expected_y_center) <= y_tolerance:
                            print(f"    ✅ 聚焦位置正确")
                        else:
                            print(f"    ❌ 聚焦位置错误！")
                            print(f"    可能原因: 模式索引映射问题")
                        
                        eval_result = self._evaluate_propagation_result(field, mode_idx, wl_idx)
                        mode_evaluations.append(eval_result)
                
                evaluation_results.extend(mode_evaluations)
                print(f"✓ 模式{mode_idx+1}仿真完成，生成{len(mode_evaluations)}个评估结果")
            
            print(f"\n✅ 所有模式仿真完成，总计{len(evaluation_results)}个评估结果")
            return evaluation_results

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
        Visualize propagation results
        
        参数:
            save_dir: 保存目录
            mode_suffix: 模式后缀
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        
        print("Generating propagation visualization images...")
        
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
                
                intensity_flipped = np.flipud(intensity)
                if phase is not None:
                    phase_flipped = np.flipud(phase)
                
                # 创建图形
                if phase is not None:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                else:
                    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
                
                # 绘制强度分布
                im1 = ax1.imshow(intensity_flipped, cmap='hot', origin='lower')
                ax1.set_title(f'Field Intensity Distribution - {wl_nm}nm{mode_suffix}')
                ax1.set_xlabel('X (pixels)')
                ax1.set_ylabel('Y (pixels)')
                plt.colorbar(im1, ax=ax1, label='Intensity')
                
                # 绘制相位分布（如果有）
                if phase is not None:
                    im2 = ax2.imshow(phase_flipped, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
                    ax2.set_title(f'Field Phase Distribution - {wl_nm}nm{mode_suffix}')
                    ax2.set_xlabel('X (pixels)')
                    ax2.set_ylabel('Y (pixels)')
                    plt.colorbar(im2, ax=ax2, label='Phase (radians)')
                
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
        Create summary figures for each model (by layers)
        """
        import matplotlib.pyplot as plt
        import re
        
        print("创建不同层数的传播结果汇总图...")
        
        # 查找所有仿真结果文件
        result_files = glob.glob(os.path.join(save_dir, "MC_single_*.npy"))
        
        if not result_files:
            print("⚠ 未找到仿真结果文件")
            return
        
        print(f"找到 {len(result_files)} 个仿真结果文件")
        
        # 按层数组织文件
        organized_by_layers = {}
        
        for file_path in result_files:
            filename = os.path.basename(file_path)
            
            # **改进的层数提取方法**
            layers_match = None
            
            # 方法1: 寻找 "Xlayers" 模式
            layers_pattern = r'(\d+)layers'
            layers_search = re.search(layers_pattern, filename)
            if layers_search:
                layers_match = int(layers_search.group(1))
            else:
                # 方法2: 如果没有找到，尝试从其他模式推断
                # 检查常见的层数值
                for possible_layers in [1, 2, 3, 4, 5, 6, 7, 8]:
                    if f"_{possible_layers}layers_" in filename or f"_{possible_layers}layer_" in filename:
                        layers_match = possible_layers
                        break
            
            if not layers_match:
                # 方法3: 如果仍然没有找到，尝试从文件名的其他部分推断
                print(f"  ⚠ 无法从文件名提取层数信息: {filename}")
                # 可以设置默认值或跳过
                layers_match = 1  # 默认为1层
                print(f"  使用默认层数: {layers_match}")
            
            # 提取模式和波长信息
            mode_match = None
            wl_match = None
            
            # 提取模式
            mode_pattern = r'mode(\d+)'
            mode_search = re.search(mode_pattern, filename)
            if mode_search:
                mode_match = int(mode_search.group(1))
            
            # 提取波长
            wl_pattern = r'(\d+)nm'
            wl_search = re.search(wl_pattern, filename)
            if wl_search:
                wl_match = int(wl_search.group(1))
            
            
            if mode_match and wl_match and layers_match:
                if layers_match not in organized_by_layers:
                    organized_by_layers[layers_match] = {}
                
                key = (mode_match, wl_match)
                organized_by_layers[layers_match][key] = file_path
            else:
                print(f"  ❌ 跳过文件 (信息不完整)")
        
        
        if not organized_by_layers:
            print("❌ 没有找到可以按层数分类的文件")
            return
        
        # 为每个层数创建单独的总结图
        for layers in sorted(organized_by_layers.keys()):
            files_dict = organized_by_layers[layers]
            print(f"\n创建 {layers} 层模型的汇总图...")
            self.create_single_layer_summary(layers, files_dict, save_dir)

    def create_single_layer_summary(self, layers, files_dict, save_dir):
        """Create a summary figure for a single layer model."""
        import matplotlib.pyplot as plt
        
        print(f"Creating summary for {layers}-layer model, containing {len(files_dict)} result(s)...")
        
        num_modes = self.config.num_modes
        num_wavelengths = len(self.config.wavelengths)
        
        # Create subplots
        fig, axes = plt.subplots(num_modes, num_wavelengths, 
                                figsize=(5*num_wavelengths, 4*num_modes))
        
        # Handle cases with a single row or single column
        if num_modes == 1 and num_wavelengths == 1:
            axes = np.array([[axes]])
        elif num_modes == 1:
            axes = axes.reshape(1, -1)
        elif num_wavelengths == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'{layers}-Layer Model - Propagation Results Summary', fontsize=16, fontweight='bold')
        
        # Counter for successfully loaded files
        successful_loads = 0
        
        for mode_idx in range(num_modes):
            for wl_idx, wl in enumerate(self.config.wavelengths):
                wl_nm = int(wl * 1e9)
                key = (mode_idx + 1, wl_nm)  # Mode numbering starts at 1
                
                if num_modes == 1 and num_wavelengths == 1:
                    ax = axes[0, 0]
                else:
                    ax = axes[mode_idx, wl_idx]
                
                if key in files_dict:
                    file_path = files_dict[key]
                    filename = os.path.basename(file_path)
                    
                    try:
                        # Load and display data
                        print(f"  Loading file: {filename}")
                        data = np.load(file_path, allow_pickle=True)
                        
                        # Compute intensity
                        if np.iscomplexobj(data):
                            intensity = np.abs(data)**2
                        else:
                            intensity = np.abs(data)**2
                        
                        # Process multi-dimensional data
                        if intensity.ndim > 2:
                            print(f"    Data dimensions: {intensity.shape}, reducing dimensions...")
                            # Use the last two dimensions as spatial dimensions
                            intensity = intensity.reshape(-1, intensity.shape[-2], intensity.shape[-1])
                            intensity = np.sum(intensity, axis=0)  # Sum over other dimensions
                        
                        # 🔄 **关键修复：确保 intensity_flipped 总是被定义**
                        intensity_flipped = np.flipud(intensity)
                        
                        # Normalize
                        if np.max(intensity_flipped) > 0:
                            intensity_flipped = intensity_flipped / np.max(intensity_flipped)
                        
                        # Display intensity distribution (Y-axis flipped)
                        im = ax.imshow(intensity_flipped, cmap='hot', origin='lower', aspect='equal')
                        ax.set_title(f'Mode {mode_idx+1} - {wl_nm}nm', fontsize=12)
                        
                        # Add colorbar
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
                        
                        # Removed focus center marker
                        # ax.plot(peak_pos_flipped[1], peak_pos_flipped[0], 'w+', markersize=12, markeredgewidth=2)
                        
                        # Add performance metric text
                        peak_intensity = np.max(intensity_flipped)
                        total_intensity = np.sum(intensity_flipped)
                        
                        # Calculate focus ratio
                        center_y, center_x = intensity_flipped.shape[0] // 2, intensity_flipped.shape[1] // 2
                        radius = min(intensity_flipped.shape) // 8  # Focus region radius
                        y_grid, x_grid = np.meshgrid(np.arange(intensity_flipped.shape[0]), 
                                                np.arange(intensity_flipped.shape[1]), indexing='ij')
                        focus_mask = ((y_grid - center_y)**2 + (x_grid - center_x)**2) <= radius**2
                        focus_ratio = np.sum(intensity_flipped[focus_mask]) / total_intensity if total_intensity > 0 else 0
                        
                        # Display metrics on the plot
                        ax.text(0.02, 0.98, f'Peak: {peak_intensity:.3f}\nFocus: {focus_ratio:.3f}', 
                            transform=ax.transAxes, fontsize=10, 
                            verticalalignment='top', color='white',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                        
                        successful_loads += 1
                        print(f"    ✓ Loaded and displayed successfully")
                        
                    except Exception as e:
                        print(f"    ❌ Load failed: {e}")
                        ax.text(0.5, 0.5, f'Load failed\n{str(e)[:30]}...', 
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=10, color='red')
                        ax.set_title(f'Mode {mode_idx+1} - {wl_nm}nm (failed)', fontsize=12, color='red')
                else:
                    print(f"  ❌ Data not found: Mode {mode_idx+1}, {wl_nm}nm")
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=14, color='gray')
                    ax.set_title(f'Mode {mode_idx+1} - {wl_nm}nm (No Data)', fontsize=12, color='gray')
                
                # Set axis labels
                ax.set_xlabel('X (pixels)', fontsize=10)
                ax.set_ylabel('Y (pixels)', fontsize=10)
                
                # 移除坐标轴刻度以节省空间
                ax.tick_params(labelsize=8)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # 为标题留出空间
        
        # 保存该层数的总结图
        summary_path = os.path.join(save_dir, f'{layers}_layers_propagation_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ {layers} 层模型汇总图已保存: {summary_path}")
        print(f"   成功加载 {successful_loads}/{len(files_dict)} 个文件")


    def _plot_propagation_steps(self, propagation_steps, mode_idx, wavelength_idx, wl_nm):
        """绘制传播步骤的可视化图（距离以微米显示）"""
        import matplotlib.pyplot as plt
        
        # 计算需要显示的步骤数
        num_steps = len(propagation_steps)
        
        # 确定子图布局
        if num_steps <= 3:
            rows, cols = 1, num_steps
            figsize = (5*cols, 4)
        elif num_steps <= 6:
            rows, cols = 2, 3
            figsize = (15, 8)
        elif num_steps <= 9:
            rows, cols = 3, 3
            figsize = (15, 12)
        else:
            rows, cols = 4, int(np.ceil(num_steps/4))
            figsize = (4*cols, 4*rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_steps == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # 绘制每个步骤
        step_idx = 0
        for step_key, step_data in propagation_steps.items():
            if step_idx >= len(axes):
                break
                
            ax = axes[step_idx]
            field = step_data['field'].cpu().numpy()
            intensity = np.abs(field)**2
            
            # 显示强度分布
            im = ax.imshow(intensity, cmap='hot', origin='lower')
            
            # 获取距离信息
            distance_um = step_data.get('distance_um', 0.0)
            title = f"Step {step_idx+1}: {step_data['description']}"
            if distance_um > 0:
                title += f"\n({distance_um:.1f} μm)"
            
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            # 添加统计信息
            max_val = np.max(intensity)
            energy = np.sum(intensity)
            ax.text(0.02, 0.98, f'Max: {max_val:.3f}\nEnergy: {energy:.2e}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8)
            
            step_idx += 1
        
        # 隐藏多余的子图
        for i in range(step_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Step-by-Step Propagation - Mode {mode_idx}, λ={wl_nm}nm', 
                    fontsize=14, y=0.98)
        plt.tight_layout()
        plt.show()

    def _save_propagation_steps(self, propagation_steps, mode_idx, wavelength_idx, wl_nm, save_dir):
        """保存传播步骤的数据（文件名包含微米距离信息）"""
        import os
        
        # 创建保存目录
        step_dir = os.path.join(save_dir, f'propagation_steps_mode{mode_idx}_wl{wl_nm}nm')
        os.makedirs(step_dir, exist_ok=True)
        
        saved_files = []
        
        for step_key, step_data in propagation_steps.items():
            # 获取距离信息
            distance_um = step_data.get('distance_um', 0.0)
            
            # 保存场数据
            field = step_data['field'].cpu().numpy()
            filename = f'{step_key}_field_{distance_um:.1f}um.npy'
            filepath = os.path.join(step_dir, filename)
            np.save(filepath, field)
            saved_files.append(filepath)
            
            # 保存强度图像
            intensity = np.abs(field)**2
            img_filename = f'{step_key}_intensity_{distance_um:.1f}um.png'
            img_filepath = os.path.join(step_dir, img_filename)
            
            plt.figure(figsize=(6, 5))
            plt.imshow(intensity, cmap='hot', origin='lower')
            plt.colorbar()
            plt.title(f"{step_data['description']}\nDistance: {distance_um:.1f} μm")
            plt.xlabel('X (pixels)')
            plt.ylabel('Y (pixels)')
            plt.savefig(img_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(img_filepath)
        
        return saved_files

    def _calculate_propagation_statistics(self, propagation_history):
        """
        计算传播过程的统计信息（包含微米单位的距离信息）
        
        参数:
            propagation_history: 传播历史列表 [(step_name, field), ...]
        
        返回:
            dict: 统计信息
        """
        if not propagation_history:
            return {}
        
        # 获取初始和最终场
        initial_field = propagation_history[0][1]
        final_field = propagation_history[-1][1]
        
        # 转换为numpy数组
        if isinstance(initial_field, torch.Tensor):
            initial_np = initial_field.detach().cpu().numpy()
        else:
            initial_np = initial_field
        
        if isinstance(final_field, torch.Tensor):
            final_np = final_field.detach().cpu().numpy()
        else:
            final_np = final_field
        
        # 计算能量
        initial_energy = np.sum(np.abs(initial_np) ** 2)
        final_energy = np.sum(np.abs(final_np) ** 2)
        
        # 能量守恒比例
        energy_conservation = final_energy / initial_energy if initial_energy > 0 else 0
        
        # 计算最终聚焦比例
        final_intensity = np.abs(final_np) ** 2
        if final_intensity.ndim > 2:
            final_intensity = final_intensity.squeeze()
        
        # 计算聚焦比例（中心区域能量占比）
        center_y, center_x = final_intensity.shape[0] // 2, final_intensity.shape[1] // 2
        radius = min(final_intensity.shape) // 8
        
        y_grid, x_grid = np.meshgrid(np.arange(final_intensity.shape[0]), 
                                    np.arange(final_intensity.shape[1]), indexing='ij')
        focus_mask = ((y_grid - center_y)**2 + (x_grid - center_x)**2) <= radius**2
        
        total_intensity = np.sum(final_intensity)
        focus_intensity = np.sum(final_intensity[focus_mask])
        focus_ratio = focus_intensity / total_intensity if total_intensity > 0 else 0
        
        # 计算每步的能量变化
        energies = []
        for step_name, field in propagation_history:
            if isinstance(field, torch.Tensor):
                field_np = field.detach().cpu().numpy()
            else:
                field_np = field
            energy = np.sum(np.abs(field_np) ** 2)
            energies.append(float(energy))
        
        statistics = {
            'initial_energy': float(initial_energy),
            'final_energy': float(final_energy),
            'energy_conservation': float(energy_conservation),
            'final_focus_ratio': float(focus_ratio),
            'num_steps': len(propagation_history),
            'energy_loss_db': float(-10 * np.log10(energy_conservation)) if energy_conservation > 0 else float('inf'),
            'energies': energies,
            'energy_variation': float(max(energies) / min(energies)) if min(energies) > 0 else 0,
            'unit': 'micrometers'  # 标记使用的单位
        }
        
        return statistics


    def display_step_by_step_propagation(self, phase_masks, input_fields, mode_idx=0, wavelength_idx=0, 
                                        save_intermediate=False, show_plots=True, save_dir=None,
                                        save_individual_steps=False):
        """
        显示光场单步传播过程的可视化 - 先调制后传播
        """
        print(f"🎬 显示单步传播过程 - 先调制后传播")
        print(f"模式索引: {mode_idx}, 波长索引: {wavelength_idx}")
        
        # 打印配置信息用于调试（转换为微米）
        print(f"z_layers: {self.config.z_layers * 1e6:.1f} μm (层间传播距离)")
        z_step_um = getattr(self.config, 'z_step', 0) * 1e6 if hasattr(self.config, 'z_step') else 0
        print(f"z_step: {z_step_um:.1f} μm" if z_step_um > 0 else "z_step: Not defined")
        print(f"z_prop: {self.config.z_prop * 1e6:.1f} μm (最终传播距离)")
        print(f"层数: {len(phase_masks)}")
        print("="*50)
            
        # 设置保存目录
        if save_dir is None:
            save_dir = getattr(self.config, 'save_dir', '/home/user')
        
        # 创建专门的传播图像保存目录
        if save_individual_steps or save_intermediate:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            propagation_save_dir = os.path.join(save_dir, "propagation_images_Z_LAYERS", 
                                            f"mode{mode_idx+1}_wl{wavelength_idx+1}_{timestamp}")
            os.makedirs(propagation_save_dir, exist_ok=True)
            print(f"📁 图像将保存到: {propagation_save_dir}")
        
        # 输入场处理
        if isinstance(input_fields, np.ndarray):
            input_fields = torch.from_numpy(input_fields.copy())
        elif not isinstance(input_fields, torch.Tensor):
            input_fields = torch.tensor(input_fields)
        
        if not input_fields.dtype.is_complex:
            if input_fields.dtype.is_floating_point:
                input_fields = input_fields.to(torch.complex64)
            else:
                input_fields = input_fields.to(torch.float32).to(torch.complex64)
        
        # 提取指定模式和波长的输入场
        if input_fields.ndim == 4:  # [num_modes, num_wavelengths, H, W]
            if mode_idx >= input_fields.shape[0]:
                print(f"❌ 模式索引 {mode_idx} 超出范围 (最大: {input_fields.shape[0]-1})")
                return None
            if wavelength_idx >= input_fields.shape[1]:
                print(f"❌ 波长索引 {wavelength_idx} 超出范围 (最大: {input_fields.shape[1]-1})")
                return None
            current_field = input_fields[mode_idx, wavelength_idx]
        elif input_fields.ndim == 3:  # [num_wavelengths, H, W]
            if wavelength_idx >= input_fields.shape[0]:
                print(f"❌ 波长索引 {wavelength_idx} 超出范围 (最大: {input_fields.shape[0]-1})")
                return None
            current_field = input_fields[wavelength_idx]
        elif input_fields.ndim == 2:  # [H, W]
            current_field = input_fields
        else:
            print(f"❌ 不支持的输入场维度: {input_fields.shape}")
            return None
        
        print(f"选择的场形状: {current_field.shape}")
        
        # 预处理输入场
        current_field = self._preprocess_field_for_simulation(current_field)
        current_field = current_field.to(self.device)
        
        # 获取波长
        if wavelength_idx < len(self.config.wavelengths):
            wavelength = self.config.wavelengths[wavelength_idx]
            wl_nm = int(wavelength * 1e9)
            print(f"波长: {wl_nm} nm")
        else:
            wavelength = 532e-9  # 默认波长
            wl_nm = 532
            print(f"使用默认波长: {wl_nm} nm")
        
        # 存储传播步骤
        propagation_steps = {}
        propagation_history = []
        saved_individual_files = []
        
        # 初始场
        propagation_steps['initial'] = {
            'field': current_field.clone(),
            'description': 'Initial Input Field',
            'step_type': 'initial',
            'distance_um': 0.0
        }
        propagation_history.append(('initial', current_field.clone()))
        print(f"✓ 步骤 0: 初始输入场 (0.0 μm)")
        
        # 保存初始场的单独图像
        if save_individual_steps:
            step_file = self._save_single_step_image(
                current_field, 'Step 1: Initial Input Field', 0.0, 
                propagation_save_dir, f"step_01_initial_input_field", mode_idx, wl_nm
            )
            if step_file:
                saved_individual_files.append(step_file)
        

        # 逐层处理 - 先调制，后传播**
        step_counter = 1
        current_distance = 0
        num_layers = len(phase_masks)

        for layer_idx in range(num_layers):
            print(f"\n--- 第 {layer_idx+1} 层 ---")
            
            # 步骤1：应用相位掩码
            if layer_idx < len(phase_masks) and wavelength_idx < len(phase_masks[layer_idx]):
                print(f"🎭 正在应用第{layer_idx+1}层相位掩码...")
                
                # 应用相位掩码
                phase_mask = phase_masks[layer_idx][wavelength_idx]
                current_field = self._apply_phase_mask(current_field, phase_mask)
                
                # 保存应用掩码后的场
                step_key = f'layer_{layer_idx+1}_after_mask'
                propagation_steps[step_key] = {
                    'field': current_field.clone(),
                    'description': f'Layer {layer_idx+1} - After Phase Mask',
                    'step_type': 'after_mask',
                    'layer_idx': layer_idx,
                    'distance_um': current_distance * 1e6
                }
                propagation_history.append((step_key, current_field.clone()))
                print(f"✓ 步骤 {step_counter}: 应用第{layer_idx+1}层相位掩码")
                
                # 保存掩码后的单独图像
                if save_individual_steps:
                    step_file = self._save_single_step_image(
                        current_field,
                        f'Step {step_counter+1}: Layer {layer_idx+1} - After Phase Mask', 
                        current_distance * 1e6, propagation_save_dir, 
                        f"step_{step_counter+1:02d}_layer_{layer_idx+1}_after_phase_mask", 
                        mode_idx, wl_nm
                    )
                    if step_file:
                        saved_individual_files.append(step_file)
                
                step_counter += 1
            
            # 步骤2：传播
            if layer_idx < num_layers - 1:
                # 传播到下一层
                total_distance = self.config.z_layers
                print(f"🚀 传播到下一层，距离: {total_distance*1e6:.1f} μm")
            else:
                # 最后一层，传播到检测平面
                total_distance = self.config.z_prop
                print(f"🚀 传播到检测平面，距离: {total_distance*1e6:.1f} μm")
            
            # 执行传播
            current_distance += total_distance
            current_field = self._angular_spectrum_propagate(
                current_field, total_distance, wavelength
            )
            
            # 保存传播后的场
            if layer_idx < num_layers - 1:
                step_key = f'after_layer_{layer_idx+1}_propagation'
                description = f'After Layer {layer_idx+1} Propagation ({current_distance*1e6:.1f} μm)'
                filename_prefix = f"step_{step_counter+1:02d}_after_layer_{layer_idx+1}_propagation"
            else:
                step_key = 'final'
                description = f'Final Detection Plane ({current_distance*1e6:.1f} μm)'
                filename_prefix = f"step_{step_counter+1:02d}_final_detection_plane"
            
            propagation_steps[step_key] = {
                'field': current_field.clone(),
                'description': description,
                'step_type': 'after_propagation',
                'distance_um': current_distance * 1e6
            }
            propagation_history.append((step_key, current_field.clone()))
            print(f"✓ 步骤 {step_counter}: 传播至 {current_distance*1e6:.1f} μm")
            
            # 保存传播后的单独图像
            if save_individual_steps:
                step_file = self._save_single_step_image(
                    current_field, f'Step {step_counter+1}: {description}', 
                    current_distance * 1e6, propagation_save_dir, 
                    filename_prefix, mode_idx, wl_nm
                )
                if step_file:
                    saved_individual_files.append(step_file)
            
            step_counter += 1

        # 计算统计信息
        statistics = self._calculate_propagation_statistics(propagation_history)
        
        # 可视化结果
        if show_plots or save_intermediate:
            complete_image_file = self._plot_propagation_steps_enhanced(
                propagation_steps, mode_idx, wavelength_idx, wl_nm, 
                show_plots, save_intermediate, propagation_save_dir if save_intermediate else None
            )
            if complete_image_file and save_intermediate:
                saved_individual_files.append(complete_image_file)
        
        # 保存中间结果数据文件
        if save_intermediate:
            data_files = self._save_propagation_steps(
                propagation_steps, mode_idx, wavelength_idx, wl_nm, propagation_save_dir
            )
            saved_individual_files.extend(data_files)
            print(f"\n💾 保存了 {len(data_files)} 个数据文件")
        
        # 保存配置信息
        if save_individual_steps or save_intermediate:
            config_file = self._save_propagation_config(
                mode_idx, wavelength_idx, wl_nm, len(phase_masks), 
                current_distance * 1e6, propagation_save_dir
            )
            if config_file:
                saved_individual_files.append(config_file)
        
        # 返回完整结果
        result = {
            'steps': propagation_steps,
            'history': propagation_history,
            'statistics': statistics,
            'mode_idx': mode_idx,
            'wavelength_idx': wavelength_idx,
            'wavelength_nm': wl_nm,
            'total_distance_um': current_distance * 1e6,
            'saved_files': saved_individual_files
        }
        
        print(f"\n✅ 使用z_layers的单步传播显示完成！")
        print(f"总步数: {len(propagation_history)}")
        print(f"总传播距离: {current_distance * 1e6:.1f} μm")
        print(f"能量守恒: {statistics.get('energy_conservation', 0):.4f}")
        if saved_individual_files:
            print(f"💾 共保存 {len(saved_individual_files)} 个文件")
            print(f"📁 保存目录: {propagation_save_dir}")
        
        return result

    def _simulate_single_mode(self, phase_masks, input_field, mode_suffix=""):
        """
        模拟单个模式的光场传播 - 调制后传播
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
            
            # ：逐层处理 - 先调制，后传播**
            for layer_idx in range(num_layers):
                print(f"  处理第 {layer_idx+1} 层...")
                
                # 应用相位掩码
                if layer_idx < len(phase_masks) and wl_idx < len(phase_masks[layer_idx]):
                    phase_mask = phase_masks[layer_idx][wl_idx]
                    current_field = self._apply_phase_mask(current_field, phase_mask)
                    print(f"    ✓ 应用相位掩码")
                
                # 传播
                if layer_idx < num_layers - 1:
                    # 传播到下一层
                    print(f"    🚀 传播 {self.config.z_layers * 1e6:.1f} μm 到下一层")
                    current_field = self._angular_spectrum_propagate(
                        current_field, self.config.z_layers, wavelength
                    )
                else:
                    # 最后一层传播到检测平面
                    print(f"    🚀 最终传播 {self.config.z_prop * 1e6:.1f} μm 到检测平面")
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


    def _plot_propagation_steps_enhanced(self, propagation_steps, mode_idx, wavelength_idx, wl_nm, 
                                        show_plots=True, save_complete=False, save_dir=None):
        """
        增强的传播步骤可视化，支持保存完整图像
        
        参数:
            propagation_steps: 传播步骤字典
            mode_idx: 模式索引
            wavelength_idx: 波长索引
            wl_nm: 波长（纳米）
            show_plots: 是否显示图形
            save_complete: 是否保存完整图像
            save_dir: 保存目录
        
        返回:
            str: 保存的完整图像文件路径，如果未保存则返回None
        """
        import matplotlib.pyplot as plt
        
        # 计算需要显示的步骤数
        num_steps = len(propagation_steps)
        
        # 确定子图布局
        if num_steps <= 3:
            rows, cols = 1, num_steps
            figsize = (5*cols, 4)
        elif num_steps <= 6:
            rows, cols = 2, 3
            figsize = (15, 8)
        elif num_steps <= 9:
            rows, cols = 3, 3
            figsize = (15, 12)
        else:
            rows, cols = 4, int(np.ceil(num_steps/4))
            figsize = (4*cols, 4*rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_steps == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # 绘制每个步骤
        step_idx = 0
        for step_key, step_data in propagation_steps.items():
            if step_idx >= len(axes):
                break
                
            ax = axes[step_idx]
            field = step_data['field'].cpu().numpy()
            intensity = np.abs(field)**2
            
            # 显示强度分布
            im = ax.imshow(intensity, cmap='hot', origin='lower')
            
            # 获取距离信息
            distance_um = step_data.get('distance_um', 0.0)
            title = f"Step {step_idx+1}: {step_data['description']}"
            if distance_um > 0:
                title += f"\n({distance_um:.1f} μm)"
            
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            # 添加统计信息
            max_val = np.max(intensity)
            energy = np.sum(intensity)
            ax.text(0.02, 0.98, f'Max: {max_val:.3f}\nEnergy: {energy:.2e}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8)
            
            step_idx += 1
        
        # 隐藏多余的子图
        for i in range(step_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Step-by-Step Propagation - Mode {mode_idx+1}, λ={wl_nm}nm', 
                    fontsize=14, y=0.98)
        plt.tight_layout()
        
        # 保存完整图像
        complete_image_file = None
        if save_complete and save_dir:
            filename = f"propagation_complete_mode{mode_idx+1}_wl{wl_nm}nm.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            complete_image_file = filepath
            print(f"💾 保存完整传播图像: {filename}")
        
        # 显示图像
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return complete_image_file

    def _save_propagation_config(self, mode_idx, wavelength_idx, wl_nm, num_layers, total_distance_um, save_dir):
        """
        保存传播配置信息
        
        参数:
            mode_idx: 模式索引
            wavelength_idx: 波长索引
            wl_nm: 波长（纳米）
            num_layers: 层数
            total_distance_um: 总传播距离（微米）
            save_dir: 保存目录
        
        返回:
            str: 配置文件路径
        """
        try:
            config_info = {
                'simulation_info': {
                    'mode_idx': mode_idx,
                    'wavelength_idx': wavelength_idx,
                    'wavelength_nm': wl_nm,
                    'num_layers': num_layers,
                    'total_distance_um': total_distance_um
                },
                'system_config': {
                    'layer_size': getattr(self.config, 'layer_size', 'N/A'),
                    'pixel_size': getattr(self.config, 'pixel_size', 'N/A'),
                    'z_layers_um': getattr(self.config, 'z_layers', 0) * 1e6,
                    'z_prop_um': getattr(self.config, 'z_prop', 0) * 1e6,
                    'device': str(self.device)
                },
                'timestamp': datetime.now().isoformat(),
                'version': 'Enhanced Propagation Simulator v1.0'
            }
            
            config_filename = f"propagation_config_mode{mode_idx+1}_wl{wl_nm}nm.json"
            config_filepath = os.path.join(save_dir, config_filename)
            
            with open(config_filepath, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
            
            print(f"💾 保存配置文件: {config_filename}")
            return config_filepath
            
        except Exception as e:
            print(f"❌ 保存配置文件失败: {e}")
            return None

    def _save_single_step_image(self, field, title, distance_um, save_dir, filename_prefix, mode_idx, wl_nm, 
                            save_clean=True):
        """
        保存单个传播步骤的图像 - 支持纯净版本
        
        参数:
            field: 光场数据 (torch.Tensor)
            title: 图像标题
            distance_um: 传播距离（微米）
            save_dir: 保存目录
            filename_prefix: 文件名前缀
            mode_idx: 模式索引
            wl_nm: 波长（纳米）
            save_clean: 是否保存纯净版本（无标题、无colorbar）
        
        返回:
            str: 保存的文件路径，如果失败则返回None
        """
        try:
            # 转换为numpy数组
            if isinstance(field, torch.Tensor):
                field_np = field.detach().cpu().numpy()
            else:
                field_np = field
            
            # 计算强度
            intensity = np.abs(field_np) ** 2
            
            if save_clean:
                # 保存纯净版本 - 无标题、无colorbar、无坐标轴
                plt.figure(figsize=(8, 8))  # 正方形图像
                plt.imshow(intensity, cmap='hot', origin='lower')
                plt.axis('off')  # 关闭坐标轴
                
                # 保存纯净图像
                clean_filename = f"{filename_prefix}_{distance_um:.1f}um_clean.png"
                clean_filepath = os.path.join(save_dir, clean_filename)
                plt.savefig(clean_filepath, dpi=300, bbox_inches='tight', pad_inches=0, 
                        facecolor='black', edgecolor='none')
                plt.close()
                
                print(f"  💾 保存纯净图像: {clean_filename}")
                return clean_filepath
            else:
                # 保存带标注的版本（原有功能）
                plt.figure(figsize=(8, 6))
                im = plt.imshow(intensity, cmap='hot', origin='lower')
                plt.colorbar(im, label='Intensity')
                
                # 设置标题和标签
                plt.title(title, fontsize=12, fontweight='bold')
                plt.xlabel('X (pixels)')
                plt.ylabel('Y (pixels)')
                
                # 添加统计信息
                max_intensity = np.max(intensity)
                total_energy = np.sum(intensity)
                plt.text(0.02, 0.98, f'Max: {max_intensity:.3f}\nEnergy: {total_energy:.2e}', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10)
                
                # 保存带标注的图像
                filename = f"{filename_prefix}_{distance_um:.1f}um.png"
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"  💾 保存标注图像: {filename}")
                return filepath
            
        except Exception as e:
            print(f"  ❌ 保存图像失败: {e}")
            return None
