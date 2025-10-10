# diagnosis.py
# -*- coding: utf-8 -*-
"""
检查仿真光场与标签位置匹配度的诊断工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import os

class FocusMatchDiagnostics:
    """光场-标签匹配度诊断工具"""
    
    def __init__(self, config):
        self.config = config
    
    def check_focus_match(self, simulated_fields, labels, evaluation_regions, model_name=""):
        """
        检查仿真光场与标签的匹配度
        
        Args:
            simulated_fields: 仿真得到的光场 [num_modes, num_wavelengths, H, W]
            labels: 标签 [num_modes, num_wavelengths, H, W] 
            evaluation_regions: 评估区域列表
            model_name: 模型名称
        """
        
        print(f"\n🔍 检查光场-标签匹配度 ({model_name})")
        print("-" * 50)
        
        match_results = []
        
        for mode_idx in range(self.config.num_modes):
            for wl_idx in range(len(self.config.wavelengths)):
                
                # 获取仿真光场强度
                if isinstance(simulated_fields, torch.Tensor):
                    sim_field = simulated_fields[mode_idx, wl_idx].cpu().numpy()
                else:
                    sim_field = np.array(simulated_fields[mode_idx, wl_idx])
                
                sim_intensity = np.abs(sim_field)**2
                
                # 获取标签
                if isinstance(labels, torch.Tensor):
                    label = labels[mode_idx, wl_idx].cpu().numpy()
                else:
                    label = np.array(labels[mode_idx, wl_idx])
                
                # 找到仿真光场的峰值位置
                sim_peak_y, sim_peak_x = np.unravel_index(np.argmax(sim_intensity), sim_intensity.shape)
                sim_peak_intensity = np.max(sim_intensity)
                
                # 找到标签的峰值位置
                label_peak_y, label_peak_x = np.unravel_index(np.argmax(label), label.shape)
                label_peak_intensity = np.max(label)
                
                # 计算位置偏差
                position_error = np.sqrt((sim_peak_x - label_peak_x)**2 + (sim_peak_y - label_peak_y)**2)
                
                # 获取对应的评估区域
                region_idx = wl_idx * self.config.num_modes + mode_idx
                if region_idx < len(evaluation_regions):
                    xs, xe, ys, ye = evaluation_regions[region_idx]
                    region_center_x = (xs + xe) / 2
                    region_center_y = (ys + ye) / 2
                    
                    # 计算仿真光场相对于评估区域的偏差
                    sim_to_region_error = np.sqrt((sim_peak_x - region_center_x)**2 + (sim_peak_y - region_center_y)**2)
                else:
                    region_center_x = region_center_y = None
                    sim_to_region_error = None
                
                # 计算光场质量指标
                # 1. 峰值背景比
                background_intensity = np.mean(sim_intensity)
                peak_to_background = sim_peak_intensity / (background_intensity + 1e-10)
                
                # 2. 在目标区域内的能量比例
                if region_idx < len(evaluation_regions):
                    xs, xe, ys, ye = evaluation_regions[region_idx]
                    target_region_intensity = np.sum(sim_intensity[ys:ye, xs:xe])
                    total_intensity = np.sum(sim_intensity)
                    energy_ratio = target_region_intensity / total_intensity if total_intensity > 0 else 0
                else:
                    energy_ratio = 0
                
                # 输出结果
                wl_nm = self.config.wavelengths[wl_idx] * 1e9
                print(f"  模式{mode_idx+1}, 波长{wl_nm:.0f}nm:")
                print(f"    仿真光场峰值: ({sim_peak_x:.1f}, {sim_peak_y:.1f}), 强度={sim_peak_intensity:.6f}")
                print(f"    标签峰值位置: ({label_peak_x:.1f}, {label_peak_y:.1f}), 强度={label_peak_intensity:.6f}")
                print(f"    位置偏差: {position_error:.1f} 像素")
                
                if region_center_x is not None:
                    print(f"    评估区域中心: ({region_center_x:.1f}, {region_center_y:.1f})")
                    print(f"    光场到区域偏差: {sim_to_region_error:.1f} 像素")
                
                print(f"    峰值背景比: {peak_to_background:.1f}")
                print(f"    目标区域能量: {energy_ratio:.1%}")
                
                # 判断匹配质量
                if position_error <= 2:
                    match_status = "excellent"
                    print(f"    ✅ 位置匹配优秀")
                elif position_error <= 5:
                    match_status = "good"
                    print(f"    ✅ 位置匹配良好")
                elif position_error <= 10:
                    match_status = "fair"
                    print(f"    ⚠️  位置偏差较大")
                else:
                    match_status = "poor"
                    print(f"    ❌ 位置偏差严重")
                
                # 判断聚焦质量
                if peak_to_background > 10 and energy_ratio > 0.3:
                    focus_status = "good"
                    print(f"    ✅ 聚焦质量良好")
                elif peak_to_background > 5 and energy_ratio > 0.2:
                    focus_status = "fair"
                    print(f"    ⚠️  聚焦质量一般")
                else:
                    focus_status = "poor"
                    print(f"    ❌ 聚焦质量差")
                
                print()
                
                # 保存结果
                match_results.append({
                    'mode_idx': mode_idx,
                    'wavelength_idx': wl_idx,
                    'wavelength_nm': wl_nm,
                    'sim_peak_pos': (sim_peak_x, sim_peak_y),
                    'label_peak_pos': (label_peak_x, label_peak_y),
                    'position_error': position_error,
                    'sim_to_region_error': sim_to_region_error,
                    'peak_to_background': peak_to_background,
                    'energy_ratio': energy_ratio,
                    'match_status': match_status,
                    'focus_status': focus_status
                })
        
        # 统计总体匹配情况
        self._print_summary(match_results, model_name)
        
        return match_results
    
    def _print_summary(self, match_results, model_name):
        """打印匹配度总结"""
        
        print(f"📊 {model_name} 总体匹配度统计:")
        
        # 位置匹配统计
        excellent_match = len([r for r in match_results if r['match_status'] == 'excellent'])
        good_match = len([r for r in match_results if r['match_status'] == 'good'])
        fair_match = len([r for r in match_results if r['match_status'] == 'fair'])
        poor_match = len([r for r in match_results if r['match_status'] == 'poor'])
        total = len(match_results)
        
        print(f"  位置匹配: 优秀{excellent_match}/{total}, 良好{good_match}/{total}, "
              f"一般{fair_match}/{total}, 差{poor_match}/{total}")
        
        # 聚焦质量统计
        good_focus = len([r for r in match_results if r['focus_status'] == 'good'])
        fair_focus = len([r for r in match_results if r['focus_status'] == 'fair'])
        poor_focus = len([r for r in match_results if r['focus_status'] == 'poor'])
        
        print(f"  聚焦质量: 良好{good_focus}/{total}, 一般{fair_focus}/{total}, 差{poor_focus}/{total}")
        
        # 平均指标
        avg_position_error = np.mean([r['position_error'] for r in match_results])
        avg_peak_to_bg = np.mean([r['peak_to_background'] for r in match_results])
        avg_energy_ratio = np.mean([r['energy_ratio'] for r in match_results])
        
        print(f"  平均位置偏差: {avg_position_error:.1f} 像素")
        print(f"  平均峰值背景比: {avg_peak_to_bg:.1f}")
        print(f"  平均目标区域能量: {avg_energy_ratio:.1%}")
        
        # 总体评价
        if excellent_match + good_match >= total * 0.8:
            print(f"  🎉 总体评价: 匹配度优秀!")
        elif excellent_match + good_match >= total * 0.6:
            print(f"  ✅ 总体评价: 匹配度良好")
        elif excellent_match + good_match >= total * 0.4:
            print(f"  ⚠️  总体评价: 匹配度一般，需要改进")
        else:
            print(f"  ❌ 总体评价: 匹配度差，需要重新调整参数")
    
    def visualize_focus_comparison(self, simulated_fields, labels, evaluation_regions, 
                                 save_dir, model_name="", show_plots=False):
        """
        可视化光场与标签的对比
        """
        
        print(f"\n📸 生成对比图像...")
        
        fig, axes = plt.subplots(self.config.num_modes, len(self.config.wavelengths) * 3, 
                                figsize=(len(self.config.wavelengths) * 9, self.config.num_modes * 3))
        
        if self.config.num_modes == 1:
            axes = axes.reshape(1, -1)
        
        for mode_idx in range(self.config.num_modes):
            for wl_idx in range(len(self.config.wavelengths)):
                
                # 获取数据
                if isinstance(simulated_fields, torch.Tensor):
                    sim_field = simulated_fields[mode_idx, wl_idx].cpu().numpy()
                else:
                    sim_field = np.array(simulated_fields[mode_idx, wl_idx])
                
                sim_intensity = np.abs(sim_field)**2
                
                if isinstance(labels, torch.Tensor):
                    label = labels[mode_idx, wl_idx].cpu().numpy()
                else:
                    label = np.array(labels[mode_idx, wl_idx])
                
                # 绘制仿真光场
                col_idx = wl_idx * 3
                im1 = axes[mode_idx, col_idx].imshow(sim_intensity, cmap='hot')
                axes[mode_idx, col_idx].set_title(f'仿真光场\n模式{mode_idx+1}, {self.config.wavelengths[wl_idx]*1e9:.0f}nm')
                plt.colorbar(im1, ax=axes[mode_idx, col_idx])
                
                # 绘制标签
                im2 = axes[mode_idx, col_idx+1].imshow(label, cmap='hot')
                axes[mode_idx, col_idx+1].set_title(f'标签\n模式{mode_idx+1}, {self.config.wavelengths[wl_idx]*1e9:.0f}nm')
                plt.colorbar(im2, ax=axes[mode_idx, col_idx+1])
                
                # 绘制叠加对比
                # 归一化到相同范围
                sim_norm = sim_intensity / np.max(sim_intensity) if np.max(sim_intensity) > 0 else sim_intensity
                label_norm = label / np.max(label) if np.max(label) > 0 else label
                
                # 创建RGB图像：红色=仿真，绿色=标签，黄色=重叠
                overlay = np.zeros((*sim_intensity.shape, 3))
                overlay[:, :, 0] = sim_norm  # 红色通道
                overlay[:, :, 1] = label_norm  # 绿色通道
                
                axes[mode_idx, col_idx+2].imshow(overlay)
                axes[mode_idx, col_idx+2].set_title(f'叠加对比\n红=仿真, 绿=标签, 黄=重叠')
                
                # 标记评估区域
                region_idx = wl_idx * self.config.num_modes + mode_idx
                if region_idx < len(evaluation_regions):
                    xs, xe, ys, ye = evaluation_regions[region_idx]
                    for ax in [axes[mode_idx, col_idx], axes[mode_idx, col_idx+1], axes[mode_idx, col_idx+2]]:
                        rect = plt.Rectangle((xs, ys), xe-xs, ye-ys, 
                                           fill=False, color='cyan', linewidth=2)
                        ax.add_patch(rect)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f"focus_comparison_{model_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  对比图像已保存: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 🎨 新增功能：绘制详细的光场和标签位置对比图
    def plot_detailed_field_and_labels(self, simulated_fields, labels, evaluation_regions, 
                                      save_dir, model_name="", show_plots=True):
        """
        绘制详细的光场分布和标签位置对比图
        
        Args:
            simulated_fields: 仿真光场 [num_modes, num_wavelengths, H, W]
            labels: 标签 [num_modes, num_wavelengths, H, W]
            evaluation_regions: 评估区域列表
            save_dir: 保存目录
            model_name: 模型名称
            show_plots: 是否显示图像
        """
        
        print(f"\n🎨 生成详细光场和标签位置对比图 ({model_name})...")
        
        # 转换为numpy数组
        if isinstance(simulated_fields, torch.Tensor):
            sim_fields_np = simulated_fields.cpu().numpy()
        else:
            sim_fields_np = np.array(simulated_fields)
            
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)
        
        # 获取目标位置
        target_positions = []
        for region in evaluation_regions:
            center_x = (region[0] + region[1]) // 2
            center_y = (region[2] + region[3]) // 2
            target_positions.append((center_x, center_y))
        
        # 创建大型综合对比图
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f'🔍 {model_name} 光场分布与标签位置详细对比', fontsize=20, fontweight='bold')
        
        # 为每个模式创建详细对比
        for mode_idx in range(min(self.config.num_modes, 3)):  # 最多显示3个模式
            wl_idx = 0  # 使用第一个波长
            
            # 获取光场数据
            sim_intensity = np.abs(sim_fields_np[mode_idx, wl_idx])**2
            label_intensity = labels_np[mode_idx, wl_idx]
            
            # 输入光场强度（如果有的话，这里用仿真结果代替）
            ax_input = plt.subplot(3, 4, mode_idx * 4 + 1)
            im1 = ax_input.imshow(sim_intensity, cmap='hot', origin='lower')
            ax_input.set_title(f'模式{mode_idx+1} 仿真光场强度', fontweight='bold')
            ax_input.set_xlabel('X (像素)')
            ax_input.set_ylabel('Y (像素)')
            plt.colorbar(im1, ax=ax_input, shrink=0.6)
            
            # 标签强度分布
            ax_label = plt.subplot(3, 4, mode_idx * 4 + 2)
            im2 = ax_label.imshow(label_intensity, cmap='hot', origin='lower')
            ax_label.set_title(f'模式{mode_idx+1} 标签分布', fontweight='bold')
            ax_label.set_xlabel('X (像素)')
            ax_label.set_ylabel('Y (像素)')
            plt.colorbar(im2, ax=ax_label, shrink=0.6)
            
            # 叠加对比图
            ax_overlay = plt.subplot(3, 4, mode_idx * 4 + 3)
            
            # 归一化
            sim_norm = sim_intensity / np.max(sim_intensity) if np.max(sim_intensity) > 0 else sim_intensity
            label_norm = label_intensity / np.max(label_intensity) if np.max(label_intensity) > 0 else label_intensity
            
            # 创建RGB叠加图
            overlay = np.zeros((*sim_intensity.shape, 3))
            overlay[:, :, 0] = sim_norm  # 红色=仿真
            overlay[:, :, 1] = label_norm  # 绿色=标签
            
            ax_overlay.imshow(overlay, origin='lower')
            ax_overlay.set_title(f'模式{mode_idx+1} 叠加对比\n红=仿真, 绿=标签, 黄=重叠', fontweight='bold')
            ax_overlay.set_xlabel('X (像素)')
            ax_overlay.set_ylabel('Y (像素)')
            
            # 标记目标位置和实际峰值
            if mode_idx < len(target_positions):
                target_x, target_y = target_positions[mode_idx]
                
                # 在所有子图上标记目标位置
                for ax in [ax_input, ax_label, ax_overlay]:
                    circle = patches.Circle((target_x, target_y), radius=8, 
                                          linewidth=2, edgecolor='cyan', facecolor='none')
                    ax.add_patch(circle)
                    ax.plot(target_x, target_y, 'c+', markersize=12, markeredgewidth=2)
                
                # 找到实际峰值位置
                peak_y, peak_x = np.unravel_index(np.argmax(sim_intensity), sim_intensity.shape)
                label_peak_y, label_peak_x = np.unravel_index(np.argmax(label_intensity), label_intensity.shape)
                
                # 标记实际峰值
                ax_input.plot(peak_x, peak_y, 'r*', markersize=12, markeredgewidth=2)
                ax_label.plot(label_peak_x, label_peak_y, 'g*', markersize=12, markeredgewidth=2)
                ax_overlay.plot(peak_x, peak_y, 'r*', markersize=12, markeredgewidth=2)
                ax_overlay.plot(label_peak_x, label_peak_y, 'g*', markersize=12, markeredgewidth=2)
                
                # 计算并显示偏差
                sim_deviation = np.sqrt((peak_x - target_x)**2 + (peak_y - target_y)**2)
                label_deviation = np.sqrt((label_peak_x - target_x)**2 + (label_peak_y - target_y)**2)
                
                ax_input.text(10, sim_intensity.shape[0]-20, f'偏差: {sim_deviation:.1f}px', 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                             fontweight='bold', fontsize=9)
                
                ax_label.text(10, label_intensity.shape[0]-20, f'偏差: {label_deviation:.1f}px', 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                             fontweight='bold', fontsize=9)
            
            # 光场分布剖面图
            ax_profile = plt.subplot(3, 4, mode_idx * 4 + 4)
            
            if mode_idx < len(target_positions):
                target_x, target_y = target_positions[mode_idx]
                
                # 沿目标位置的剖面
                if 0 <= target_y < sim_intensity.shape[0]:
                    sim_horizontal = sim_intensity[target_y, :]
                    label_horizontal = label_intensity[target_y, :]
                    ax_profile.plot(sim_horizontal, 'r-', linewidth=2, label=f'仿真(y={target_y})')
                    ax_profile.plot(label_horizontal, 'g-', linewidth=2, label=f'标签(y={target_y})')
                
                # 标记目标位置
                ax_profile.axvline(x=target_x, color='cyan', linestyle='--', alpha=0.7, label='目标X')
            
            ax_profile.set_title(f'模式{mode_idx+1} 光场强度剖面', fontweight='bold')
            ax_profile.set_xlabel('X位置 (像素)')
            ax_profile.set_ylabel('强度')
            ax_profile.legend(fontsize=8)
            ax_profile.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存详细对比图
        detailed_path = os.path.join(save_dir, f'{model_name}_detailed_field_label_comparison.png')
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        print(f"📊 详细光场对比图已保存: {detailed_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return detailed_path

    def plot_3d_field_distribution(self, simulated_fields, save_dir, model_name="", show_plots=True):
        """
        生成3D光场分布图
        
        Args:
            simulated_fields: 仿真光场
            save_dir: 保存目录
            model_name: 模型名称
            show_plots: 是否显示图像
        """
        
        print(f"\n🌊 生成3D光场分布图 ({model_name})...")
        
        # 转换数据
        if isinstance(simulated_fields, torch.Tensor):
            fields_np = simulated_fields.cpu().numpy()
        else:
            fields_np = np.array(simulated_fields)
        
        # 为提高3D渲染性能，降采样到较小尺寸
        original_size = fields_np.shape[-1]
        target_size = 100
        
        if original_size > target_size:
            # 简单降采样
            step = original_size // target_size
            fields_downsampled = fields_np[:, :, ::step, ::step]
        else:
            fields_downsampled = fields_np
        
        # 创建3D图
        fig = plt.figure(figsize=(18, 6))
        fig.suptitle(f'🌊 {model_name} 3D光场分布', fontsize=16, fontweight='bold')
        
        for mode_idx in range(min(self.config.num_modes, 3)):
            ax = fig.add_subplot(1, 3, mode_idx + 1, projection='3d')
            
            intensity = np.abs(fields_downsampled[mode_idx, 0])**2
            
            # 创建坐标网格
            size = intensity.shape[0]
            x = np.linspace(0, size-1, size)
            y = np.linspace(0, size-1, size)
            X, Y = np.meshgrid(x, y)
            
            # 3D表面图
            surf = ax.plot_surface(X, Y, intensity, cmap='hot', alpha=0.8)
            
            # 找到峰值位置
            peak_y, peak_x = np.unravel_index(np.argmax(intensity), intensity.shape)
            peak_intensity = np.max(intensity)
            
            # 标记峰值点
            ax.scatter([peak_x], [peak_y], [peak_intensity], color='red', s=100, alpha=1.0)
            
            ax.set_title(f'模式{mode_idx+1} 3D强度分布\n峰值位置:({peak_x},{peak_y})', fontweight='bold')
            ax.set_xlabel('X (像素)')
            ax.set_ylabel('Y (像素)')
            ax.set_zlabel('强度')
            
            # 添加颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5)
        
        plt.tight_layout()
        
        # 保存3D图
        path_3d = os.path.join(save_dir, f'{model_name}_3d_field_distribution.png')
        plt.savefig(path_3d, dpi=300, bbox_inches='tight')
        print(f"🌊 3D光场分布图已保存: {path_3d}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return path_3d

    def plot_focus_overview(self, simulated_fields, evaluation_regions, save_dir, model_name="", show_plots=True):
        """
        生成聚焦效果总览图
        
        Args:
            simulated_fields: 仿真光场
            evaluation_regions: 评估区域
            save_dir: 保存目录
            model_name: 模型名称
            show_plots: 是否显示图像
        """
        
        print(f"\n🎯 生成聚焦效果总览图 ({model_name})...")
        
        # 转换数据
        if isinstance(simulated_fields, torch.Tensor):
            fields_np = simulated_fields.cpu().numpy()
        else:
            fields_np = np.array(simulated_fields)
        
        # 获取目标位置
        target_positions = []
        for region in evaluation_regions:
            center_x = (region[0] + region[1]) // 2
            center_y = (region[2] + region[3]) // 2
            target_positions.append((center_x, center_y))
        
        # 创建聚焦效果总览图
        fig, axes = plt.subplots(1, min(self.config.num_modes, 3), figsize=(18, 6))
        if self.config.num_modes == 1:
            axes = [axes]
        
        fig.suptitle(f'🎯 {model_name} 聚焦效果总览', fontsize=16, fontweight='bold')
        
        for mode_idx in range(min(self.config.num_modes, 3)):
            ax = axes[mode_idx] if len(axes) > 1 else axes[0]
            intensity = np.abs(fields_np[mode_idx, 0])**2
            
            # 显示光场
            im = ax.imshow(intensity, cmap='hot', origin='lower', extent=[0, intensity.shape[1], 0, intensity.shape[0]])
            
            # 标记目标位置
            if mode_idx < len(target_positions):
                target_x, target_y = target_positions[mode_idx]
                
                # 目标区域框
                rect = patches.Rectangle((target_x-7, target_y-7), 14, 14, 
                                       linewidth=2, edgecolor='cyan', facecolor='none')
                ax.add_patch(rect)
                
                # 目标中心点
                ax.plot(target_x, target_y, 'c+', markersize=20, markeredgewidth=3)
                
                # 找到实际峰值
                peak_y, peak_x = np.unravel_index(np.argmax(intensity), intensity.shape)
                ax.plot(peak_x, peak_y, 'r*', markersize=20, markeredgewidth=2)
                
                # 连线显示偏差
                ax.plot([target_x, peak_x], [target_y, peak_y], 'w--', linewidth=2, alpha=0.8)
                
                # 计算并显示偏差
                deviation = np.sqrt((peak_x - target_x)**2 + (peak_y - target_y)**2)
                
                ax.set_title(f'模式{mode_idx+1}\n目标:({target_x},{target_y}) 实际:({peak_x},{peak_y})\n偏差:{deviation:.1f}px', 
                           fontweight='bold')
                
                # 添加图例
                if mode_idx == 0:
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='+', color='cyan', linestyle='None', 
                               markersize=15, markeredgewidth=3, label='目标位置'),
                        Line2D([0], [0], marker='*', color='red', linestyle='None', 
                               markersize=15, markeredgewidth=2, label='实际峰值'),
                        Line2D([0], [0], color='white', linestyle='--', linewidth=2, label='偏差连线')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            ax.set_xlabel('X (像素)')
            ax.set_ylabel('Y (像素)')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        
        # 保存聚焦总览图
        overview_path = os.path.join(save_dir, f'{model_name}_focus_overview.png')
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        print(f"🎯 聚焦效果总览图已保存: {overview_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return overview_path

    def generate_statistical_report(self, match_results, save_dir, model_name=""):
        """
        生成详细的统计报告
        
        Args:
            match_results: 匹配结果列表
            save_dir: 保存目录
            model_name: 模型名称
        """
        
        print(f"\n📊 生成统计报告 ({model_name})...")
        
        if not match_results:
            print("❌ 没有匹配结果数据")
            return None
        
        # 创建统计图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'📊 {model_name} 详细统计分析报告', fontsize=16, fontweight='bold')
        
        # 提取数据
        position_errors = [r['position_error'] for r in match_results]
        energy_ratios = [r['energy_ratio'] * 100 for r in match_results]  # 转换为百分比
        peak_to_bg_ratios = [r['peak_to_background'] for r in match_results]
        wavelengths = [r['wavelength_nm'] for r in match_results]
        modes = [f"模式{r['mode_idx']+1}" for r in match_results]
        
        # 1. 位置偏差分布
        ax1.hist(position_errors, bins=10, color='lightcoral', alpha=0.7, edgecolor='black')
        ax1.set_title('位置偏差分布', fontweight='bold')
        ax1.set_xlabel('位置偏差 (像素)')
        ax1.set_ylabel('频次')
        ax1.axvline(np.mean(position_errors), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(position_errors):.1f}px')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 能量聚焦比例分布
        ax2.hist(energy_ratios, bins=10, color='lightblue', alpha=0.7, edgecolor='black')
        ax2.set_title('能量聚焦比例分布', fontweight='bold')
        ax2.set_xlabel('能量比例 (%)')
        ax2.set_ylabel('频次')
        ax2.axvline(np.mean(energy_ratios), color='blue', linestyle='--', 
                   label=f'平均值: {np.mean(energy_ratios):.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 峰值背景比分布
        ax3.hist(peak_to_bg_ratios, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
        ax3.set_title('峰值背景比分布', fontweight='bold')
        ax3.set_xlabel('峰值背景比')
        ax3.set_ylabel('频次')
        ax3.axvline(np.mean(peak_to_bg_ratios), color='green', linestyle='--', 
                   label=f'平均值: {np.mean(peak_to_bg_ratios):.1f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 各模式性能对比
        unique_modes = list(set(modes))
        mode_errors = []
        mode_energies = []
        
        for mode in unique_modes:
            mode_indices = [i for i, m in enumerate(modes) if m == mode]
            mode_errors.append(np.mean([position_errors[i] for i in mode_indices]))
            mode_energies.append(np.mean([energy_ratios[i] for i in mode_indices]))
        
        x_pos = np.arange(len(unique_modes))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, mode_errors, width, label='位置偏差(px)', color='orange', alpha=0.7)
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x_pos + width/2, mode_energies, width, label='能量比例(%)', color='purple', alpha=0.7)
        
        ax4.set_title('各模式性能对比', fontweight='bold')
        ax4.set_xlabel('模式')
        ax4.set_ylabel('位置偏差 (像素)', color='orange')
        ax4_twin.set_ylabel('能量比例 (%)', color='purple')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(unique_modes)
        
        # 添加数值标签
        for bar, value in zip(bars1, mode_errors):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        for bar, value in zip(bars2, mode_energies):
            ax4_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                         f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存统计报告图
        report_path = os.path.join(save_dir, f'{model_name}_statistical_report.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"📊 统计报告图已保存: {report_path}")
        plt.show()
        
        # 生成文本报告
        report_text_path = os.path.join(save_dir, f'{model_name}_report.txt')
        with open(report_text_path, 'w', encoding='utf-8') as f:
            f.write(f"🔍 {model_name} 光场匹配度详细报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("📊 总体统计:\n")
            f.write(f"  总测试点数: {len(match_results)}\n")
            f.write(f"  平均位置偏差: {np.mean(position_errors):.2f} ± {np.std(position_errors):.2f} 像素\n")
            f.write(f"  平均能量聚焦比例: {np.mean(energy_ratios):.1f} ± {np.std(energy_ratios):.1f} %\n")
            f.write(f"  平均峰值背景比: {np.mean(peak_to_bg_ratios):.1f} ± {np.std(peak_to_bg_ratios):.1f}\n\n")
            
            f.write("🎯 匹配质量分布:\n")
            excellent = len([r for r in match_results if r['match_status'] == 'excellent'])
            good = len([r for r in match_results if r['match_status'] == 'good'])
            fair = len([r for r in match_results if r['match_status'] == 'fair'])
            poor = len([r for r in match_results if r['match_status'] == 'poor'])
            
            f.write(f"  优秀 (≤2px): {excellent} ({excellent/len(match_results)*100:.1f}%)\n")
            f.write(f"  良好 (≤5px): {good} ({good/len(match_results)*100:.1f}%)\n")
            f.write(f"  一般 (≤10px): {fair} ({fair/len(match_results)*100:.1f}%)\n")
            f.write(f"  差 (>10px): {poor} ({poor/len(match_results)*100:.1f}%)\n\n")
            
            f.write("🔥 聚焦质量分布:\n")
            good_focus = len([r for r in match_results if r['focus_status'] == 'good'])
            fair_focus = len([r for r in match_results if r['focus_status'] == 'fair'])
            poor_focus = len([r for r in match_results if r['focus_status'] == 'poor'])
            
            f.write(f"  良好: {good_focus} ({good_focus/len(match_results)*100:.1f}%)\n")
            f.write(f"  一般: {fair_focus} ({fair_focus/len(match_results)*100:.1f}%)\n")
            f.write(f"  差: {poor_focus} ({poor_focus/len(match_results)*100:.1f}%)\n\n")
            
            f.write("📋 详细结果:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'模式':<6} {'波长(nm)':<10} {'目标位置':<12} {'实际位置':<12} {'偏差(px)':<10} {'能量(%)':<8} {'状态':<8}\n")
            f.write("-" * 80 + "\n")
            
            for r in match_results:
                target_pos = f"({r['label_peak_pos'][0]:.0f},{r['label_peak_pos'][1]:.0f})"
                actual_pos = f"({r['sim_peak_pos'][0]:.0f},{r['sim_peak_pos'][1]:.0f})"
                f.write(f"{r['mode_idx']+1:<6} {r['wavelength_nm']:<10.0f} {target_pos:<12} "
                       f"{actual_pos:<12} {r['position_error']:<10.1f} {r['energy_ratio']*100:<8.1f} "
                       f"{r['match_status']:<8}\n")
        
        print(f"📄 文本报告已保存: {report_text_path}")
        
        return report_path, report_text_path

    def comprehensive_analysis(self, simulated_fields, labels, evaluation_regions, 
                             save_dir, model_name="", show_plots=True):
        """
        综合分析函数 - 一次性生成所有图表和报告
        
        Args:
            simulated_fields: 仿真光场
            labels: 标签
            evaluation_regions: 评估区域
            save_dir: 保存目录
            model_name: 模型名称
            show_plots: 是否显示图像
        """
        
        print(f"\n🚀 开始综合分析 ({model_name})...")
        print("=" * 60)
        
        # 1. 检查匹配度
        match_results = self.check_focus_match(simulated_fields, labels, evaluation_regions, model_name)
        
        # 2. 生成详细光场对比图
        self.plot_detailed_field_and_labels(simulated_fields, labels, evaluation_regions, 
                                           save_dir, model_name, show_plots)
        
        # 3. 生成3D分布图
        self.plot_3d_field_distribution(simulated_fields, save_dir, model_name, show_plots)
        
        # 4. 生成聚焦总览图
        self.plot_focus_overview(simulated_fields, evaluation_regions, save_dir, model_name, show_plots)
        
        # 5. 生成统计报告
        self.generate_statistical_report(match_results, save_dir, model_name)
        
        # 6. 原有的对比图
        self.visualize_focus_comparison(simulated_fields, labels, evaluation_regions, 
                                      save_dir, model_name, show_plots)
        
        print(f"\n✅ {model_name} 综合分析完成！")
        print(f"📁 所有文件已保存到: {save_dir}")
        
        return match_results

