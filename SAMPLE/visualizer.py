import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from matplotlib.colors import LinearSegmentedColormap

class ImprovedVisualizer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 🔥 改进的配色方案
        self.energy_cmap = plt.cm.plasma
        self.phase_cmap = plt.cm.hsv
        
        # 创建自定义配色
        colors_wavelength = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        self.wavelength_colors = colors_wavelength[:len(config.wavelengths)]

    def plot_improved_training_history(self, training_history, save_path=None):
        """🔥 改进的训练历史可视化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(len(training_history['total_loss']))
        
        # 1. 总损失
        axes[0, 0].plot(epochs, training_history['total_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('总损失', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('训练轮次')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # 2. 分项损失
        axes[0, 1].plot(epochs, training_history['efficiency_loss'], label='效率损失', linewidth=2)
        axes[0, 1].plot(epochs, training_history['separation_loss'], label='分离损失', linewidth=2)
        axes[0, 1].plot(epochs, training_history['crosstalk_loss'], label='串扰损失', linewidth=2)
        axes[0, 1].plot(epochs, training_history['concentration_loss'], label='集中损失', linewidth=2)
        axes[0, 1].plot(epochs, training_history['smoothing_loss'], label='平滑损失', linewidth=2)
        axes[0, 1].set_title('分项损失', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('损失值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # 3. 效率演化
        efficiencies_array = np.array(training_history['efficiencies'])
        for w_idx in range(len(self.config.wavelengths)):
            wavelength_nm = int(self.config.wavelengths[w_idx] * 1e9)
            axes[0, 2].plot(epochs, efficiencies_array[:, w_idx], 
                           label=f'{wavelength_nm}nm', 
                           color=self.wavelength_colors[w_idx], linewidth=2)
        axes[0, 2].set_title('各波长效率演化', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('训练轮次')
        axes[0, 2].set_ylabel('效率')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
        
        # 4. 学习率
        axes[1, 0].plot(epochs, training_history['learning_rates'], 'g-', linewidth=2)
        axes[1, 0].set_title('学习率调度', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('训练轮次')
        axes[1, 0].set_ylabel('学习率')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # 5. 效率分布
        final_efficiencies = efficiencies_array[-1] if len(efficiencies_array) > 0 else [0] * len(self.config.wavelengths)
        wavelength_labels = [f'{int(wl*1e9)}nm' for wl in self.config.wavelengths]
        bars = axes[1, 1].bar(wavelength_labels, final_efficiencies, 
                             color=self.wavelength_colors, alpha=0.7)
        axes[1, 1].set_title('最终效率分布', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('效率')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, eff in zip(bars, final_efficiencies):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{eff:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. 损失收敛分析
        window_size = max(1, len(training_history['total_loss']) // 20)
        if len(training_history['total_loss']) > window_size:
            smoothed_loss = np.convolve(training_history['total_loss'], 
                                      np.ones(window_size)/window_size, mode='valid')
            smoothed_epochs = epochs[window_size-1:]
            axes[1, 2].plot(epochs, training_history['total_loss'], alpha=0.3, color='blue')
            axes[1, 2].plot(smoothed_epochs, smoothed_loss, 'r-', linewidth=2, label='平滑曲线')
            axes[1, 2].legend()
        else:
            axes[1, 2].plot(epochs, training_history['total_loss'], 'b-', linewidth=2)
        
        axes[1, 2].set_title('损失收敛分析', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('训练轮次')
        axes[1, 2].set_ylabel('总损失')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_yscale('log')
        
        plt.suptitle('🔥 改进训练历史分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_wavelength_dependent_phase_masks(self, model, save_path=None):
        """🔥 可视化每个波长的独立相位掩膜"""
        phase_masks_vis = model.get_phase_masks_for_visualization()
        num_layers = len(phase_masks_vis)
        num_wavelengths = len(self.config.wavelengths)
        
        fig, axes = plt.subplots(num_layers, num_wavelengths, 
                                figsize=(5*num_wavelengths, 5*num_layers))
        
        # 处理单层或单波长情况
        if num_layers == 1:
            axes = [axes] if num_wavelengths > 1 else [[axes]]
        elif num_wavelengths == 1:
            axes = [[ax] for ax in axes]
        
        for layer_idx in range(num_layers):
            for w_idx in range(num_wavelengths):
                phase = phase_masks_vis[layer_idx][w_idx].numpy()
                phase = (phase % (2 * np.pi))
                
                ax = axes[layer_idx][w_idx] if num_layers > 1 else axes[0][w_idx]
                im = ax.imshow(phase, cmap=self.phase_cmap, vmin=0, vmax=2*np.pi)
                
                wavelength_nm = int(self.config.wavelengths[w_idx] * 1e9)
                if num_layers > 1:
                    ax.set_title(f"层{layer_idx+1} - {wavelength_nm}nm", fontweight='bold')
                else:
                    ax.set_title(f"{wavelength_nm}nm", fontweight='bold')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        # 添加颜色条
        if num_layers > 1:
            cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                               fraction=0.046, pad=0.08)
        else:
            cbar = fig.colorbar(im, ax=axes[0], orientation='horizontal', 
                               fraction=0.046, pad=0.08)
        cbar.set_label('相位 (弧度)', fontsize=12)
        
        plt.suptitle('🔥 波长独立相位掩膜', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_improved_energy_distribution(self, output_fields, save_path=None):
        """🔥 改进的能量分布可视化"""
        num_wavelengths = len(self.config.wavelengths)
        
        fig, axes = plt.subplots(2, num_wavelengths, figsize=(6*num_wavelengths, 12))
        
        # 处理单波长情况
        if num_wavelengths == 1:
            axes = axes.reshape(-1, 1)
        
        for w_idx, wavelength in enumerate(self.config.wavelengths):
            field = output_fields[w_idx]
            energy = torch.abs(field)**2
            energy_np = energy.detach().cpu().numpy()
            
            # 第一行：原始能量分布
            energy_normalized = energy_np / np.max(energy_np)
            im1 = axes[0, w_idx].imshow(energy_normalized, cmap=self.energy_cmap, origin='lower')
            
            # 添加检测区域和等高线
            offset_x, offset_y = self.config.offsets[w_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            detect_region = plt.Rectangle(
                (center_x - self.config.detect_size // 2, center_y - self.config.detect_size // 2),
                self.config.detect_size, self.config.detect_size,
                fill=False, edgecolor='white', linestyle='--', linewidth=3
            )
            axes[0, w_idx].add_patch(detect_region)
            
            # 添加等高线
            contour = axes[0, w_idx].contour(energy_normalized, levels=8, colors='white', alpha=0.6, linewidths=1)
            
            wavelength_nm = wavelength * 1e9
            axes[0, w_idx].set_title(f"能量分布 - {wavelength_nm:.0f}nm", fontweight='bold')
            axes[0, w_idx].set_xlabel('X (像素)')
            axes[0, w_idx].set_ylabel('Y (像素)')
            
            # 第二行：3D能量分布
            x = np.arange(self.config.field_size)
            y = np.arange(self.config.field_size)
            X, Y = np.meshgrid(x, y)
            
            ax_3d = fig.add_subplot(2, num_wavelengths, num_wavelengths + w_idx + 1, projection='3d')
            
            # 降采样以提高性能
            step = max(1, self.config.field_size // 50)
            X_sub = X[::step, ::step]
            Y_sub = Y[::step, ::step]
            energy_sub = energy_normalized[::step, ::step]
            
            surf = ax_3d.plot_surface(X_sub, Y_sub, energy_sub, cmap=self.energy_cmap, alpha=0.8)
            ax_3d.set_title(f"3D视图 - {wavelength_nm:.0f}nm", fontweight='bold')
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('归一化能量')
            
            # 移除原来的axes[1, w_idx]
            axes[1, w_idx].remove()
        
        plt.suptitle('🔥 改进能量分布分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_comparison_original_vs_improved(self, original_results, improved_results, save_path=None):
        """🔥 原版 vs 改进版对比"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 效率对比
        wavelengths_nm = [int(wl*1e9) for wl in self.config.wavelengths]
        x = np.arange(len(wavelengths_nm))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, original_results['final_efficiencies'], width, 
                      label='原版', alpha=0.7, color='lightblue')
        axes[0, 0].bar(x + width/2, improved_results['final_efficiencies'], width,
                      label='改进版', alpha=0.7, color='orange')
        
        axes[0, 0].set_title('效率对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('波长 (nm)')
        axes[0, 0].set_ylabel('效率')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(wavelengths_nm)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (orig, impr) in enumerate(zip(original_results['final_efficiencies'], 
                                           improved_results['final_efficiencies'])):
            axes[0, 0].text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center', va='bottom')
            axes[0, 0].text(i + width/2, impr + 0.01, f'{impr:.3f}', ha='center', va='bottom')
        
        # 2. 训练损失对比
        orig_epochs = range(len(original_results['training_history']['total_loss']))
        impr_epochs = range(len(improved_results['training_history']['total_loss']))
        
        axes[0, 1].plot(orig_epochs, original_results['training_history']['total_loss'], 
                       label='原版', linewidth=2, alpha=0.7)
        axes[0, 1].plot(impr_epochs, improved_results['training_history']['total_loss'], 
                       label='改进版', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('训练损失对比', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('损失值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # 3. 训练时间对比
        times = [original_results['training_time'], improved_results['training_time']]
        labels = ['原版', '改进版']
        colors = ['lightblue', 'orange']
        
        bars = axes[0, 2].bar(labels, times, color=colors, alpha=0.7)
        axes[0, 2].set_title('训练时间对比', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('时间 (秒)')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        for bar, time in zip(bars, times):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                           f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. 平均效率提升
        orig_avg = np.mean(original_results['final_efficiencies'])
        impr_avg = np.mean(improved_results['final_efficiencies'])
        improvement = (impr_avg - orig_avg) / orig_avg * 100
        
        axes[1, 0].bar(['原版', '改进版'], [orig_avg, impr_avg], 
                      color=['lightblue', 'orange'], alpha=0.7)
        axes[1, 0].set_title(f'平均效率提升: +{improvement:.1f}%', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('平均效率')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        axes[1, 0].text(0, orig_avg + 0.01, f'{orig_avg:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[1, 0].text(1, impr_avg + 0.01, f'{impr_avg:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. 最终损失对比
        final_losses = [original_results['training_history']['total_loss'][-1],
                       improved_results['training_history']['total_loss'][-1]]
        
        bars = axes[1, 1].bar(['原版', '改进版'], final_losses, 
                             color=['lightblue', 'orange'], alpha=0.7)
        axes[1, 1].set_title('最终损失对比', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('损失值')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, loss in zip(bars, final_losses):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                           f'{loss:.2e}', ha='center', va='bottom', fontweight='bold')
        
        # 6. 改进总结
        axes[1, 2].axis('off')
        summary_text = f"""
                    🔥 改进效果总结:

                    📈 平均效率提升: {improvement:+.1f}%
                    ⏱️ 训练时间: {original_results['training_time']:.1f}s → {improved_results['training_time']:.1f}s
                    📉 最终损失降低: {(1-final_losses[1]/final_losses[0])*100:.1f}%

                    🔧 主要改进:
                    • 波长独立相位掩膜
                    • 多目标损失函数优化
                    • AdamW + 余弦退火调度
                    • 相位平滑约束
                    • 差分检测机制

                    ✅ 性能提升显著!
                            """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('🔥 原版 vs 改进版 全面对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
