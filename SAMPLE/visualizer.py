import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from matplotlib.colors import LinearSegmentedColormap

class SimpleVisualizer:
    def __init__(self, config):
        self.config = config
        self.device = config.device  # 添加设备属性
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 创建自定义热图配色方案，用于能量分布可视化
        self.energy_cmap = plt.cm.viridis
        
        # 创建相位图配色方案
        self.phase_cmap = plt.cm.hsv

    def plot_detector_regions(self, save_path=None):
        """绘制检测区域的位置"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制场边界
        field_boundary = plt.Rectangle((0, 0), self.config.field_size, self.config.field_size, 
                                       fill=False, edgecolor='black', linestyle='--')
        ax.add_patch(field_boundary)
        
        # 为每个波长绘制检测区域
        colors = ['blue', 'red']
        for i, (offset_x, offset_y) in enumerate(self.config.offsets):
            # 计算检测区域的中心
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            # 绘制检测区域
            detect_region = plt.Rectangle(
                (center_x - self.config.detect_size // 2, center_y - self.config.detect_size // 2),
                self.config.detect_size, self.config.detect_size,
                fill=True, alpha=0.3, edgecolor=colors[i], facecolor=colors[i]
            )
            ax.add_patch(detect_region)
            
            # 添加波长标签
            wavelength_nm = self.config.wavelengths[i] * 1e9
            ax.text(center_x, center_y, f"{wavelength_nm:.0f}nm", 
                   ha='center', va='center', color='white', fontweight='bold')
        
        ax.set_xlim(0, self.config.field_size)
        ax.set_ylim(0, self.config.field_size)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('检测区域位置')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_training_losses(self, losses, num_layer_options, save_path=None):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 6))
        
        for i, num_layers in enumerate(num_layer_options):
            plt.plot(losses[i], label=f"{num_layers}层")
        
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title('不同层数模型的训练损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_visibility_comparison(self, visibility, num_layer_options, save_path=None):
        """绘制不同层数模型的可见度比较"""
        plt.figure(figsize=(8, 6))
        
        plt.bar(range(len(num_layer_options)), visibility, color='skyblue')
        plt.xticks(range(len(num_layer_options)), [f"{n}层" for n in num_layer_options])
        
        # 在每个柱形上方显示具体数值
        for i, v in enumerate(visibility):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        plt.xlabel('模型层数')
        plt.ylabel('可见度')
        plt.title('不同层数模型的可见度比较')
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_phase_masks_layers(self, phase_masks, num_layers, save_path=None):
        """绘制相位掩膜（多层版本）"""
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))
        
        # 处理单层情况
        if num_layers == 1:
            axes = [axes]
        
        for i in range(num_layers):
            # 将相位值标准化到[0, 2π]范围 - 确保在CPU上处理
            phase = phase_masks[i].detach().cpu().numpy()
            phase = (phase % (2 * np.pi))
            
            im = axes[i].imshow(phase, cmap=self.phase_cmap, vmin=0, vmax=2*np.pi)
            axes[i].set_title(f"层 {i+1}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('相位 (弧度)')
        
        plt.suptitle(f"{num_layers}层模型的相位掩膜", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_energy_distribution(self, energy_normalized, num_layers, save_path=None):
        """绘制能量分布"""
        # 获取每个波长的能量分布
        num_wavelengths = len(self.config.wavelengths)
        
        # 创建一个大图，包含所有波长的能量分布
        fig, axes = plt.subplots(num_wavelengths, 1, figsize=(8, 4*num_wavelengths))
        
        # 处理单波长情况
        if num_wavelengths == 1:
            axes = [axes]
        
        # 为每个波长绘制能量分布
        for w_idx in range(num_wavelengths):
            # 确保数据在CPU上处理
            energy = energy_normalized[w_idx].detach().cpu().numpy()
            
            # 使用热图显示能量分布
            im = axes[w_idx].imshow(energy, cmap=self.energy_cmap, origin='lower')
            
            # 添加检测区域标记
            offset_x, offset_y = self.config.offsets[w_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            detect_region = plt.Rectangle(
                (center_x - self.config.detect_size // 2, center_y - self.config.detect_size // 2),
                self.config.detect_size,
                fill=False, edgecolor='white', linestyle='--', linewidth=2
            )
            axes[w_idx].add_patch(detect_region)
            
            # 添加等高线以更好地显示能量分布
            contour = axes[w_idx].contour(energy, levels=5, colors='white', alpha=0.5, linewidths=0.8)
            
            # 设置标题和标签
            wavelength_nm = self.config.wavelengths[w_idx] * 1e9
            axes[w_idx].set_title(f"波长: {wavelength_nm:.0f}nm")
            axes[w_idx].set_xlabel('X (像素)')
            axes[w_idx].set_ylabel('Y (像素)')
            
            # 添加颜色条
            cbar = fig.colorbar(im, ax=axes[w_idx])
            cbar.set_label('归一化能量')
        
        plt.suptitle(f"{num_layers}层模型的能量分布", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_field_propagation(self, model, input_fields, save_path=None):
        """绘制光场传播过程"""
        # 获取模型的层数
        num_layers = len(model.phase_masks)
        
        # 获取波长数量
        num_wavelengths = len(self.config.wavelengths)
        
        # 确保输入场在正确设备上
        if isinstance(input_fields, list):
            input_fields = [field.to(self.device) for field in input_fields]
        else:
            input_fields = input_fields.to(self.device)
        
        # 使用模型计算每一层的场分布
        with torch.no_grad():
            # 获取每一层的场分布
            all_fields = model.get_all_fields(input_fields)
        
        # 创建一个大图，包含所有传播步骤
        fig, axes = plt.subplots(num_wavelengths, num_layers + 1, 
                                 figsize=(4*(num_layers+1), 4*num_wavelengths))
        
        # 处理单波长或单层的情况
        if num_wavelengths == 1:
            axes = [axes]
        if num_layers == 0:  # 处理没有层的情况
            axes = [[ax] for ax in axes] if num_wavelengths > 1 else [axes]
        
        # 绘制每个波长在每一层的场分布
        for w_idx in range(num_wavelengths):
            wavelength_nm = self.config.wavelengths[w_idx] * 1e9
            
            # 绘制输入场 - 确保在CPU上处理
            if isinstance(input_fields, list):
                input_field = input_fields[w_idx].detach().cpu().numpy()
            else:
                input_field = input_fields[w_idx].detach().cpu().numpy()
            input_intensity = np.abs(input_field)**2
            
            im = axes[w_idx][0].imshow(input_intensity, cmap='inferno', origin='lower')
            axes[w_idx][0].set_title(f"输入场\n{wavelength_nm:.0f}nm")
            axes[w_idx][0].set_xticks([])
            axes[w_idx][0].set_yticks([])
            
            # 为每一层绘制场分布
            for l_idx in range(num_layers):
                # 确保在CPU上处理
                field = all_fields[w_idx][l_idx].detach().cpu().numpy()
                intensity = np.abs(field)**2
                
                # 归一化强度以便更好地可视化
                intensity = intensity / np.max(intensity)
                
                im = axes[w_idx][l_idx+1].imshow(intensity, cmap='inferno', origin='lower')
                axes[w_idx][l_idx+1].set_title(f"层 {l_idx+1} 之后\n{wavelength_nm:.0f}nm")
                axes[w_idx][l_idx+1].set_xticks([])
                axes[w_idx][l_idx+1].set_yticks([])
                
                # 添加检测区域标记（仅在最后一层）
                if l_idx == num_layers - 1:
                    offset_x, offset_y = self.config.offsets[w_idx]
                    center_x = self.config.field_size // 2 + offset_x
                    center_y = self.config.field_size // 2 + offset_y
                    
                    detect_region = plt.Rectangle(
                        (center_x - self.config.detect_size // 2, center_y - self.config.detect_size // 2),
                        self.config.detect_size, self.config.detect_size,
                        fill=False, edgecolor='white', linestyle='--', linewidth=2
                    )
                    axes[w_idx][l_idx+1].add_patch(detect_region)
                
                # 添加等高线
                contour = axes[w_idx][l_idx+1].contour(intensity, levels=5, colors='white', alpha=0.5, linewidths=0.8)
        
        # 添加颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('归一化强度')
        
        plt.suptitle("光场传播过程", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_loss_curve(self, losses, title, save_path=None):
        """绘制训练损失曲线（兼容单模型情况）"""
        plt.figure(figsize=(8, 5))
        plt.plot(losses, label='训练损失')
        plt.xlabel('训练步数')
        plt.ylabel('损失')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_energy_distributions(self, output_fields, wavelengths, save_path=None):
        """绘制多波长能量分布"""
        num_wavelengths = len(wavelengths)
        
        # 创建子图
        fig, axes = plt.subplots(1, num_wavelengths, figsize=(6*num_wavelengths, 5))
        
        # 处理单波长情况
        if num_wavelengths == 1:
            axes = [axes]
        
        for w_idx, wavelength in enumerate(wavelengths):
            # 计算能量分布 - 确保在CPU上处理
            field = output_fields[w_idx]
            energy = torch.abs(field)**2
            energy_np = energy.detach().cpu().numpy()
            
            # 归一化
            energy_normalized = energy_np / np.max(energy_np)
            
            # 绘制能量分布
            im = axes[w_idx].imshow(energy_normalized, cmap=self.energy_cmap, origin='lower')
            
            # 添加检测区域标记
            offset_x, offset_y = self.config.offsets[w_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            detect_region = plt.Rectangle(
                (center_x - self.config.detect_size // 2, center_y - self.config.detect_size // 2),
                self.config.detect_size, self.config.detect_size,
                fill=False, edgecolor='white', linestyle='--', linewidth=2
            )
            axes[w_idx].add_patch(detect_region)
            
            # 设置标题和标签
            wavelength_nm = wavelength * 1e9
            axes[w_idx].set_title(f"波长: {wavelength_nm:.0f}nm")
            axes[w_idx].set_xlabel('X (像素)')
            axes[w_idx].set_ylabel('Y (像素)')
            
            # 添加颜色条
            cbar = fig.colorbar(im, ax=axes[w_idx])
            cbar.set_label('归一化能量')
        
        plt.suptitle("多波长能量分布", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_phase_masks(self, phase_masks, save_path=None):
        """绘制相位掩膜"""
        num_layers = len(phase_masks)
        
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))
        
        # 处理单层情况
        if num_layers == 1:
            axes = [axes]
        
        for i in range(num_layers):
            # 将相位值标准化到[0, 2π]范围 - 确保在CPU上处理
            phase = phase_masks[i].detach().cpu().numpy()
            phase = (phase % (2 * np.pi))
            
            im = axes[i].imshow(phase, cmap=self.phase_cmap, vmin=0, vmax=2*np.pi)
            axes[i].set_title(f"层 {i+1}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.046, pad=0.08)
        cbar.set_label('相位 (弧度)')
        
        plt.suptitle(f"相位掩膜", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_field_propagation_simple(self, fields, title, save_path=None):
        """绘制光场传播过程（简化版本）"""
        num_steps = len(fields)
        
        fig, axes = plt.subplots(1, num_steps, figsize=(4*num_steps, 4))
        
        # 处理单步情况
        if num_steps == 1:
            axes = [axes]
        
        for i, field in enumerate(fields):
            # 计算强度 - 确保在CPU上处理
            intensity = torch.abs(field)**2
            intensity_np = intensity.detach().cpu().numpy()
            
            # 归一化
            intensity_normalized = intensity_np / np.max(intensity_np)
            
            # 绘制强度分布
            im = axes[i].imshow(intensity_normalized, cmap='inferno', origin='lower')
            axes[i].set_title(f"步骤 {i+1}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
            # 添加等高线
            contour = axes[i].contour(intensity_normalized, levels=5, colors='white', alpha=0.5, linewidths=0.8)
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.046, pad=0.08)
        cbar.set_label('归一化强度')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_target_regions(self, output_fields, wavelengths, offsets, detect_size, save_path=None):
        """绘制目标区域的能量分布"""
        num_wavelengths = len(wavelengths)
        
        fig, axes = plt.subplots(1, num_wavelengths, figsize=(6*num_wavelengths, 5))
        
        # 处理单波长情况
        if num_wavelengths == 1:
            axes = [axes]
        
        for w_idx, wavelength in enumerate(wavelengths):
            # 确保在CPU上处理
            field = output_fields[w_idx]
            energy = torch.abs(field)**2
            energy_np = energy.detach().cpu().numpy()
            
            # 绘制完整的能量分布
            im = axes[w_idx].imshow(energy_np, cmap=self.energy_cmap, origin='lower')
            
            # 突出显示目标区域
            offset_x, offset_y = offsets[w_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            # 绘制目标区域边界
            detect_region = plt.Rectangle(
                (center_x - detect_size // 2, center_y - detect_size // 2),
                detect_size, detect_size,
                fill=False, edgecolor='red', linestyle='-', linewidth=3
            )
            axes[w_idx].add_patch(detect_region)
            
            # 设置标题
            wavelength_nm = wavelength * 1e9
            axes[w_idx].set_title(f"波长: {wavelength_nm:.0f}nm 目标区域")
            axes[w_idx].set_xlabel('X (像素)')
            axes[w_idx].set_ylabel('Y (像素)')
            
            # 添加颜色条
            cbar = fig.colorbar(im, ax=axes[w_idx])
            cbar.set_label('能量')
        
        plt.suptitle("目标区域能量分布", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_wavelength_separation_metrics(self, separation_metrics, save_path):
        """绘制波长分离性能指标"""
        # 过滤出波长相关的指标（排除'overall'）
        wavelength_metrics = {k: v for k, v in separation_metrics.items() if k != 'overall'}
        
        if not wavelength_metrics:
            print("没有找到波长相关的指标数据")
            return
        
        wavelengths = list(wavelength_metrics.keys())
        efficiencies = [metrics['efficiency'] for metrics in wavelength_metrics.values()]
        crosstalks = [metrics['avg_crosstalk'] for metrics in wavelength_metrics.values()]
        snrs = [metrics['snr'] for metrics in wavelength_metrics.values()]
        contrasts = [metrics['contrast'] for metrics in wavelength_metrics.values()]

        # 创建2x2的子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 效率图
        bars1 = ax1.bar(wavelengths, efficiencies, color='blue', alpha=0.7)
        ax1.set_title('目标区域效率', fontsize=14, fontweight='bold')
        ax1.set_ylabel('效率', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, efficiencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 串扰图
        bars2 = ax2.bar(wavelengths, crosstalks, color='red', alpha=0.7)
        ax2.set_title('平均串扰', fontsize=14, fontweight='bold')
        ax2.set_ylabel('串扰', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars2, crosstalks):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(crosstalks)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. 信噪比图
        bars3 = ax3.bar(wavelengths, snrs, color='green', alpha=0.7)
        ax3.set_title('信噪比 (SNR)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('SNR', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars3, snrs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(snrs)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 对比度图
        bars4 = ax4.bar(wavelengths, contrasts, color='orange', alpha=0.7)
        ax4.set_title('对比度', fontsize=14, fontweight='bold')
        ax4.set_ylabel('对比度', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars4, contrasts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(contrasts)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 调整x轴标签
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('波长', fontsize=12)
        
        # 添加整体性能信息
        if 'overall' in separation_metrics:
            overall = separation_metrics['overall']
            fig.suptitle(f'波长分离性能指标\n'
                        f'平均效率: {overall["avg_efficiency"]:.3f}, '
                        f'平均串扰: {overall["avg_crosstalk"]:.3f}, '
                        f'分离比率: {overall["separation_ratio"]:.1f}',
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle('波长分离性能指标', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 为标题留出空间
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"波长分离性能指标图已保存到: {save_path}")

