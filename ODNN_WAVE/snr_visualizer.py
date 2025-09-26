import numpy as np
import matplotlib.pyplot as plt
from base_visualizer import BaseVisualizer

class SNRVisualizer(BaseVisualizer):
    """信噪比专用可视化器"""
    
    def calculate_signal_noise_ratio(self, field_data, target_region_ratio=0.25):
        """计算目标区域和背景区域的信噪比"""
        if np.iscomplexobj(field_data):
            intensity = np.abs(field_data)**2
        else:
            intensity = np.abs(field_data)**2
        
        if intensity.ndim > 2:
            intensity = np.sum(intensity, axis=tuple(range(intensity.ndim-2)))
        
        if intensity.ndim != 2:
            return {'snr_db': 0, 'signal_power': 0, 'noise_power': 0, 'contrast_ratio': 1}
        
        H, W = intensity.shape
        max_intensity = np.max(intensity)
        if max_intensity <= 0:
            return {'snr_db': 0, 'signal_power': 0, 'noise_power': 0, 'contrast_ratio': 1}
        
        intensity_norm = intensity / max_intensity
        
        # 基于峰值位置的目标区域定义
        peak_pos = np.unravel_index(np.argmax(intensity), intensity.shape)
        peak_y, peak_x = peak_pos
        
        # 计算目标区域半径
        target_area = H * W * target_region_ratio
        target_radius = int(np.sqrt(target_area / np.pi))
        target_radius = max(target_radius, min(H, W) // 8)
        
        # 创建目标区域掩码
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        target_mask = ((y_grid - peak_y)**2 + (x_grid - peak_x)**2) <= target_radius**2
        
        # 基于强度阈值的自适应目标区域
        threshold = np.percentile(intensity_norm.flatten(), 90)
        adaptive_target_mask = intensity_norm >= threshold
        
        if np.sum(adaptive_target_mask) > 0.05 * H * W:
            final_target_mask = adaptive_target_mask
        else:
            final_target_mask = target_mask
        
        background_mask = ~final_target_mask
        
        signal_region = intensity_norm[final_target_mask]
        noise_region = intensity_norm[background_mask]
        
        if len(signal_region) == 0 or len(noise_region) == 0:
            return {'snr_db': 0, 'signal_power': 0, 'noise_power': 0, 'contrast_ratio': 1}
        
        signal_power = np.mean(signal_region)
        noise_power = np.mean(noise_region)
        
        snr_linear = signal_power / (noise_power + 1e-10)
        snr_db = 10 * np.log10(snr_linear + 1e-10)
        
        contrast_ratio = signal_power / (noise_power + 1e-10)
        
        peak_signal = np.max(signal_region)
        peak_snr_linear = peak_signal / (noise_power + 1e-10)
        peak_snr_db = 10 * np.log10(peak_snr_linear + 1e-10)
        
        return {
            'snr_db': snr_db,
            'peak_snr_db': peak_snr_db,
            'signal_power': signal_power,
            'noise_power': noise_power,
            'contrast_ratio': contrast_ratio,
            'signal_region_size': len(signal_region),
            'background_region_size': len(noise_region),
            'target_mask': final_target_mask,
            'background_mask': background_mask
        }

    def create_snr_only_visualization(self, snr_data, config, num_layer_options, 
                                save_path=None, title_suffix=""):
        """创建SNR独立可视化"""
        if not snr_data:
            print("❌ 没有SNR数据")
            return None

        print("🎨 创建SNR独立可视化...")

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'SNR Analysis Dashboard{title_suffix}', 
                    fontsize=18, fontweight='bold', y=0.95)

        self._create_snr_performance_bar_chart(axes[0, 0], snr_data, config, num_layer_options)
        self._create_snr_heatmap(axes[0, 1], snr_data, config, num_layer_options)
        self._create_snr_distribution_histogram(axes[1, 0], snr_data, config)
        self._create_contrast_analysis(axes[1, 1], snr_data, config, num_layer_options)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        if save_path is None:
            save_path = os.path.join(config.save_dir, f'snr_analysis{title_suffix}.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"✅ SNR分析已保存: {save_path}")
        return fig

    def _create_snr_performance_bar_chart(self, ax, snr_data, config, num_layer_options):
        """SNR性能柱状图"""
        modes = list(range(config.num_modes))
        bar_width = 0.25
        x_positions = np.arange(len(num_layer_options))
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for mode_idx in modes:
            snr_values = []
            for layers in num_layer_options:
                mode_layer_values = []
                for key, data in snr_data.items():
                    if self._match_config(key, layers, mode_idx):
                        if 'snr_db' in data:
                            snr_score = max(0, min(1, data['snr_db'] / 20.0))
                            mode_layer_values.append(snr_score)
                
                avg_snr = np.mean(mode_layer_values) if mode_layer_values else 0
                snr_values.append(avg_snr)
            
            bars = ax.bar(x_positions + mode_idx * bar_width, snr_values,
                        bar_width, label=f'Mode {mode_idx+1}',
                        color=colors[mode_idx], alpha=0.8, edgecolor='black')
            
            for bar, value in zip(bars, snr_values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel('SNR Score (0-1)', fontsize=12, fontweight='bold')
        ax.set_title('SNR Performance by Layers and Modes', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels([f'{layers}L' for layers in num_layer_options])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)

    def _match_config(self, key, layers, mode_idx):
        """匹配配置（层数和模式）"""
        return (f'layers{layers}' in key or f'L{layers}_' in key) and \
               (f'mode{mode_idx+1}' in key or f'_M{mode_idx+1}_' in key)
