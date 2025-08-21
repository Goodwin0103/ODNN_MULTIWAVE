from matplotlib import pyplot as plt
import numpy as np
import os
import csv
import json
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class Visualizer:
    def __init__(self, config):
        self.config = config
        
        # 设置英文字体和样式
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        
        # 设置颜色主题
        self.colors = {
            'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'gradient': plt.cm.viridis,
            'heatmap': 'RdYlBu_r',
            'intensity': 'hot'
        }
    
    def organize_visibility_by_mode(self, results, config, num_layer_options):
        """
        Organize visibility data by mode: each mode's performance across different wavelengths and layer numbers
        
        Based on debug info, data structure is:
        results['visibility'][layer_idx] = [mode0_vis, mode1_vis, mode2_vis]
        Each value corresponds to the visibility of that mode at that layer number (450nm wavelength only)
        """
        print("📊 Organizing visibility data by mode...")
        
        num_modes = config.num_modes
        num_wavelengths = len(config.wavelengths)
        vis_data = results['visibility']
        
        print(f"Configuration: {num_modes} modes, {num_wavelengths} wavelengths, {len(num_layer_options)} layer options")
        print(f"Raw data structure: {[len(layer_data) for layer_data in vis_data]}")
        
        # Check data structure
        if len(vis_data[0]) == num_modes:
            print("✅ Detected: Single wavelength data (450nm), others set to 0")
            return self._organize_single_wavelength_data(vis_data, num_modes, num_wavelengths, num_layer_options)
        elif len(vis_data[0]) == num_modes * num_wavelengths:
            print("✅ Detected: Complete multi-wavelength data")
            return self._organize_multi_wavelength_data(vis_data, num_modes, num_wavelengths, num_layer_options)
        else:
            print("⚠ Data structure mismatch, recalculating from weights_pred")
            return self._recalculate_from_weights(results, config, num_layer_options)
    
    def _organize_single_wavelength_data(self, vis_data, num_modes, num_wavelengths, num_layer_options):
        """Handle single wavelength data (current situation)"""
        visibility_by_mode = []
        
        for mode_idx in range(num_modes):
            mode_data = []
            for layer_idx, num_layers in enumerate(num_layer_options):
                wavelength_data = []
                for wave_idx in range(num_wavelengths):
                    if wave_idx == 0:  # 450nm has data
                        vis_value = float(vis_data[layer_idx][mode_idx])
                    else:  # 550nm, 650nm set to 0
                        vis_value = 0.0
                    wavelength_data.append(vis_value)
                mode_data.append(wavelength_data)
            visibility_by_mode.append(mode_data)
        
        return visibility_by_mode
    
    def _organize_multi_wavelength_data(self, vis_data, num_modes, num_wavelengths, num_layer_options):
        """Handle complete multi-wavelength data"""
        visibility_by_mode = []
        
        for mode_idx in range(num_modes):
            mode_data = []
            for layer_idx, num_layers in enumerate(num_layer_options):
                wavelength_data = []
                for wave_idx in range(num_wavelengths):
                    vis_idx = mode_idx * num_wavelengths + wave_idx
                    vis_value = float(vis_data[layer_idx][vis_idx])
                    wavelength_data.append(vis_value)
                mode_data.append(wavelength_data)
            visibility_by_mode.append(mode_data)
        
        return visibility_by_mode
    
    def _recalculate_from_weights(self, results, config, num_layer_options):
        """Recalculate visibility from weight data"""
        if 'weights_pred' not in results:
            print("❌ Error: No weights_pred data")
            return self._create_zero_data(config, num_layer_options)
        
        weights_data = results['weights_pred']
        num_modes = config.num_modes
        num_wavelengths = len(config.wavelengths)
        visibility_by_mode = []
        
        for mode_idx in range(num_modes):
            mode_data = []
            for layer_idx, num_layers in enumerate(num_layer_options):
                wavelength_data = []
                layer_weights = weights_data[layer_idx]  # (3, 3, 9)
                
                for wave_idx in range(num_wavelengths):
                    # Extract weights for this mode at this wavelength
                    mode_start = mode_idx * 3
                    mode_end = mode_start + 3
                    mode_weights = layer_weights[:, wave_idx, mode_start:mode_end]
                    avg_weights = np.mean(mode_weights, axis=0)
                    
                    # Calculate visibility
                    visibility = self._calculate_visibility(avg_weights)
                    wavelength_data.append(visibility)
                
                mode_data.append(wavelength_data)
            visibility_by_mode.append(mode_data)
        
        return visibility_by_mode
    
    def _calculate_visibility(self, weights):
        """Calculate visibility = (max - min) / (max + min)"""
        weights = np.array(weights)
        if len(weights) <= 1:
            return 0.0
        
        max_val, min_val = np.max(weights), np.min(weights)
        if max_val + min_val == 0:
            return 0.0
        return (max_val - min_val) / (max_val + min_val)
    
    def _create_zero_data(self, config, num_layer_options):
        """Create zero data structure"""
        num_modes = config.num_modes
        num_wavelengths = len(config.wavelengths)
        return [[[0.0 for _ in range(num_wavelengths)] 
                 for _ in num_layer_options] 
                for _ in range(num_modes)]
    
    def plot_visibility_by_mode(self, visibility_by_mode, num_layer_options, save_path=None):
        """Plot visibility comparison grouped by mode - Enhanced version"""
        print("🎨 Plotting visibility comparison by mode...")
        
        num_modes = len(visibility_by_mode)
        wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
        colors = self.colors['primary'][:len(wavelength_labels)]
        
        # Create larger figure
        fig, axes = plt.subplots(2, num_modes, figsize=(6 * num_modes, 10))
        if num_modes == 1:
            axes = axes.reshape(-1, 1)
        
        # Top row: Bar charts
        for mode_idx in range(num_modes):
            ax = axes[0, mode_idx]
            mode_data = np.array(visibility_by_mode[mode_idx])
            x = np.arange(len(num_layer_options))
            width = 0.25
            
            for wave_idx, (color, label) in enumerate(zip(colors, wavelength_labels)):
                values = mode_data[:, wave_idx]
                bars = ax.bar(x + wave_idx * width, values, width, 
                             label=label, color=color, alpha=0.8, 
                             edgecolor='white', linewidth=0.5)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.001:  # Only show meaningful values
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', 
                               fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Number of Layers', fontsize=12)
            ax.set_ylabel('Visibility', fontsize=12)
            ax.set_title(f'Mode {mode_idx + 1} - Visibility Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels([f'{layers}' for layers in num_layer_options])
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            ax.set_ylim(0, 1.05)
            
            # Add best performance annotation
            max_val = np.max(mode_data)
            if max_val > 0:
                max_pos = np.unravel_index(np.argmax(mode_data), mode_data.shape)
                best_layer = num_layer_options[max_pos[0]]
                best_wl = wavelength_labels[max_pos[1]]
                ax.text(0.02, 0.98, f'Best: {best_layer}L@{best_wl}\n({max_val:.3f})', 
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                       fontsize=10, fontweight='bold')
        
        # Bottom row: Heatmaps
        for mode_idx in range(num_modes):
            ax = axes[1, mode_idx]
            mode_data = np.array(visibility_by_mode[mode_idx])
            
            # Create heatmap
            im = ax.imshow(mode_data.T, cmap=self.colors['heatmap'], aspect='auto', 
                          vmin=0, vmax=1, origin='lower')
            
            # Set axes
            ax.set_xticks(range(len(num_layer_options)))
            ax.set_xticklabels([f'{layers}' for layers in num_layer_options])
            ax.set_yticks(range(len(wavelength_labels)))
            ax.set_yticklabels(wavelength_labels)
            ax.set_xlabel('Number of Layers', fontsize=12)
            ax.set_ylabel('Wavelength', fontsize=12)
            ax.set_title(f'Mode {mode_idx + 1} - Heatmap', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i in range(len(num_layer_options)):
                for j in range(len(wavelength_labels)):
                    value = mode_data[i, j]
                    if value > 0.001:
                        text_color = 'white' if value < 0.5 else 'black'
                        ax.text(i, j, f'{value:.3f}', ha='center', va='center',
                               color=text_color, fontsize=10, fontweight='bold')
            
            # Add colorbar
            if mode_idx == num_modes - 1:  # Only add colorbar to last subplot
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Visibility', rotation=270, labelpad=15, fontsize=12)
        
        plt.suptitle('Detailed Visibility Analysis by Mode', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Chart saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_visibility_comparison_by_mode_wavelength(self, visibility_by_mode, num_layer_options, save_path=None):
        """Plot mode-wavelength matrix visibility chart - Enhanced version"""
        print("🎨 Plotting mode-wavelength matrix visibility chart...")
        
        num_modes = len(visibility_by_mode)
        num_wavelengths = len(visibility_by_mode[0][0])
        wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
        
        # Create more elegant layout
        fig = plt.figure(figsize=(5*num_wavelengths, 4*num_modes + 2))
        gs = fig.add_gridspec(num_modes + 1, num_wavelengths, 
                             height_ratios=[3]*num_modes + [0.3], 
                             hspace=0.3, wspace=0.3)
        
        x = np.arange(len(num_layer_options))
        
        # Main mode-wavelength grid
        for mode_idx in range(num_modes):
            for wl_idx in range(num_wavelengths):
                ax = fig.add_subplot(gs[mode_idx, wl_idx])
                
                # Get data
                vis_data = [visibility_by_mode[mode_idx][layer_idx][wl_idx] 
                           for layer_idx in range(len(num_layer_options))]
                
                # Plot bar chart with gradient colors
                colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(vis_data)))
                bars = ax.bar(x, vis_data, width=0.6, color=colors_grad, 
                             alpha=0.8, edgecolor='white', linewidth=1)
                
                # Add value labels
                for bar, val in zip(bars, vis_data):
                    height = bar.get_height()
                    if height > 0.001:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{height:.3f}', ha='center', va='bottom', 
                               fontsize=10, fontweight='bold')
                
                # Beautify settings
                if mode_idx == 0:
                    ax.set_title(f'{wavelength_labels[wl_idx]}', fontsize=14, fontweight='bold')
                if wl_idx == 0:
                    ax.set_ylabel(f'Mode {mode_idx+1}\nVisibility', fontsize=12, fontweight='bold')
                if mode_idx == num_modes-1:
                    ax.set_xlabel('Layers', fontsize=12)
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'{layers}' for layers in num_layer_options])
                else:
                    ax.set_xticks([])
                
                ax.set_ylim(0, 1.05)
                ax.grid(True, linestyle='--', alpha=0.3, axis='y')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Highlight best value
                if vis_data:
                    max_idx = np.argmax(vis_data)
                    max_val = vis_data[max_idx]
                    if max_val > 0.001:
                        bars[max_idx].set_edgecolor('red')
                        bars[max_idx].set_linewidth(3)
        
        # Bottom statistics summary
        summary_ax = fig.add_subplot(gs[-1, :])
        summary_ax.axis('off')
        
        # Calculate statistics
        stats_text = "📊 Performance Summary:  "
        for mode_idx in range(num_modes):
            mode_data = np.array(visibility_by_mode[mode_idx])
            avg_vis = np.mean(mode_data[mode_data > 0])  # Only calculate non-zero values
            max_vis = np.max(mode_data)
            stats_text += f"Mode{mode_idx+1}: Avg={avg_vis:.3f}, Max={max_vis:.3f}  |  "
        
        summary_ax.text(0.5, 0.5, stats_text[:-5], ha='center', va='center', 
                       transform=summary_ax.transAxes, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('Mode-Wavelength Visibility Matrix Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Chart saved to: {save_path}")
        
        plt.show()
        return fig
    
    def print_visibility_summary(self, visibility_by_mode, num_layer_options):
        """Print visibility summary - Enhanced version"""
        print("\n" + "="*60)
        print("📊 Detailed Visibility Data Summary")
        print("="*60)
        
        wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
        
        for mode_idx, mode_data in enumerate(visibility_by_mode):
            print(f"\n🎯 Mode {mode_idx + 1}:")
            print("-" * 40)
            print("Layers\t" + "\t".join([f"{wl:>8}" for wl in wavelength_labels]) + "\tAverage")
            print("-" * (40 + len(wavelength_labels) * 9))
            
            for layer_idx, wavelength_data in enumerate(mode_data):
                avg_val = np.mean(wavelength_data)
                values_str = "\t".join([f"{val:8.4f}" for val in wavelength_data])
                print(f"{num_layer_options[layer_idx]:2d}\t{values_str}\t{avg_val:8.4f}")
            
            # Mode statistics
            mode_array = np.array(mode_data)
            print(f"\n   📈 Mode {mode_idx + 1} Statistics:")
            print(f"      Maximum: {np.max(mode_array):.4f}")
            print(f"      Minimum: {np.min(mode_array):.4f}")
            print(f"      Average: {np.mean(mode_array):.4f}")
            print(f"      Std Dev: {np.std(mode_array):.4f}")
        
        # Overall statistics
        print(f"\n🏆 Overall Statistics:")
        print("-" * 40)
        all_data = np.array(visibility_by_mode)
        for mode_idx in range(len(visibility_by_mode)):
            mode_avg = np.mean(all_data[mode_idx])
            print(f"   Mode {mode_idx + 1} Overall Average: {mode_avg:.4f}")
        
        overall_avg = np.mean(all_data)
        print(f"   🎯 System Overall Average: {overall_avg:.4f}")
        
        # Best configuration recommendations
        print(f"\n💡 Best Configuration Recommendations:")
        print("-" * 40)
        for mode_idx, mode_data in enumerate(visibility_by_mode):
            mode_array = np.array(mode_data)
            best_pos = np.unravel_index(np.argmax(mode_array), mode_array.shape)
            best_layer = num_layer_options[best_pos[0]]
            best_wl = wavelength_labels[best_pos[1]]
            best_val = mode_array[best_pos]
            print(f"   Mode {mode_idx + 1}: {best_layer} layers @ {best_wl} (Visibility: {best_val:.4f})")
    
    def save_visibility_data(self, visibility_by_mode, num_layer_options, save_path):
        """Save visibility data to CSV - Enhanced version"""
        print("💾 Saving visibility data to CSV...")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
        
        # Save detailed data
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write metadata
            writer.writerow(['# Multi-mode Multi-wavelength Visibility Data'])
            writer.writerow(['# Generation Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(['# Number of Modes:', len(visibility_by_mode)])
            writer.writerow(['# Number of Wavelengths:', len(wavelength_labels)])
            writer.writerow(['# Layer Configuration:', ' '.join(map(str, num_layer_options))])
            writer.writerow([])  # Empty line
            
            # Write header
            header = ['Mode', 'Layers'] + wavelength_labels + ['Average', 'Maximum', 'Minimum', 'Std Dev']
            writer.writerow(header)
            
            # Write data
            for mode_idx, mode_data in enumerate(visibility_by_mode):
                for layer_idx, wavelength_data in enumerate(mode_data):
                    avg_val = np.mean(wavelength_data)
                    max_val = np.max(wavelength_data)
                    min_val = np.min(wavelength_data)
                    std_val = np.std(wavelength_data)
                    
                    row = [f'Mode_{mode_idx+1}', num_layer_options[layer_idx]] + \
                          [f'{val:.6f}' for val in wavelength_data] + \
                          [f'{avg_val:.6f}', f'{max_val:.6f}', f'{min_val:.6f}', f'{std_val:.6f}']
                    writer.writerow(row)
        
        print(f"✅ Visibility data saved to: {save_path}")
        
        # Also save JSON format
        json_path = save_path.replace('.csv', '.json')
        json_data = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'num_modes': len(visibility_by_mode),
                'num_wavelengths': len(wavelength_labels),
                'wavelengths': wavelength_labels,
                'layer_options': num_layer_options
            },
            'data': {}
        }
        
        for mode_idx, mode_data in enumerate(visibility_by_mode):
            mode_key = f'mode_{mode_idx+1}'
            json_data['data'][mode_key] = {}
            for layer_idx, wavelength_data in enumerate(mode_data):
                layer_key = f'{num_layer_options[layer_idx]}layers'
                json_data['data'][mode_key][layer_key] = {
                    'wavelength_data': wavelength_data,
                    'statistics': {
                        'mean': float(np.mean(wavelength_data)),
                        'max': float(np.max(wavelength_data)),
                        'min': float(np.min(wavelength_data)),
                        'std': float(np.std(wavelength_data))
                    }
                }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON format data saved to: {json_path}")

    # ===== Cross Matrix Analysis Methods - Enhanced version =====
    
    def plot_cross_matrix_comparison(self, all_results, save_path=None):
        """Plot Cross Matrix comparison for all models - Enhanced version"""
        print("🎨 Plotting Cross Matrix comparison...")
        
        if not all_results:
            print("❌ No Cross Matrix results to plot")
            return
        
        layer_numbers = sorted(all_results.keys())
        separation_qualities = [all_results[n]['separation_quality'] for n in layer_numbers]
        avg_snrs = [np.mean(all_results[n]['avg_snrs']) for n in layer_numbers]
        
        # Create more elegant comparison chart
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 1], 
                             hspace=0.3, wspace=0.3)
        
        # 1. Separation quality comparison
        ax1 = fig.add_subplot(gs[0, 0])
        colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(layer_numbers)))
        bars1 = ax1.bar(range(len(layer_numbers)), separation_qualities, 
                        color=colors1, alpha=0.8, edgecolor='navy', linewidth=1.5)
        
        # Beautify bar chart
        for i, (bar, val) in enumerate(zip(bars1, separation_qualities)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
            
            # Highlight best value
            if val == max(separation_qualities):
                bar.set_edgecolor('red')
                bar.set_linewidth(3)
                ax1.annotate('Best', xy=(i, height), xytext=(i, height + 0.05),
                           ha='center', fontsize=10, fontweight='bold', color='red',
                           arrowprops=dict(arrowstyle='->', color='red'))
        
        ax1.set_xlabel('Number of Layers', fontsize=12)
        ax1.set_ylabel('Mode Separation Quality', fontsize=12)
        ax1.set_title('🎯 Mode Separation Quality Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(layer_numbers)))
        ax1.set_xticklabels([f'{n}L' for n in layer_numbers])
        ax1.set_ylim(0, max(separation_qualities) * 1.15)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Average SNR comparison
        ax2 = fig.add_subplot(gs[0, 1])
        colors2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(layer_numbers)))
        bars2 = ax2.bar(range(len(layer_numbers)), avg_snrs, 
                        color=colors2, alpha=0.8, edgecolor='darkred', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars2, avg_snrs)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_snrs)*0.02,
                    f'{val:.2f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
            
            # Highlight best value
            if val == max(avg_snrs):
                bar.set_edgecolor('blue')
                bar.set_linewidth(3)
                ax2.annotate('Best', xy=(i, height), xytext=(i, height + max(avg_snrs)*0.1),
                           ha='center', fontsize=10, fontweight='bold', color='blue',
                           arrowprops=dict(arrowstyle='->', color='blue'))
        
        ax2.set_xlabel('Number of Layers', fontsize=12)
        ax2.set_ylabel('Average SNR', fontsize=12)
        ax2.set_title('📡 Average SNR Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(layer_numbers)))
        ax2.set_xticklabels([f'{n}L' for n in layer_numbers])
        ax2.set_ylim(0, max(avg_snrs) * 1.2)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. Comprehensive performance radar chart
        ax3 = fig.add_subplot(gs[0, 2], projection='polar')
        
        # Normalize data for radar chart
        norm_sep = np.array(separation_qualities) / max(separation_qualities) if max(separation_qualities) > 0 else np.zeros_like(separation_qualities)
        norm_snr = np.array(avg_snrs) / max(avg_snrs) if max(avg_snrs) > 0 else np.zeros_like(avg_snrs)
        
        # Set angles
        angles = np.linspace(0, 2 * np.pi, len(layer_numbers), endpoint=False)
        
        # Plot radar chart
        ax3.plot(angles, norm_sep, 'o-', linewidth=2, label='Separation', color='blue', alpha=0.7)
        ax3.fill(angles, norm_sep, alpha=0.25, color='blue')
        ax3.plot(angles, norm_snr, 's-', linewidth=2, label='SNR', color='red', alpha=0.7)
        ax3.fill(angles, norm_snr, alpha=0.25, color='red')
        
        # Set labels
        
        # Set labels
        ax3.set_xticks(angles)
        ax3.set_xticklabels([f'{n}L' for n in layer_numbers])
        ax3.set_ylim(0, 1)
        ax3.set_title('🎯 Comprehensive Performance Radar', fontsize=14, fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax3.grid(True)
        
        # 4. Bottom statistics table
        stats_ax = fig.add_subplot(gs[1, :])
        stats_ax.axis('off')
        
        # Create statistics table
        table_data = []
        for i, layer_num in enumerate(layer_numbers):
            table_data.append([
                f'{layer_num}L',
                f'{separation_qualities[i]:.4f}',
                f'{avg_snrs[i]:.3f}',
                f'{separation_qualities[i] * avg_snrs[i]:.4f}',  # Composite score
                '⭐' if separation_qualities[i] == max(separation_qualities) else '',
                '🏆' if avg_snrs[i] == max(avg_snrs) else ''
            ])
        
        table = stats_ax.table(cellText=table_data,
                              colLabels=['Model', 'Separation', 'Avg SNR', 'Composite', 'Best Sep', 'Best SNR'],
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Beautify table
        table[(0, 0)].set_facecolor('#4CAF50')
        table[(0, 1)].set_facecolor('#2196F3')
        table[(0, 2)].set_facecolor('#FF9800')
        table[(0, 3)].set_facecolor('#9C27B0')
        table[(0, 4)].set_facecolor('#F44336')
        table[(0, 5)].set_facecolor('#607D8B')
        
        for i in range(6):
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle('Cross Matrix Performance Comprehensive Analysis', fontsize=18, fontweight='bold', y=0.95)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Cross Matrix comparison chart saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_snr_heatmap(self, all_results, save_path=None):
        """Plot SNR heatmap - Enhanced version"""
        print("🎨 Plotting SNR heatmap...")
        
        if not all_results:
            print("❌ No SNR results to plot")
            return
        
        layer_numbers = sorted(all_results.keys())
        wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
        
        # Create SNR matrix [layers, modes, wavelengths]
        num_layers = len(layer_numbers)
        num_modes = self.config.num_modes
        num_wavelengths = len(self.config.wavelengths)
        
        # Create more elegant layout
        fig = plt.figure(figsize=(6*num_modes, 8))
        gs = fig.add_gridspec(3, num_modes, height_ratios=[3, 3, 1], hspace=0.4, wspace=0.3)
        
        # Top row: SNR heatmap for each mode
        for mode_idx in range(num_modes):
            ax = fig.add_subplot(gs[0, mode_idx])
            
            # Build SNR matrix for this mode [layers, wavelengths]
            snr_matrix = np.zeros((num_layers, num_wavelengths))
            
            for i, layer_num in enumerate(layer_numbers):
                if 'snr_matrix' in all_results[layer_num]:
                    snr_data = all_results[layer_num]['snr_matrix']
                    if mode_idx < snr_data.shape[0]:
                        snr_matrix[i, :] = snr_data[mode_idx, :]
            
            # Plot heatmap
            im = ax.imshow(snr_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
            ax.set_title(f'🎯 Mode {mode_idx+1} SNR Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Wavelength', fontsize=12)
            if mode_idx == 0:
                ax.set_ylabel('Number of Layers', fontsize=12)
            ax.set_xticks(range(num_wavelengths))
            ax.set_xticklabels(wavelength_labels)
            ax.set_yticks(range(num_layers))
            ax.set_yticklabels([f'{n}L' for n in layer_numbers])
            
            # Add value labels and borders
            for i in range(num_layers):
                for j in range(num_wavelengths):
                    value = snr_matrix[i, j]
                    text_color = 'white' if value < np.max(snr_matrix) * 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha="center", va="center", 
                           color=text_color, fontsize=10, fontweight='bold')
                    
                    # Highlight maximum value
                    if value == np.max(snr_matrix):
                        rect = patches.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                                               linewidth=3, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('SNR', rotation=270, labelpad=15, fontsize=11)
        
        # Middle row: Layer comparison bar charts
        for mode_idx in range(num_modes):
            ax = fig.add_subplot(gs[1, mode_idx])
            
            # Calculate average SNR for each layer
            avg_snrs_by_layer = []
            for i, layer_num in enumerate(layer_numbers):
                if 'snr_matrix' in all_results[layer_num]:
                    snr_data = all_results[layer_num]['snr_matrix']
                    if mode_idx < snr_data.shape[0]:
                        avg_snr = np.mean(snr_data[mode_idx, :])
                        avg_snrs_by_layer.append(avg_snr)
                    else:
                        avg_snrs_by_layer.append(0)
                else:
                    avg_snrs_by_layer.append(0)
            
            # Plot bar chart
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(layer_numbers)))
            bars = ax.bar(range(len(layer_numbers)), avg_snrs_by_layer, 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, val in zip(bars, avg_snrs_by_layer):
                height = bar.get_height()
                if height > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(avg_snrs_by_layer)*0.02,
                           f'{height:.2f}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
            
            ax.set_title(f'📊 Mode {mode_idx+1} Average SNR Comparison', fontsize=12, fontweight='bold')
            ax.set_xlabel('Number of Layers', fontsize=11)
            if mode_idx == 0:
                ax.set_ylabel('Average SNR', fontsize=11)
            ax.set_xticks(range(len(layer_numbers)))
            ax.set_xticklabels([f'{n}L' for n in layer_numbers])
            ax.grid(True, alpha=0.3, axis='y')
        
        # Bottom: Overall statistics
        stats_ax = fig.add_subplot(gs[2, :])
        stats_ax.axis('off')
        
        # Create statistics text
        stats_text = "📈 SNR Statistics Summary:\n"
        for mode_idx in range(num_modes):
            mode_snrs = []
            for layer_num in layer_numbers:
                if 'snr_matrix' in all_results[layer_num]:
                    snr_data = all_results[layer_num]['snr_matrix']
                    if mode_idx < snr_data.shape[0]:
                        mode_snrs.extend(snr_data[mode_idx, :])
            
            if mode_snrs:
                avg_snr = np.mean(mode_snrs)
                max_snr = np.max(mode_snrs)
                min_snr = np.min(mode_snrs)
                stats_text += f"Mode{mode_idx+1}: Avg={avg_snr:.3f}, Max={max_snr:.3f}, Min={min_snr:.3f}   "
        
        stats_ax.text(0.5, 0.5, stats_text, ha='center', va='center', 
                     transform=stats_ax.transAxes, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.suptitle('Signal-to-Noise Ratio (SNR) Comprehensive Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ SNR heatmap saved to: {save_path}")
        
        plt.show()
        return fig
    
    def save_cross_matrix_summary(self, all_results, save_path):
        """Save Cross Matrix analysis summary - Enhanced version"""
        print("💾 Saving Cross Matrix analysis summary...")
        
        if not all_results:
            print("❌ No Cross Matrix results to save")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare detailed data
        summary_data = []
        layer_numbers = sorted(all_results.keys())
        
        for layer_num in layer_numbers:
            results = all_results[layer_num]
            
            row = {
                'layers': layer_num,
                'separation_quality': results['separation_quality'],
                'avg_snr': np.mean(results['avg_snrs']),
                'max_snr': np.max(results['avg_snrs']),
                'min_snr': np.min(results['avg_snrs']),
                'std_snr': np.std(results['avg_snrs']),
                'composite_score': results['separation_quality'] * np.mean(results['avg_snrs'])  # Composite score
            }
            
            # Add detailed SNR for each mode
            for mode_idx, snr in enumerate(results['avg_snrs']):
                row[f'mode_{mode_idx+1}_snr'] = snr
            
            # Add SNR for each wavelength (if SNR matrix exists)
            if 'snr_matrix' in results:
                snr_matrix = results['snr_matrix']
                wavelength_labels = [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]
                for wl_idx, wl_label in enumerate(wavelength_labels):
                    for mode_idx in range(snr_matrix.shape[0]):
                        if wl_idx < snr_matrix.shape[1]:
                            row[f'mode_{mode_idx+1}_{wl_label}_snr'] = snr_matrix[mode_idx, wl_idx]
            
            summary_data.append(row)
        
        # Save as CSV
        if summary_data:
            fieldnames = summary_data[0].keys()
            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write metadata comments
                f.write(f'# Cross Matrix Analysis Summary\n')
                f.write(f'# Generation Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'# Number of Models Analyzed: {len(summary_data)}\n')
                f.write(f'# Layer Configuration: {layer_numbers}\n')
                f.write(f'# Composite Score = Separation Quality × Average SNR\n')
                f.write('\n')
                
                writer.writeheader()
                writer.writerows(summary_data)
            
            print(f"✅ Cross Matrix summary saved to: {save_path}")
        
        # Save detailed JSON data
        json_path = save_path.replace('.csv', '_detailed.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json_data = {
                'metadata': {
                    'generation_time': datetime.now().isoformat(),
                    'num_models': len(all_results),
                    'layer_numbers': layer_numbers,
                    'analysis_type': 'Cross Matrix Performance Analysis'
                },
                'summary_statistics': {
                    'best_separation_quality': {
                        'value': max(r['separation_quality'] for r in all_results.values()),
                        'model': max(all_results.items(), key=lambda x: x[1]['separation_quality'])[0]
                    },
                    'best_avg_snr': {
                        'value': max(np.mean(r['avg_snrs']) for r in all_results.values()),
                        'model': max(all_results.items(), key=lambda x: np.mean(x[1]['avg_snrs']))[0]
                    },
                    'best_composite_score': {
                        'value': max(r['separation_quality'] * np.mean(r['avg_snrs']) for r in all_results.values()),
                        'model': max(all_results.items(), key=lambda x: x[1]['separation_quality'] * np.mean(x[1]['avg_snrs']))[0]
                    }
                },
                'detailed_results': {}
            }
            
            # Convert numpy arrays to lists for JSON serialization
            for layer_num, results in all_results.items():
                json_data['detailed_results'][str(layer_num)] = {
                    'separation_quality': float(results['separation_quality']),
                    'avg_snrs': [float(x) for x in results['avg_snrs']],
                    'cross_matrix': results['cross_matrix'].tolist() if 'cross_matrix' in results else [],
                    'snr_matrix': results['snr_matrix'].tolist() if 'snr_matrix' in results else [],
                    'composite_score': float(results['separation_quality'] * np.mean(results['avg_snrs']))
                }
            
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Detailed Cross Matrix data saved to: {json_path}")
        
        # Generate performance report
        self._generate_cross_matrix_report(all_results, save_path.replace('.csv', '_report.txt'))
    
    def _generate_cross_matrix_report(self, all_results, report_path):
        """Generate Cross Matrix performance report"""
        print("📝 Generating Cross Matrix performance report...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("Cross Matrix Performance Analysis Report")
        report_lines.append("="*80)
        report_lines.append(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of Models Analyzed: {len(all_results)}")
        report_lines.append("")
        
        layer_numbers = sorted(all_results.keys())
        
        # Performance rankings
        report_lines.append("🏆 Performance Rankings")
        report_lines.append("-"*50)
        
        # Ranking by separation quality
        sep_ranking = sorted(all_results.items(), key=lambda x: x[1]['separation_quality'], reverse=True)
        report_lines.append("📊 Separation Quality Ranking:")
        for rank, (layers, results) in enumerate(sep_ranking, 1):
            report_lines.append(f"  {rank}. {layers}-layer model: {results['separation_quality']:.4f}")
        report_lines.append("")
        
        # Ranking by average SNR
        snr_ranking = sorted(all_results.items(), key=lambda x: np.mean(x[1]['avg_snrs']), reverse=True)
        report_lines.append("📡 Average SNR Ranking:")
        for rank, (layers, results) in enumerate(snr_ranking, 1):
            avg_snr = np.mean(results['avg_snrs'])
            report_lines.append(f"  {rank}. {layers}-layer model: {avg_snr:.3f}")
        report_lines.append("")
        
        # Ranking by composite score
        composite_ranking = sorted(all_results.items(), 
                                 key=lambda x: x[1]['separation_quality'] * np.mean(x[1]['avg_snrs']), 
                                 reverse=True)
        report_lines.append("🎯 Composite Score Ranking:")
        for rank, (layers, results) in enumerate(composite_ranking, 1):
            composite_score = results['separation_quality'] * np.mean(results['avg_snrs'])
            report_lines.append(f"  {rank}. {layers}-layer model: {composite_score:.4f}")
        report_lines.append("")
        
        # Detailed analysis
        report_lines.append("📈 Detailed Performance Analysis")
        report_lines.append("-"*50)
        
        for layers in layer_numbers:
            results = all_results[layers]
            report_lines.append(f"\n🔍 {layers}-layer Model:")
            report_lines.append(f"   Separation Quality: {results['separation_quality']:.4f}")
            report_lines.append(f"   Average SNR: {np.mean(results['avg_snrs']):.3f}")
            report_lines.append(f"   SNR Range: {np.min(results['avg_snrs']):.3f} - {np.max(results['avg_snrs']):.3f}")
            report_lines.append(f"   SNR Std Dev: {np.std(results['avg_snrs']):.3f}")
            report_lines.append(f"   Composite Score: {results['separation_quality'] * np.mean(results['avg_snrs']):.4f}")
            
            # SNR details for each mode
            report_lines.append("   Mode SNR Details:")
            for mode_idx, snr in enumerate(results['avg_snrs']):
                report_lines.append(f"     Mode{mode_idx+1}: {snr:.3f}")
        
        # Recommendations
        report_lines.append(f"\n💡 Performance Recommendations")
        report_lines.append("-"*50)
        
        best_sep_model = sep_ranking[0][0]
        best_snr_model = snr_ranking[0][0]
        best_composite_model = composite_ranking[0][0]
        
        report_lines.append(f"• Best Separation Performance: {best_sep_model}-layer model")
        report_lines.append(f"• Best SNR Performance: {best_snr_model}-layer model")
        report_lines.append(f"• Best Overall Performance: {best_composite_model}-layer model")
        
        if best_composite_model == best_sep_model == best_snr_model:
            report_lines.append(f"• 🎉 {best_composite_model}-layer model excels in all metrics - Highly Recommended!")
        elif best_composite_model == best_sep_model:
            report_lines.append(f"• 🎯 {best_composite_model}-layer model best for separation quality and overall performance")
        elif best_composite_model == best_snr_model:
            report_lines.append(f"• 📡 {best_composite_model}-layer model best for SNR and overall performance")
        else:
            report_lines.append(f"• ⚖️ Choose based on application: {best_sep_model}L for separation priority, {best_snr_model}L for SNR priority")
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Also print key information
        print("\n" + "="*60)
        print("🎯 Cross Matrix Performance Analysis Key Results")
        print("="*60)
        print(f"🏆 Best Overall Performance: {best_composite_model}-layer model")
        print(f"📊 Best Separation Quality: {best_sep_model}-layer model")
        print(f"📡 Best SNR Performance: {best_snr_model}-layer model")
        print("="*60)
        
        print(f"✅ Detailed report saved to: {report_path}")

    def plot_training_losses(self, losses_list, layer_nums):
        """Plot training loss curves - Enhanced version"""
        print("📈 Plotting training loss curves...")
        
        if not losses_list or not layer_nums:
            print("⚠ No valid loss data")
            return
        
        # Create more elegant figure
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[2, 1], 
                             hspace=0.3, wspace=0.3)
        
        # Main plot: Loss curves
        ax_main = fig.add_subplot(gs[0, :])
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(losses_list)))
        
        valid_data = []
        for i, (losses, num_layers) in enumerate(zip(losses_list, layer_nums)):
            if losses is None or len(losses) == 0:
                print(f"⚠ {num_layers}-layer model has no loss data")
                continue
            
            # Ensure losses is a list or array
            if isinstance(losses, (list, np.ndarray)):
                epochs = range(1, len(losses) + 1)
                line = ax_main.plot(epochs, losses, 
                        label=f'{num_layers}-layer Model', 
                        color=colors[i], 
                        linewidth=3, 
                        marker='o', 
                        markersize=4,
                        alpha=0.8)
                
                # Annotate final loss value
                final_loss = losses[-1]
                ax_main.annotate(f'{final_loss:.4f}', 
                        xy=(len(losses), final_loss),
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=10, 
                        color=colors[i],
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
                
                valid_data.append((num_layers, losses, colors[i]))
            else:
                print(f"⚠ {num_layers}-layer model loss data format error: {type(losses)}")
        
        ax_main.set_xlabel('Training Epoch', fontsize=14)
        ax_main.set_ylabel('Loss Value', fontsize=14)
        ax_main.set_title('Training Loss Comparison Across Different Layer Models', fontsize=16, fontweight='bold')
        ax_main.legend(fontsize=12, loc='upper right')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_yscale('log')  # Use log scale
        
        # Beautify main plot
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        
        # Bottom left: Convergence speed comparison
        ax_conv = fig.add_subplot(gs[1, 0])
        
        if valid_data:
            conv_rates = []
            model_names = []
            colors_conv = []
            
            for num_layers, losses, color in valid_data:
                if len(losses) > 10:  # Ensure sufficient data points
                    # Calculate convergence speed (ratio of first 10% to last 10% losses)
                    early_loss = np.mean(losses[:max(1, len(losses)//10)])
                    late_loss = np.mean(losses[-max(1, len(losses)//10):])
                    conv_rate = early_loss / late_loss if late_loss > 0 else 1
                    
                    conv_rates.append(conv_rate)
                    model_names.append(f'{num_layers}L')
                    colors_conv.append(color)
            
            if conv_rates:
                bars = ax_conv.bar(range(len(conv_rates)), conv_rates, 
                                  color=colors_conv, alpha=0.8, edgecolor='black')
                
                # Add value labels
                for bar, rate in zip(bars, conv_rates):
                    height = bar.get_height()
                    ax_conv.text(bar.get_x() + bar.get_width()/2., height + max(conv_rates)*0.02,
                               f'{rate:.1f}x', ha='center', va='bottom', 
                               fontsize=10, fontweight='bold')
                
                ax_conv.set_xlabel('Model', fontsize=12)
                ax_conv.set_ylabel('Convergence Factor', fontsize=12)
                ax_conv.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
                ax_conv.set_xticks(range(len(model_names)))
                ax_conv.set_xticklabels(model_names)
                ax_conv.grid(True, alpha=0.3, axis='y')
        
        # Bottom right: Final loss comparison
        ax_final = fig.add_subplot(gs[1, 1])
        
        if valid_data:
            final_losses = []
            model_names = []
            colors_final = []
            
            for num_layers, losses, color in valid_data:
                final_losses.append(losses[-1])
                model_names.append(f'{num_layers}L')
                colors_final.append(color)
            
            bars = ax_final.bar(range(len(final_losses)), final_losses, 
                               color=colors_final, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, loss in zip(bars, final_losses):
                height = bar.get_height()
                ax_final.text(bar.get_x() + bar.get_width()/2., height + max(final_losses)*0.02,
                             f'{loss:.4f}', ha='center', va='bottom', 
                             fontsize=10, fontweight='bold')
            
            ax_final.set_xlabel('Model', fontsize=12)
            ax_final.set_ylabel('Final Loss', fontsize=12)
            ax_final.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
            ax_final.set_xticks(range(len(model_names)))
            ax_final.set_xticklabels(model_names, rotation=45)
            ax_final.grid(True, alpha=0.3, axis='y')
            ax_final.set_yscale('log')  # Use log scale
        
        plt.suptitle('Training Loss Comprehensive Analysis', fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_propagation_comparison(self, all_propagation_results, save_path=None):
        """Plot propagation simulation comparison for all models"""
        print("🎨 Plotting propagation simulation comparison...")
        
        num_models = len(all_propagation_results)
        num_wavelengths = len(self.config.wavelengths)
        
        fig, axes = plt.subplots(num_models, num_wavelengths, 
                                figsize=(5*num_wavelengths, 4*num_models))
        
        if num_models == 1:
            axes = axes.reshape(1, -1)
        if num_wavelengths == 1:
            axes = axes.reshape(-1, 1)
        
        for model_idx, (model_key, model_data) in enumerate(all_propagation_results.items()):
            for wl_idx, (wl_key, wl_data) in enumerate(model_data['propagation_results'].items()):
                ax = axes[model_idx, wl_idx]
                
                # Draw final field distribution and compute intensity
                final_field = wl_data['propagation_result']['fields'][-1]
                intensity = np.abs(final_field)**2
                
                im = ax.imshow(intensity, extent=[-self.Lx/2, self.Lx/2, -self.Ly/2, self.Ly/2],
                               cmap='hot', aspect='equal')
                
                ax.set_title(f'{model_key} - {wl_key}\nVisibility: {wl_data["coupling_analysis"]["visibility"]:.3f}')
                ax.set_xlabel('x (μm)')
                ax.set_ylabel('y (μm)')
                
                # 添加颜色条
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✅ 传播对比图保存至: {save_path}")
        
        plt.show()
        return fig

    def generate_propagation_summary(self, all_propagation_results):
        """生成传播仿真摘要报告"""
        print("📋 生成传播仿真摘要报告...")
        
        summary = {
            'models': {},
            'best_model': None,
            'best_visibility': 0,
            'wavelength_analysis': {}
        }
        
        # 分析每个模型
        for model_key, model_data in all_propagation_results.items():
            model_summary = {
                'num_layers': model_data['num_layers'],
                'avg_visibility': np.mean(model_data['visibility']),
                'wavelength_results': {}
            }
            
            total_bmp_vis = 0
            for wl_key, wl_data in model_data['propagation_results'].items():
                coupling_analysis = wl_data['coupling_analysis']
                model_summary['wavelength_results'][wl_key] = {
                    'bmp_visibility': coupling_analysis['visibility'],
                    'detector_responses': coupling_analysis['normalized_responses'],
                    'max_response': max(coupling_analysis['normalized_responses']),
                    'min_response': min(coupling_analysis['normalized_responses'])
                }
                total_bmp_vis += coupling_analysis['visibility']
            
            model_summary['avg_bmp_visibility'] = total_bmp_vis / len(model_data['propagation_results'])
            summary['models'][model_key] = model_summary
            
            # 更新最佳模型
            if model_summary['avg_bmp_visibility'] > summary['best_visibility']:
                summary['best_visibility'] = model_summary['avg_bmp_visibility']
                summary['best_model'] = model_key
        
        # 波长分析
        for wl_nm in [f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]:
            wl_visibilities = []
            for model_data in summary['models'].values():
                if wl_nm in model_data['wavelength_results']:
                    wl_visibilities.append(model_data['wavelength_results'][wl_nm]['bmp_visibility'])
            
            if wl_visibilities:
                summary['wavelength_analysis'][wl_nm] = {
                    'avg_visibility': np.mean(wl_visibilities),
                    'max_visibility': max(wl_visibilities),
                    'min_visibility': min(wl_visibilities),
                    'std_visibility': np.std(wl_visibilities)
                }
        
        return summary

    def save_propagation_data(self, all_propagation_results, save_path):
        """保存传播仿真数据"""
        print(f"💾 保存传播仿真数据至: {save_path}")
        
        # 准备可序列化的数据
        serializable_data = {}
        
        for model_key, model_data in all_propagation_results.items():
            serializable_data[model_key] = {
                'num_layers': model_data['num_layers'],
                'visibility': model_data['visibility'].tolist() if isinstance(model_data['visibility'], np.ndarray) else model_data['visibility'],
                'wavelength_results': {}
            }
            
            for wl_key, wl_data in model_data['propagation_results'].items():
                coupling_analysis = wl_data['coupling_analysis']
                serializable_data[model_key]['wavelength_results'][wl_key] = {
                    'bmp_visibility': float(coupling_analysis['visibility']),
                    'detector_responses': [float(r) for r in coupling_analysis['normalized_responses']],
                    'wavelength': float(wl_data['wavelength'])
                }
        
        # 保存为JSON文件
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 传播仿真数据已保存")
