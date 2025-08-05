import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

class SingleModeDualWavelengthDataGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # MMF parameters
        self.core_radius = 25e-6  # 25 μm core radius
        self.na = 0.22  # Numerical aperture
        
        # Load or generate MMF mode data
        self.mmf_mode_data = self.load_or_generate_mmf_modes()
        
    def load_or_generate_mmf_modes(self):
        """Load MMF mode data or generate if not available"""
        mmf_file_path = os.path.join(self.config.save_dir, 'mmf_modes.npz')
        
        if os.path.exists(mmf_file_path):
            print("Loading MMF mode data from file...")
            data = np.load(mmf_file_path)
            return {
                'fundamental_mode': torch.tensor(data['fundamental_mode'], dtype=torch.complex64, device=self.device)
            }
        else:
            print("MMF data file not found, generating Gaussian fundamental mode")
            return self.generate_gaussian_fundamental_mode()
    
    def generate_gaussian_fundamental_mode(self):
        """Generate a Gaussian approximation of the fundamental mode"""
        # Create coordinate grid
        x = torch.linspace(-self.config.field_size//2, self.config.field_size//2, 
                          self.config.field_size, device=self.device) * self.config.pixel_size
        y = torch.linspace(-self.config.field_size//2, self.config.field_size//2, 
                          self.config.field_size, device=self.device) * self.config.pixel_size
        X, Y = torch.meshgrid(x, y, indexing='ij')
        R = torch.sqrt(X**2 + Y**2)
        
        # Gaussian beam waist (approximation for fundamental mode)
        w0 = self.core_radius / 2  # Beam waist
        
        # Generate Gaussian fundamental mode
        fundamental_mode = torch.exp(-R**2 / w0**2).to(torch.complex64)
        
        # Normalize
        fundamental_mode = fundamental_mode / torch.sqrt(torch.sum(torch.abs(fundamental_mode)**2))
        
        return {'fundamental_mode': fundamental_mode}
    
    def generate_input_fields(self):
        """Generate input fields for both wavelengths"""
        input_fields = []
        
        for wl_idx, wavelength in enumerate(self.config.wavelengths):
            # Use fundamental mode as base
            base_field = self.mmf_mode_data['fundamental_mode'].clone()
            
            # Add wavelength-dependent phase variation
            phase_variation = torch.exp(1j * torch.rand(1, device=self.device) * 2 * np.pi)
            input_field = base_field * phase_variation
            
            # Add some random amplitude variation (±10%)
            amp_variation = 0.9 + 0.2 * torch.rand(1, device=self.device)
            input_field = input_field * amp_variation
            
            input_fields.append(input_field)
        
        return input_fields
    
    def generate_target_labels(self, input_fields):
        """Generate target intensity distributions"""
        target_labels = []
        
        for wl_idx, wavelength in enumerate(self.config.wavelengths):
            # Create target distribution (Gaussian spot at specified offset)
            target = torch.zeros(self.config.field_size, self.config.field_size, device=self.device)
            
            # Get target position
            offset_x, offset_y = self.config.offsets[wl_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            # Create coordinate grids
            x = torch.arange(self.config.field_size, device=self.device) - center_x
            y = torch.arange(self.config.field_size, device=self.device) - center_y
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Gaussian target spot
            sigma = self.config.detect_size / 4  # Spot size
            target = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
            
            # Normalize
            target = target / torch.sum(target)
            
            target_labels.append(target)
        
        return target_labels
    
    def visualize_separation_concept(self, save_path=None):
        """Visualize the wavelength separation concept"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Generate sample input fields
        input_fields = self.generate_input_fields()
        
        for wl_idx, wavelength in enumerate(self.config.wavelengths):
            wavelength_nm = int(wavelength * 1e9)
            
            # Input field intensity
            ax_input = axes[wl_idx, 0]
            input_intensity = torch.abs(input_fields[wl_idx])**2
            im_input = ax_input.imshow(input_intensity.cpu().numpy(), cmap='viridis')
            ax_input.set_title(f'输入场强度\n波长: {wavelength_nm}nm')
            ax_input.axis('off')
            plt.colorbar(im_input, ax=ax_input, fraction=0.046)
            
            # Target region visualization
            ax_target = axes[wl_idx, 1]
            # Create a visualization showing where this wavelength should go
            target_vis = torch.zeros(self.config.field_size, self.config.field_size, device=self.device)
            
            # Get target position
            offset_x, offset_y = self.config.offsets[wl_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            # Create target region
            x_start = max(0, center_x - self.config.detect_size // 2)
            x_end = min(self.config.field_size, center_x + self.config.detect_size // 2)
            y_start = max(0, center_y - self.config.detect_size // 2)
            y_end = min(self.config.field_size, center_y + self.config.detect_size // 2)
            
            target_vis[y_start:y_end, x_start:x_end] = 1.0
            
            im_target = ax_target.imshow(target_vis.cpu().numpy(), cmap='hot')
            ax_target.set_title(f'目标检测区域\n波长: {wavelength_nm}nm')
            ax_target.axis('off')
            plt.colorbar(im_target, ax=ax_target, fraction=0.046)
            
            # Conceptual output (ideal case)
            ax_output = axes[wl_idx, 2]
            # Create ideal output where energy is concentrated in target region
            ideal_output = torch.zeros(self.config.field_size, self.config.field_size, device=self.device)
            
            # Create Gaussian spot in target region
            x = torch.arange(self.config.field_size, device=self.device) - center_x
            y = torch.arange(self.config.field_size, device=self.device) - center_y
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            sigma = self.config.detect_size / 4
            ideal_output = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
            
            im_output = ax_output.imshow(ideal_output.cpu().numpy(), cmap='plasma')
            ax_output.set_title(f'理想输出分布\n波长: {wavelength_nm}nm')
            ax_output.axis('off')
            plt.colorbar(im_output, ax=ax_output, fraction=0.046)
        
        plt.suptitle('双波长分离概念图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Separation concept visualization saved to: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def visualize_detector_layout(self, save_path=None):
        """Visualize the detector layout and target regions"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create field background
        field_background = torch.zeros(self.config.field_size, self.config.field_size, device=self.device)
        ax.imshow(field_background.cpu().numpy(), cmap='gray', alpha=0.3)
        
        # Colors for different wavelengths
        colors = ['blue', 'red', 'green', 'orange']
        
        # Draw target regions for each wavelength
        for wl_idx, wavelength in enumerate(self.config.wavelengths):
            wavelength_nm = int(wavelength * 1e9)
            color = colors[wl_idx % len(colors)]
            
            # Get target position
            offset_x, offset_y = self.config.offsets[wl_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            # Draw target region rectangle
            rect = plt.Rectangle(
                (center_x - self.config.detect_size // 2, center_y - self.config.detect_size // 2),
                self.config.detect_size, self.config.detect_size,
                fill=True, facecolor=color, alpha=0.3, edgecolor=color, linewidth=3
            )
            ax.add_patch(rect)
            
            # Add wavelength label
            ax.text(center_x, center_y, f'{wavelength_nm}nm', 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            
            # Add arrow pointing to the region
            arrow_start_x = self.config.field_size // 2
            arrow_start_y = self.config.field_size // 2
            arrow_dx = offset_x * 0.7
            arrow_dy = offset_y * 0.7
            
            ax.annotate('', xy=(arrow_start_x + arrow_dx, arrow_start_y + arrow_dy),
                       xytext=(arrow_start_x, arrow_start_y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Draw field boundary
        field_boundary = plt.Rectangle((0, 0), self.config.field_size, self.config.field_size,
                                     fill=False, edgecolor='black', linestyle='--', linewidth=2)
        ax.add_patch(field_boundary)
        
        # Mark center
        ax.plot(self.config.field_size // 2, self.config.field_size // 2, 'k+', markersize=15, markeredgewidth=3)
        ax.text(self.config.field_size // 2 + 5, self.config.field_size // 2 + 5, '中心', fontsize=12)
        
        ax.set_xlim(0, self.config.field_size)
        ax.set_ylim(0, self.config.field_size)
        ax.set_xlabel('X (像素)', fontsize=12)
        ax.set_ylabel('Y (像素)', fontsize=12)
        ax.set_title('检测器布局和目标区域', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = []
        for wl_idx, wavelength in enumerate(self.config.wavelengths):
            wavelength_nm = int(wavelength * 1e9)
            color = colors[wl_idx % len(colors)]
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, 
                                               edgecolor=color, label=f'{wavelength_nm}nm'))
        ax.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detector layout visualization saved to: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def create_dataset(self, num_samples=1000):
        """Create dataset of input-target pairs"""
        dataset = []
        
        for _ in range(num_samples):
            # Generate input fields
            input_fields = self.generate_input_fields()
            
            # Generate corresponding target labels
            target_labels = self.generate_target_labels(input_fields)
            
            dataset.append({
                'input_fields': input_fields,
                'target_labels': target_labels
            })
        
        return dataset
    
    def create_dataloader(self, num_samples=1000, batch_size=4, shuffle=True):
        """Create DataLoader for training"""
        dataset = self.create_dataset(num_samples)
        
        class CustomDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        custom_dataset = CustomDataset(dataset)
        return DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)
    
    def visualize_data(self, save_path=None):
        """Visualize input data and target labels"""
        # Generate sample data
        input_fields = self.generate_input_fields()
        target_labels = self.generate_target_labels(input_fields)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for wl_idx, wl in enumerate(self.config.wavelengths):
            # Input field amplitude
            ax_amp = axes[wl_idx, 0]
            input_amp = torch.abs(input_fields[wl_idx])
            im_amp = ax_amp.imshow(input_amp.cpu().numpy(), cmap='viridis')
            ax_amp.set_title(f'Wavelength {int(wl*1e9)}nm\nInput Field Amplitude')
            ax_amp.axis('off')
            plt.colorbar(im_amp, ax=ax_amp)
            
            # Input field phase
            ax_phase = axes[wl_idx, 1]
            input_phase = torch.angle(input_fields[wl_idx])
            im_phase = ax_phase.imshow(input_phase.cpu().numpy(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
            ax_phase.set_title(f'Wavelength {int(wl*1e9)}nm\nInput Field Phase')
            ax_phase.axis('off')
            plt.colorbar(im_phase, ax=ax_phase)
            
            # Target intensity distribution
            ax_target = axes[wl_idx, 2]
            im_target = ax_target.imshow(target_labels[wl_idx].cpu().numpy(), cmap='hot')
            ax_target.set_title(f'Wavelength {int(wl*1e9)}nm\nTarget Distribution')
            ax_target.axis('off')
            plt.colorbar(im_target, ax=ax_target)
            
            # Target region overlay
            ax_overlay = axes[wl_idx, 3]
            input_intensity = torch.abs(input_fields[wl_idx])**2
            im_overlay = ax_overlay.imshow(input_intensity.cpu().numpy(), cmap='gray', alpha=0.7)
            
            # Add target region rectangle
            offset_x, offset_y = self.config.offsets[wl_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            rect = plt.Rectangle(
                (center_x - self.config.detect_size//2, center_y - self.config.detect_size//2),
                self.config.detect_size, self.config.detect_size,
                fill=False, edgecolor='red', linewidth=2
            )
            ax_overlay.add_patch(rect)
            ax_overlay.set_title(f'Wavelength {int(wl*1e9)}nm\nTarget Region Overlay')
            ax_overlay.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Data visualization saved to: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def visualize_mmf_modes(self, save_path=None):
        """Visualize MMF mode profiles"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Fundamental mode amplitude
        fundamental_mode = self.mmf_mode_data['fundamental_mode']
        
        ax_amp = axes[0]
        mode_amp = torch.abs(fundamental_mode)
        im_amp = ax_amp.imshow(mode_amp.cpu().numpy(), cmap='viridis')
        ax_amp.set_title('Fundamental Mode\nAmplitude')
        ax_amp.axis('off')
        plt.colorbar(im_amp, ax=ax_amp)
        
        # Fundamental mode phase
        ax_phase = axes[1]
        mode_phase = torch.angle(fundamental_mode)
        im_phase = ax_phase.imshow(mode_phase.cpu().numpy(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title('Fundamental Mode\nPhase')
        ax_phase.axis('off')
        plt.colorbar(im_phase, ax=ax_phase)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"MMF modes visualization saved to: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def get_mode_coupling_efficiency(self, output_field, target_mode):
        """Calculate mode coupling efficiency"""
        # Normalize fields
        output_norm = output_field / torch.sqrt(torch.sum(torch.abs(output_field)**2))
        target_norm = target_mode / torch.sqrt(torch.sum(torch.abs(target_mode)**2))
        
        # Calculate overlap integral
        overlap = torch.sum(torch.conj(output_norm) * target_norm)
        efficiency = torch.abs(overlap)**2
        
        return efficiency.item()
    
    def calculate_separation_metrics(self, output_fields):
        """Calculate wavelength separation performance metrics"""
        metrics = {}
        
        for wl_idx, wavelength in enumerate(self.config.wavelengths):
            field = output_fields[wl_idx]
            intensity = torch.abs(field)**2
            
            # Get target region
            offset_x, offset_y = self.config.offsets[wl_idx]
            center_x = self.config.field_size // 2 + offset_x
            center_y = self.config.field_size // 2 + offset_y
            
            x_start = center_x - self.config.detect_size // 2
            x_end = center_x + self.config.detect_size // 2
            y_start = center_y - self.config.detect_size // 2
            y_end = center_y + self.config.detect_size // 2
            
            # Calculate metrics
            total_power = torch.sum(intensity).item()
            target_power = torch.sum(intensity[y_start:y_end, x_start:x_end]).item()
            
            efficiency = target_power / total_power if total_power > 0 else 0
            
            # Calculate crosstalk to other wavelength regions
            crosstalk_powers = []
            for other_wl_idx in range(len(self.config.wavelengths)):
                if other_wl_idx != wl_idx:
                    other_offset_x, other_offset_y = self.config.offsets[other_wl_idx]
                    other_center_x = self.config.field_size // 2 + other_offset_x
                    other_center_y = self.config.field_size // 2 + other_offset_y
                    
                    other_x_start = other_center_x - self.config.detect_size // 2
                    other_x_end = other_center_x + self.config.detect_size // 2
                    other_y_start = other_center_y - self.config.detect_size // 2
                    other_y_end = other_center_y + self.config.detect_size // 2
                    
                    crosstalk_power = torch.sum(intensity[other_y_start:other_y_end, other_x_start:other_x_end]).item()
                    crosstalk_powers.append(crosstalk_power / total_power if total_power > 0 else 0)
            
            avg_crosstalk = np.mean(crosstalk_powers) if crosstalk_powers else 0
            
            metrics[f"wavelength_{int(wavelength*1e9)}nm"] = {
                'efficiency': efficiency,
                'crosstalk': avg_crosstalk,
                'extinction_ratio': efficiency / (avg_crosstalk + 1e-10)
            }
        
        return metrics
    
    def save_mmf_modes(self, save_path=None):
        """Save MMF mode data to file"""
        if save_path is None:
            save_path = os.path.join(self.config.save_dir, 'mmf_modes.npz')
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert to CPU numpy arrays for saving
        np.savez(save_path, 
                 fundamental_mode=self.mmf_mode_data['fundamental_mode'].cpu().numpy())
        print(f"MMF modes saved to: {save_path}")
    
    # Backward compatibility methods
    def generate_input_data(self):
        """Generate input data (compatibility method)"""
        return self.generate_input_fields()

# 为了向后兼容，保留SimpleDataGenerator类
class SimpleDataGenerator(SingleModeDualWavelengthDataGenerator):
    """Backward compatibility wrapper"""
    pass
