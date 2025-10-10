# -*- coding: utf-8 -*-
"""
Optical Field Propagation Simulation with Custom Model Selection
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import matplotlib.animation as animation
import json

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class CustomMaskLoader:
    """Custom Phase Mask Loader with Model Selection"""
    
    def __init__(self, results_dir="./results"):
        self.results_dir = Path(results_dir)
        self.available_models = self._scan_available_models()
    
    def _scan_available_models(self):
        """Scan all available model files"""
        print("🔍 Scanning available models...")
        
        model_files = list(self.results_dir.rglob("*.pth"))
        models_info = []
        
        for model_file in model_files:
            try:
                # Get file info
                file_size = model_file.stat().st_size / (1024*1024)  # MB
                mod_time = model_file.stat().st_mtime
                
                # Try to load and get basic info
                checkpoint = torch.load(model_file, map_location='cpu')
                
                info = {
                    'path': model_file,
                    'name': model_file.name,
                    'size_mb': file_size,
                    'modified_time': mod_time,
                    'keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else ['tensor'],
                    'loadable': True
                }
                
                # Try to extract phase mask info
                phase_masks = self._extract_phase_masks(checkpoint)
                if phase_masks is not None:
                    info['phase_mask_shape'] = list(phase_masks.shape)
                    info['num_layers'] = phase_masks.shape[0] if len(phase_masks.shape) >= 3 else 1
                else:
                    info['phase_mask_shape'] = None
                    info['num_layers'] = 0
                
                models_info.append(info)
                
            except Exception as e:
                models_info.append({
                    'path': model_file,
                    'name': model_file.name,
                    'size_mb': file_size,
                    'modified_time': mod_time,
                    'keys': [],
                    'loadable': False,
                    'error': str(e),
                    'phase_mask_shape': None,
                    'num_layers': 0
                })
        
        # Sort by modification time (newest first)
        models_info.sort(key=lambda x: x['modified_time'], reverse=True)
        
        print(f"✅ Found {len(models_info)} model files")
        return models_info
    
    def list_available_models(self):
        """List all available models for user selection"""
        print("\n📋 Available Models:")
        print("-" * 80)
        print(f"{'No.':<4} {'Name':<30} {'Layers':<8} {'Shape':<20} {'Size(MB)':<10} {'Status'}")
        print("-" * 80)
        
        for i, model in enumerate(self.available_models):
            status = "✅ OK" if model['loadable'] else "❌ Error"
            shape_str = str(model['phase_mask_shape']) if model['phase_mask_shape'] else "Unknown"
            
            print(f"{i+1:<4} {model['name'][:29]:<30} {model['num_layers']:<8} "
                  f"{shape_str[:19]:<20} {model['size_mb']:<10.1f} {status}")
        
        print("-" * 80)
        return len(self.available_models)
    
    def select_model(self, model_index=None):
        """Select a model by index or interactively"""
        if not self.available_models:
            print("❌ No models available")
            return None
        
        # If no index provided, show list and ask for selection
        if model_index is None:
            num_models = self.list_available_models()
            
            try:
                choice = input(f"\nSelect model (1-{num_models}) or press Enter for latest: ").strip()
                if choice == "":
                    model_index = 0  # Use latest (first in sorted list)
                else:
                    model_index = int(choice) - 1
            except (ValueError, KeyboardInterrupt):
                print("Using latest model...")
                model_index = 0
        
        # Validate index
        if model_index < 0 or model_index >= len(self.available_models):
            print(f"❌ Invalid model index. Using latest model.")
            model_index = 0
        
        selected_model = self.available_models[model_index]
        
        if not selected_model['loadable']:
            print(f"❌ Selected model is not loadable: {selected_model.get('error', 'Unknown error')}")
            return None
        
        print(f"\n✅ Selected Model:")
        print(f"   Name: {selected_model['name']}")
        print(f"   Layers: {selected_model['num_layers']}")
        print(f"   Shape: {selected_model['phase_mask_shape']}")
        print(f"   Size: {selected_model['size_mb']:.1f} MB")
        
        # Load the selected model
        return self._load_selected_model(selected_model)
    
    def _load_selected_model(self, model_info):
        """Load the selected model"""
        try:
            checkpoint = torch.load(model_info['path'], map_location='cpu')
            phase_masks = self._extract_phase_masks(checkpoint)
            
            if phase_masks is not None:
                return {
                    'phase_masks': phase_masks,
                    'model_info': model_info,
                    'num_layers': model_info['num_layers']
                }
            else:
                print("❌ Could not extract phase masks from selected model")
                return None
                
        except Exception as e:
            print(f"❌ Failed to load selected model: {e}")
            return None
    
    def _extract_phase_masks(self, checkpoint):
        """Extract phase masks from checkpoint"""
        possible_keys = ['phase_masks', 'model_state_dict', 'state_dict', 
                        'phase_mask', 'masks', 'model', 'phase_layers']
        
        phase_masks = None
        
        if isinstance(checkpoint, dict):
            for key in possible_keys:
                if key in checkpoint:
                    phase_masks = checkpoint[key]
                    break
        elif isinstance(checkpoint, torch.Tensor):
            phase_masks = checkpoint
        
        if phase_masks is not None:
            # Ensure 3D tensor [num_layers, height, width]
            if len(phase_masks.shape) == 2:
                phase_masks = phase_masks.unsqueeze(0)
            elif len(phase_masks.shape) == 4:
                # If 4D, take first batch
                phase_masks = phase_masks[0]
        
        return phase_masks

class CompatibleSimulator:
    """Compatible Simulator that works with your existing input data format"""
    
    def __init__(self, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        print(f"🔧 Simulator initialized, device: {self.device}")
    
    def simulate_propagation(self, phase_masks, input_fields, process_all_modes=True):
        """
        Simulate propagation with your existing input data format
        
        Args:
            phase_masks: Phase masks from selected model
            input_fields: Input fields from data_generator.generate_input_data()
                         Shape: [num_modes, num_wavelengths, height, width]
            process_all_modes: Whether to process all modes and wavelengths
        """
        print("🚀 Starting propagation simulation...")
        print(f"   Input fields shape: {input_fields.shape}")
        print(f"   Phase masks shape: {phase_masks.shape}")
        
        # Ensure tensors are on the same device
        phase_masks = phase_masks.to(self.device)
        input_fields = input_fields.to(self.device)
        
        num_modes = input_fields.shape[0]
        num_wavelengths = input_fields.shape[1]
        num_layers = phase_masks.shape[0]
        
        print(f"   Processing {num_modes} modes × {num_wavelengths} wavelengths × {num_layers} layers")
        
        all_results = []
        
        # Process each mode and wavelength combination
        for mode_idx in range(num_modes):
            for wl_idx in range(num_wavelengths):
                print(f"   Processing Mode {mode_idx+1}, Wavelength {wl_idx+1}...")
                
                # Get input field for this mode and wavelength
                input_field = input_fields[mode_idx, wl_idx]
                
                # Simulate step-by-step propagation
                propagation_steps = self._simulate_single_propagation(
                    phase_masks, input_field, mode_idx, wl_idx
                )
                
                # Calculate performance metrics
                final_field = propagation_steps['Detection_Plane']
                metrics = self._calculate_metrics(final_field)
                
                # Store result
                result = {
                    'mode_idx': mode_idx,
                    'wavelength_idx': wl_idx,
                    'propagation_steps': propagation_steps,
                    'focus_ratio': metrics['focus_ratio'],
                    'peak_intensity': metrics['peak_intensity'],
                    'beam_width': metrics['beam_width'],
                    'statistics': metrics
                }
                
                all_results.append(result)
        
        print(f"✅ Simulation completed, {len(all_results)} results generated")
        return all_results
    
    def _simulate_single_propagation(self, phase_masks, input_field, mode_idx, wl_idx):
        """Simulate single mode-wavelength propagation"""
        num_layers = phase_masks.shape[0]
        
        # Store each propagation step
        propagation_steps = {}
        current_field = input_field.clone()
        propagation_steps['Input_Field'] = current_field.clone()
        
        # Layer-by-layer propagation
        for layer_idx in range(num_layers):
            # Apply phase modulation
            phase_layer = phase_masks[layer_idx]
            modulated_field = current_field * torch.exp(1j * phase_layer)
            
            # Propagation distance
            if layer_idx < num_layers - 1:
                # Inter-layer propagation
                distance = 40e-6  # Default layer separation
                propagated_field = self._propagate_field(modulated_field, distance, wl_idx)
                propagation_steps[f'After_Layer_{layer_idx+1}'] = propagated_field.clone()
                current_field = propagated_field
            else:
                # Final propagation to detection plane
                distance = 150e-6  # Default detection distance
                final_field = self._propagate_field(modulated_field, distance, wl_idx)
                propagation_steps['Detection_Plane'] = final_field.clone()
                current_field = final_field
        
        return propagation_steps
    
    def _propagate_field(self, field, distance, wavelength_idx):
        """Propagate optical field using angular spectrum method"""
        # Default wavelengths
        wavelengths = [1310e-9, 1550e-9]
        wavelength = wavelengths[wavelength_idx] if wavelength_idx < len(wavelengths) else 1550e-9
        
        k = 2 * np.pi / wavelength
        
        # FFT-based propagation
        field_fft = torch.fft.fft2(field)
        
        size = field.shape[-1]
        pixel_size = 1e-6  # Default pixel size
        
        fx = torch.fft.fftfreq(size, d=pixel_size, device=self.device)
        fy = torch.fft.fftfreq(size, d=pixel_size, device=self.device)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        
        # Calculate propagation kernel
        kz_squared = k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2
        kz_squared = torch.clamp(kz_squared, min=0)
        kz = torch.sqrt(kz_squared)
        
        propagation_kernel = torch.exp(1j * kz * distance)
        propagated_fft = field_fft * propagation_kernel
        propagated_field = torch.fft.ifft2(propagated_fft)
        
        return propagated_field
    
    def _calculate_metrics(self, field):
        """Calculate performance metrics"""
        field_np = field.cpu().numpy()
        intensity = np.abs(field_np)**2
        
        # Basic metrics
        peak_intensity = np.max(intensity)
        total_power = np.sum(intensity)
        
        # Focus ratio (power in central region)
        center = np.array(intensity.shape) // 2
        focus_radius = 10  # pixels
        y, x = np.ogrid[:intensity.shape[0], :intensity.shape[1]]
        mask = (x - center[1])**2 + (y - center[0])**2 <= focus_radius**2
        focus_power = np.sum(intensity[mask])
        focus_ratio = focus_power / total_power if total_power > 0 else 0
        
        # Beam width (1/e^2 width)
        max_pos = np.unravel_index(np.argmax(intensity), intensity.shape)
        threshold = peak_intensity / (np.e**2)
        beam_width = np.sum(intensity > threshold)**0.5
        
        return {
            'focus_ratio': float(focus_ratio),
            'peak_intensity': float(peak_intensity),
            'total_power': float(total_power),
            'beam_width': float(beam_width)
        }

class EnhancedSaver:
    """Enhanced Result Saver compatible with your data format"""
    
    def __init__(self, save_dir="./propagation_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        print(f"📁 Save directory: {self.save_dir}")
    
    def save_all_results(self, simulation_results, model_info, config=None):
        """Save all simulation results"""
        print("💾 Saving all results...")
        
        # Save individual propagation steps
        for result in simulation_results:
            self._save_single_result(result, model_info)
        
        # Save summary
        self._save_summary(simulation_results, model_info, config)
        
        # Create animations
        self._create_animations(simulation_results, model_info)
        
        print("✅ All results saved successfully")
    
    def _save_single_result(self, result, model_info):
        """Save single simulation result"""
        mode_idx = result['mode_idx']
        wl_idx = result['wavelength_idx']
        
        # Create filename prefix
        wl_nm = 1550 if wl_idx == 1 else 1310
        prefix = f"mode{mode_idx+1}_wl{wl_nm}nm"
        
        # Save propagation step images
        propagation_steps = result['propagation_steps']
        
        for step_name, field in propagation_steps.items():
            self._save_field_image(field, f"{prefix}_{step_name}", step_name)
    
    def _save_field_image(self, field, filename, title):
        """Save field as intensity and phase images"""
        try:
            field_np = field.cpu().numpy()
            intensity = np.abs(field_np)**2
            phase = np.angle(field_np)
            
            # Create combined plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Intensity plot
            im1 = axes[0].imshow(intensity, cmap='hot', origin='lower')
            axes[0].set_title(f'Intensity - {title}')
            axes[0].set_xlabel('X (pixels)')
            axes[0].set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=axes[0])
            
            # Phase plot
            im2 = axes[1].imshow(phase, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
            axes[1].set_title(f'Phase - {title}')
            axes[1].set_xlabel('X (pixels)')
            axes[1].set_ylabel('Y (pixels)')
            plt.colorbar(im2, ax=axes[1])
            
            plt.suptitle(f'{filename}', fontsize=14)
            plt.tight_layout()
            
            # Save
            img_path = self.save_dir / f"{filename}.png"
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Failed to save image {filename}: {e}")
    
    def _create_animations(self, simulation_results, model_info):
        """Create propagation animations"""
        print("🎬 Creating animations...")
        
        # Group results by mode and wavelength
        grouped_results = {}
        for result in simulation_results:
            key = (result['mode_idx'], result['wavelength_idx'])
            grouped_results[key] = result
        
        # Create animation for each mode-wavelength combination
        for (mode_idx, wl_idx), result in grouped_results.items():
            wl_nm = 1550 if wl_idx == 1 else 1310
            prefix = f"mode{mode_idx+1}_wl{wl_nm}nm"
            
            self._create_single_animation(result['propagation_steps'], prefix)
    
    def _create_single_animation(self, propagation_steps, prefix):
        """Create animation for single mode-wavelength"""
        try:
            step_names = list(propagation_steps.keys())
            intensities = []
            phases = []
            
            for field in propagation_steps.values():
                field_np = field.cpu().numpy()
                intensities.append(np.abs(field_np)**2)
                phases.append(np.angle(field_np))
            
            # Create intensity animation
            self._save_animation(intensities, step_names, 
                               f"{prefix}_intensity_propagation.gif", 
                               "Intensity Propagation", "hot")
            
            # Create phase animation
            self._save_animation(phases, step_names, 
                               f"{prefix}_phase_propagation.gif", 
                               "Phase Propagation", "hsv", vmin=-np.pi, vmax=np.pi)
            
        except Exception as e:
            print(f"⚠️ Animation creation failed for {prefix}: {e}")
    
    def _save_animation(self, data_list, step_names, filename, title, cmap, vmin=None, vmax=None):
        """Save animation as GIF"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if vmin is None:
            vmin = min(np.min(data) for data in data_list)
        if vmax is None:
            vmax = max(np.max(data) for data in data_list)
        
        im = ax.imshow(data_list[0], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} - {step_names[0]}")
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax)
        
        def animate(frame):
            im.set_array(data_list[frame])
            ax.set_title(f"{title} - {step_names[frame]} ({frame+1}/{len(step_names)})")
            return [im]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(data_list),
                                     interval=1000, blit=False, repeat=True)
        
        gif_path = self.save_dir / filename
        anim.save(gif_path, writer='pillow', fps=1)
        plt.close()
        
        print(f"  ✓ {filename}")
    
    def _save_summary(self, simulation_results, model_info, config):
        """Save simulation summary"""
        try:
            summary = {
                'model_info': {
                    'name': model_info['model_info']['name'],
                    'num_layers': model_info['num_layers'],
                    'shape': model_info['model_info']['phase_mask_shape']
                },
                'simulation_stats': {
                    'total_results': len(simulation_results),
                    'num_modes': len(set(r['mode_idx'] for r in simulation_results)),
                    'num_wavelengths': len(set(r['wavelength_idx'] for r in simulation_results))
                },
                'performance_metrics': {}
            }
            
            # Calculate average metrics
            focus_ratios = [r['focus_ratio'] for r in simulation_results]
            peak_intensities = [r['peak_intensity'] for r in simulation_results]
            
            summary['performance_metrics'] = {
                'avg_focus_ratio': float(np.mean(focus_ratios)),
                'max_focus_ratio': float(np.max(focus_ratios)),
                'avg_peak_intensity': float(np.mean(peak_intensities)),
                'max_peak_intensity': float(np.max(peak_intensities))
            }
            
            # Save summary
            summary_path = self.save_dir / "simulation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📋 Summary saved: {summary_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save summary: {e}")

def main():
    """Main function - Interactive model selection and simulation"""
    print("=" * 60)
    print("Optical Field Propagation Simulation with Model Selection")
    print("=" * 60)
    
    # Step 1: Load and select model
    print("\n🔍 Step 1: Model Selection")
    print("-" * 30)
    
    mask_loader = CustomMaskLoader("./results")
    selected_model = mask_loader.select_model()
    
    if selected_model is None:
        print("❌ No valid model selected. Exiting.")
        return
    
    # Step 2: Generate input data (using your existing method)
    print("\n📊 Step 2: Generate Input Data")
    print("-" * 30)
    
    # Here you would use your existing data generator
    # For demonstration, I'll create a compatible input format
    print("Generating input fields...")
    
    # Mock data generator - replace with your actual data_generator.generate_input_data()
    num_modes = 3
    num_wavelengths = 2
    field_size = 256
    
    input_fields = torch.zeros(num_modes, num_wavelengths, field_size, field_size, dtype=torch.complex64)
    
    # Generate different modes
    x = torch.linspace(-field_size//2, field_size//2, field_size)
    y = torch.linspace(-field_size//2, field_size//2, field_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wavelengths):
            w0 = 20 + mode_idx * 10
            gaussian = torch.exp(-(X**2 + Y**2) / (2 * w0**2))
            
            if mode_idx == 1:
                phase = torch.atan2(Y, X)  # Vortex
            elif mode_idx == 2:
                phase = torch.pi * torch.sin(2 * torch.pi * X / field_size)
            else:
                phase = torch.zeros_like(X)
            
            input_fields[mode_idx, wl_idx] = gaussian * torch.exp(1j * phase)
    
    print(f"✓ Input fields generated: {input_fields.shape}")
    print(f"  Modes: {input_fields.shape[0]}")
    print(f"  Wavelengths: {input_fields.shape[1]}")
    print(f"  Spatial size: {input_fields.shape[2]}×{input_fields.shape[3]}")
    
    # Step 3: Run simulation
    print("\n🚀 Step 3: Run Simulation")
    print("-" * 30)
    
    simulator = CompatibleSimulator()
    simulation_results = simulator.simulate_propagation(
        phase_masks=selected_model['phase_masks'],
        input_fields=input_fields,
        process_all_modes=True
    )
    
    # Step 4: Save results
    print("\n💾 Step 4: Save Results")
    print("-" * 30)
    
    saver = EnhancedSaver("./propagation_results")
    saver.save_all_results(simulation_results, selected_model)
    
    # Step 5: Summary
    print("\n📊 Step 5: Results Summary")
    print("-" * 30)
    
    print(f"✅ Simulation completed successfully!")
    print(f"   Model: {selected_model['model_info']['name']}")
    print(f"   Layers: {selected_model['num_layers']}")
    print(f"   Results: {len(simulation_results)} propagation simulations")
    print(f"   Output directory: {saver.save_dir}")
    
    # Performance summary
    focus_ratios = [r['focus_ratio'] for r in simulation_results]
    peak_intensities = [r['peak_intensity'] for r in simulation_results]
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Average focus ratio: {np.mean(focus_ratios):.4f}")
    print(f"   Best focus ratio: {np.max(focus_ratios):.4f}")
    print(f"   Average peak intensity: {np.mean(peak_intensities):.6f}")
    
    print(f"\n📁 Generated files:")
    print(f"   - Step-by-step propagation images")
    print(f"   - Intensity and phase animations (GIF)")
    print(f"   - Simulation summary (JSON)")

if __name__ == "__main__":
    main()
