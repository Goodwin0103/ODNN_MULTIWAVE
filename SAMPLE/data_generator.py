import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from scipy.interpolate import griddata

class SingleModeDualWavelengthDataGenerator:
    def __init__(self, config):
        self.config = config
        self.field_size = config.field_size
        self.wavelengths = config.wavelengths
        self.device = config.device
        
        # 加载或生成基础模式
        self.base_mode = self._load_or_generate_mode()
        
    def _load_or_generate_mode(self):
        """加载或生成基础光场模式"""
        # 尝试加载预存的特征模式
        eigenmode_files = [
            'eigenmodes_OM4.npy',
            '/home/shiyue/ODNN_MULTIWAVE/SAMPLE/eigenmodes_OM4.npy',
            'data/eigenmodes_OM4.npy'
        ]
        
        for file_path in eigenmode_files:
            if os.path.exists(file_path):
                print(f"找到特征模式文件，正在加载: {file_path}")
                try:
                    data = np.load(file_path)
                    print(f"成功加载特征模式数据，形状: {data.shape}")
                    
                    # 处理不同的数据格式
                    if len(data.shape) == 3:
                        # 选择第一个模式
                        mode = data[:, :, 0]
                        print(f"原始模式形状: {mode.shape}")
                    elif len(data.shape) == 2:
                        mode = data
                        print(f"原始模式形状: {mode.shape}")
                    else:
                        print(f"⚠️  未知的数据格式: {data.shape}")
                        continue
                    
                    # 调整尺寸到目标大小
                    print(f"目标形状: ({self.field_size}, {self.field_size})")
                    adjusted_mode = self._resize_mode(mode, self.field_size)
                    
                    # 转换为torch tensor
                    mode_tensor = torch.tensor(adjusted_mode, dtype=torch.complex64, device=self.device)
                    print(f"✅ 成功加载并调整模式尺寸: {mode_tensor.shape}")
                    return mode_tensor
                    
                except Exception as e:
                    print(f"⚠️  加载特征模式失败: {e}")
                    continue
        
        # 如果没有找到文件，生成高斯模式
        print("未找到特征模式文件，生成高斯模式...")
        return self._generate_gaussian_mode()
    
    def _resize_mode(self, mode, target_size):
        """调整模式尺寸"""
        current_shape = mode.shape
        print(f"原始模式形状: {current_shape}")
        
        # 处理非方形数据
        if current_shape[0] != current_shape[1]:
            print(f"⚠️  检测到非方形数据: {current_shape}")
            # 裁剪到方形
            min_size = min(current_shape[0], current_shape[1])
            start_x = (current_shape[0] - min_size) // 2
            start_y = (current_shape[1] - min_size) // 2
            mode = mode[start_x:start_x+min_size, start_y:start_y+min_size]
            print(f"🔧 裁剪到方形: {min_size}x{min_size}")
            print(f"裁剪后形状: {mode.shape}")
        
        current_size = mode.shape[0]
        print(f"当前尺寸: {current_size}x{current_size}")
        print(f"目标尺寸: {target_size}x{target_size}")
        
        if current_size == target_size:
            print("✅ 尺寸匹配，无需调整")
            return mode
        
        # 使用插值调整尺寸
        x_old = np.linspace(0, 1, current_size)
        y_old = np.linspace(0, 1, current_size)
        X_old, Y_old = np.meshgrid(x_old, y_old)
        
        x_new = np.linspace(0, 1, target_size)
        y_new = np.linspace(0, 1, target_size)
        X_new, Y_new = np.meshgrid(x_new, y_new)
        
        # 处理复数数据
        if np.iscomplexobj(mode):
            real_part = griddata((X_old.flatten(), Y_old.flatten()), 
                               mode.real.flatten(), 
                               (X_new, Y_new), method='cubic', fill_value=0)
            imag_part = griddata((X_old.flatten(), Y_old.flatten()), 
                               mode.imag.flatten(), 
                               (X_new, Y_new), method='cubic', fill_value=0)
            resized_mode = real_part + 1j * imag_part
        else:
            resized_mode = griddata((X_old.flatten(), Y_old.flatten()), 
                                  mode.flatten(), 
                                  (X_new, Y_new), method='cubic', fill_value=0)
        
        print(f"✅ 调整完成，最终尺寸: {resized_mode.shape}")
        return resized_mode
    
    def _generate_gaussian_mode(self):
        """生成高斯基模"""
        print("生成高斯基模...")
        x = torch.linspace(-1, 1, self.field_size, device=self.device)
        y = torch.linspace(-1, 1, self.field_size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 高斯光束参数
        w0 = 0.3  # 束腰半径
        gaussian = torch.exp(-(X**2 + Y**2) / w0**2)
        
        return gaussian.to(torch.complex64)
    
    def generate_input_fields(self):
        """生成多波长输入场"""
        fields = []
        
        for wavelength in self.wavelengths:
            # 为每个波长生成略有不同的模式
            wl_factor = wavelength / self.wavelengths[0]  # 归一化波长因子
            
            # 根据波长调整模式尺寸（色散效应）
            if hasattr(self.config, 'use_dispersion') and self.config.use_dispersion:
                scaled_mode = self._apply_dispersion(self.base_mode, wl_factor)
            else:
                scaled_mode = self.base_mode
            
            # 归一化
            scaled_mode = scaled_mode / torch.sqrt(torch.sum(torch.abs(scaled_mode)**2))
            fields.append(scaled_mode)
        
        return torch.stack(fields)
    
    def _apply_dispersion(self, mode, wl_factor):
        """应用色散效应"""
        # 简单的色散模型：不同波长有不同的传播常数
        dispersion_factor = 1.0 + 0.1 * (wl_factor - 1.0)
        
        # 在傅里叶域应用色散
        mode_fft = torch.fft.fft2(mode)
        
        # 创建频率网格
        kx = torch.fft.fftfreq(self.field_size, device=self.device)
        ky = torch.fft.fftfreq(self.field_size, device=self.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        k_squared = KX**2 + KY**2
        
        # 应用色散相位
        dispersion_phase = torch.exp(1j * dispersion_factor * k_squared)
        dispersed_fft = mode_fft * dispersion_phase
        
        # 逆傅里叶变换
        dispersed_mode = torch.fft.ifft2(dispersed_fft)
        
        return dispersed_mode
    
    def visualize_separation_concept(self, save_path=None):
        """可视化波长分离概念（无GUI版本）"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('多波长分离概念图', fontsize=16)
            
            # 生成输入场
            input_fields = self.generate_input_fields()
            
            # 显示输入场
            for i, (wavelength, field) in enumerate(zip(self.wavelengths, input_fields)):
                wl_nm = int(wavelength * 1e9)
                intensity = torch.abs(field).cpu().numpy()
                
                ax = axes[0, i]
                im = ax.imshow(intensity, cmap='hot')
                ax.set_title(f'输入场 - {wl_nm}nm')
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                plt.colorbar(im, ax=ax)
            
            # 显示检测区域布局
            ax = axes[1, 0]
            detection_layout = np.zeros((self.field_size, self.field_size))
            
            center = self.field_size // 2
            for i, offset in enumerate(self.config.offsets):
                detect_center_x = center + offset[0]
                detect_center_y = center + offset[1]
                
                half_size = self.config.detectsize // 2
                x_start = max(0, detect_center_x - half_size)
                x_end = min(self.field_size, detect_center_x + half_size)
                y_start = max(0, detect_center_y - half_size)
                y_end = min(self.field_size, detect_center_y + half_size)
                
                detection_layout[x_start:x_end, y_start:y_end] = i + 1
            
            im = ax.imshow(detection_layout, cmap='tab10')
            ax.set_title('检测区域布局')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.colorbar(im, ax=ax)
            
            # 显示波长信息
            ax = axes[1, 1]
            ax.axis('off')
            info_text = f"波长配置:\n"
            for i, wl in enumerate(self.wavelengths):
                wl_nm = int(wl * 1e9)
                offset = self.config.offsets[i]
                info_text += f"  {wl_nm}nm: 偏移 {offset}\n"
            
            info_text += f"\n系统参数:\n"
            info_text += f"  场尺寸: {self.field_size}×{self.field_size}\n"
            info_text += f"  检测尺寸: {self.config.detectsize}×{self.config.detectsize}\n"
            info_text += f"  像素尺寸: {self.config.pixel_size*1e6:.1f} μm"
            
            ax.text(0.1, 0.9, info_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ 分离概念图已保存: {save_path}")
            
            plt.close()  # 关闭图形，释放内存
            
        except Exception as e:
            print(f"⚠️  分离概念可视化失败: {e}")
    
    def visualize_detector_layout(self, save_path=None):
        """可视化检测器布局（无GUI版本）"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # 创建检测器布局
            layout = np.zeros((self.field_size, self.field_size))
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            center = self.field_size // 2
            
            for i, (wavelength, offset) in enumerate(zip(self.wavelengths, self.config.offsets)):
                wl_nm = int(wavelength * 1e9)
                
                detect_center_x = center + offset[0]
                detect_center_y = center + offset[1]
                
                half_size = self.config.detectsize // 2
                x_start = max(0, detect_center_x - half_size)
                x_end = min(self.field_size, detect_center_x + half_size)
                y_start = max(0, detect_center_y - half_size)
                y_end = min(self.field_size, detect_center_y + half_size)
                
                layout[x_start:x_end, y_start:y_end] = i + 1
                
                # 添加标签
                ax.text(detect_center_y, detect_center_x, f'{wl_nm}nm', 
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       color='white', bbox=dict(boxstyle='round', facecolor=colors[i % len(colors)], alpha=0.7))
            
            im = ax.imshow(layout, cmap='Set3', alpha=0.8)
            ax.set_title('检测器布局图', fontsize=16)
            ax.set_xlabel('Y (pixels)', fontsize=12)
            ax.set_ylabel('X (pixels)', fontsize=12)
            
            # 添加网格
            ax.grid(True, alpha=0.3)
            
            # 添加中心标记
            ax.plot(center, center, 'k+', markersize=15, markeredgewidth=3)
            ax.text(center, center-20, '光轴中心', ha='center', va='top', fontsize=10)
            
            plt.colorbar(im, ax=ax, label='检测区域ID')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ 检测器布局图已保存: {save_path}")
            
            plt.close()  # 关闭图形，释放内存
            
        except Exception as e:
            print(f"⚠️  检测器布局可视化失败: {e}")
