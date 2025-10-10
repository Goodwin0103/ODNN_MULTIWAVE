import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from ODNN_functions import generate_fields_ts, create_labels
import matplotlib.pyplot as plt
from light_propagation_simulation_qz import propagation
from label_utils import (
    create_evaluation_regions_by_wavelength,
    visualize_labels,
)

class Config:
    def __init__(self, num_modes=3, wavelengths=None, field_size=50, layer_size=100, **kwargs):
        self.num_modes = num_modes
        self.wavelengths = wavelengths if wavelengths is not None else np.array([450e-9, 550e-9, 650e-9])
        self.field_size = field_size
        self.layer_size = layer_size
        self.batch_size = kwargs.get('batch_size', 1)
        self.focus_radius = kwargs.get('focus_radius', 5)  # 标签中聚焦点的半径
        self.detectsize = kwargs.get('detectsize', 20)     # 🔧 添加缺失的detectsize属性
        self.offsets = kwargs.get('offsets', [(0,0) for _ in range(len(self.wavelengths))])

class MultiModeMultiWavelengthDataGenerator:
    def __init__(self, config):
        self.config = config
        self.visibility_value = 0.0
        self.training_losses = []
        self.modes = None  # 存储加载的模式数据

    def load_mmf_data(self) -> torch.Tensor:
        """加载MMF数据并进行预处理"""
        eigenmodes_OM4 = np.load('eigenmodes_OM4.npy')
        print(f"原始数据形状: {eigenmodes_OM4.shape}")
        
        # 确保有足够的模式可用
        if eigenmodes_OM4.shape[2] < self.config.num_modes:
            raise ValueError(f"需要至少 {self.config.num_modes} 个模式，但数据只有 {eigenmodes_OM4.shape[2]} 个")
        
        # 从索引0开始选择模式
        MMF_data = eigenmodes_OM4[:, :, 0:self.config.num_modes].transpose(2, 0, 1)

        print(f"选择后的数据形状: {MMF_data.shape}")
        
        # 检查每个模式的振幅范围
        for i in range(MMF_data.shape[0]):
            mode_amp = np.abs(MMF_data[i])
            print(f"模式 {i+1} 振幅范围: {np.min(mode_amp)} - {np.max(mode_amp)}")
        
        # 对每个模式单独归一化
        MMF_data_amp_norm = np.zeros_like(MMF_data, dtype=np.float32)
        for i in range(MMF_data.shape[0]):
            mode_amp = np.abs(MMF_data[i])
            MMF_data_amp_norm[i] = (mode_amp - np.min(mode_amp)) / (np.max(mode_amp) - np.min(mode_amp))
        MMF_data = MMF_data_amp_norm * np.exp(1j * np.angle(MMF_data))
        
        self.modes = torch.from_numpy(MMF_data).to(torch.complex64)
        return self.modes

    def generate_weights(self) -> torch.Tensor:
        """生成模式权重"""
        # 使用单位矩阵，确保每个模式有一个主要权重
        amplitudes = np.eye(self.config.num_modes)
        # 添加一些随机相位，使每个模式更加不同
        phases = np.eye(self.config.num_modes) * np.pi * np.random.rand(self.config.num_modes, self.config.num_modes)
        complex_weights = amplitudes * np.exp(1j * phases)
        return torch.from_numpy(complex_weights)

    def generate_input_data(self) -> torch.Tensor:
        """
        修改输入数据生成方法 - 自动选择合适的生成方法
        """
        if self.config.num_modes == 1:
            print("🔄 使用单模式多波长输入数据生成")
            return self.generate_input_data_single_mode()
        else:
            print("🔄 使用多模式多波长输入数据生成")
            # 保持原有的多模式逻辑
            if self.modes is None:
                self.load_mmf_data()
                
            complex_weights_ts = self.generate_weights()
            
            multi_mode_multi_wl_data = []
            
            for logical_mode_idx in range(self.config.num_modes):
                physical_mode_idx = logical_mode_idx
                
                mode_data = []
                for wl in self.config.wavelengths:
                    field = generate_fields_ts(
                        complex_weights_ts[physical_mode_idx:physical_mode_idx+1],
                        self.modes, 
                        num_data=1,
                        num_modes=self.config.num_modes,
                        image_size=self.config.field_size, 
                        wavelength=wl
                    )
                    mode_data.append(field.squeeze())
                multi_mode_multi_wl_data.append(torch.stack(mode_data))
            
            return torch.stack(multi_mode_multi_wl_data)

    def generate_labels_by_wavelength(self):
        """
        按波长分列生成标签 - 支持单波长的修复版本
        """
        labels = torch.zeros(self.config.num_modes, len(self.config.wavelengths), 
                            self.config.layer_size, self.config.layer_size)
        
        # 🔧 传入模式数量参数
        regions = create_evaluation_regions_by_wavelength(
            self.config.layer_size, 
            self.config.layer_size, 
            self.config.focus_radius, 
            detectsize=self.config.detectsize,
            offsets=self.config.offsets,
            num_modes=self.config.num_modes  # 添加这个参数
        )
        
        print(f"创建标签 - 区域数: {len(regions)}, 标签形状: {labels.shape}")
        
        # 🔧 单波长特殊处理
        if len(self.config.wavelengths) == 1:
            print("单波长标签生成")
            # 单波长情况下，区域按模式顺序创建
            for mode_idx in range(self.config.num_modes):
                if mode_idx < len(regions):
                    # 从evaluation region获取中心位置
                    x_start, x_end, y_start, y_end = regions[mode_idx]
                    center_x = (x_start + x_end) / 2
                    center_y = (y_start + y_end) / 2
                    
                    # 创建高斯焦点分布
                    y, x = torch.meshgrid(torch.arange(self.config.layer_size), 
                                        torch.arange(self.config.layer_size), indexing='ij')
                    
                    distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
                    sigma = self.config.focus_radius / 3
                    gaussian = torch.exp(-distance**2 / (2 * sigma**2))
                    
                    # 单波长情况：wl_idx = 0
                    labels[mode_idx, 0] = gaussian / gaussian.max()
                    
                    print(f"  模式{mode_idx+1}: 区域{mode_idx} -> 标签[{mode_idx}, 0], 中心({center_x:.1f}, {center_y:.1f})")
            
            return labels
        
        region_idx = 0
        for wl_idx in range(len(self.config.wavelengths)):    # 区域创建的外层循环
            for mode_idx in range(self.config.num_modes):     # 区域创建的内层循环
                if region_idx < len(regions):
                    # 从evaluation region获取中心位置
                    x_start, x_end, y_start, y_end = regions[region_idx]
                    center_x = (x_start + x_end) / 2
                    center_y = (y_start + y_end) / 2
                    
                    # 创建高斯焦点分布
                    y, x = torch.meshgrid(torch.arange(self.config.layer_size), 
                                        torch.arange(self.config.layer_size), indexing='ij')
                    
                    distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
                    sigma = self.config.focus_radius / 3
                    gaussian = torch.exp(-distance**2 / (2 * sigma**2))
                    
                    labels[mode_idx, wl_idx] = gaussian / gaussian.max()
                    
                    print(f"  区域{region_idx}: wl{wl_idx+1}_mode{mode_idx+1} -> 标签[{mode_idx}, {wl_idx}], 中心({center_x:.1f}, {center_y:.1f})")
                    region_idx += 1
                else:
                    print(f"  ⚠️ 区域索引{region_idx}超出范围")
        
        return labels

    def generate_labels_multi_wavelength_single_mode(self):
        """
        多波长单模式标签生成 - 专门针对单模式多波长情况优化
        水平排列各波长的焦点
        """
        print(f"🎯 生成多波长单模式标签:")
        print(f"   模式数量: {self.config.num_modes}")
        print(f"   波长数量: {len(self.config.wavelengths)}")
        
        # 确保是单模式配置
        if self.config.num_modes != 1:
            print(f"⚠️ 警告：当前配置为{self.config.num_modes}模式，建议使用单模式配置")
        
        num_wavelengths = len(self.config.wavelengths)
        labels = torch.zeros(1, num_wavelengths, 
                            self.config.layer_size, self.config.layer_size)
        
        print(f"   标签形状: {labels.shape}")
        
        # 按波长水平排列的布局
        image_width = self.config.layer_size
        image_height = self.config.layer_size
        
        # 计算每个波长的焦点位置 - 水平排列
        padding_ratio = 0.2
        available_width = image_width * (1 - 2 * padding_ratio)
        
        if num_wavelengths == 1:
            # 单波长时居中
            centers_x = [image_width / 2]
        else:
            # 多波长时均匀分布
            spacing = available_width / (num_wavelengths - 1)
            start_x = image_width * padding_ratio
            centers_x = [start_x + i * spacing for i in range(num_wavelengths)]
        
        center_y = image_height / 2  # 垂直居中
        
        for wl_idx in range(num_wavelengths):
            # 计算当前波长的焦点位置
            center_x = centers_x[wl_idx]
            
            # 应用偏移（如果有）
            if hasattr(self.config, 'offsets') and self.config.offsets:
                if wl_idx < len(self.config.offsets):
                    offset_y, offset_x = self.config.offsets[wl_idx]
                    center_x += offset_x
                    center_y_adjusted = center_y + offset_y
                else:
                    center_y_adjusted = center_y
            else:
                center_y_adjusted = center_y
            
            # 创建高斯焦点分布
            y, x = torch.meshgrid(torch.arange(self.config.layer_size), 
                                torch.arange(self.config.layer_size), indexing='ij')
            
            distance = torch.sqrt((x - center_x)**2 + (y - center_y_adjusted)**2)
            sigma = self.config.focus_radius / 3  # 高斯分布的标准差
            gaussian = torch.exp(-distance**2 / (2 * sigma**2))
            
            # 归一化到[0,1]
            labels[0, wl_idx] = gaussian / gaussian.max()
            
            wl_nm = self.config.wavelengths[wl_idx] * 1e9
            print(f"   波长{wl_nm:.0f}nm: 中心位置({center_x:.1f}, {center_y_adjusted:.1f})")
        
        return labels

    def generate_input_data_single_mode(self) -> torch.Tensor:
        """
        单模式多波长输入数据生成
        """
        if self.modes is None:
            self.load_mmf_data()
        
        # 单模式情况，只使用第一个模式
        mode_idx = 0
        complex_weights_ts = torch.ones(1, dtype=torch.complex64)  # 单位权重
        
        multi_wl_data = []
        
        for wl in self.config.wavelengths:
            field = generate_fields_ts(
                complex_weights_ts,
                self.modes[mode_idx:mode_idx+1],  # 只使用一个模式
                num_data=1,
                num_modes=1,  # 强制设为1
                image_size=self.config.field_size, 
                wavelength=wl
            )
            multi_wl_data.append(field.squeeze())
        
        # 返回形状: [1, num_wavelengths, H, W]
        return torch.stack(multi_wl_data).unsqueeze(0)

    def generate_labels(self):
        """
        生成标签的主函数 - 自动选择合适的生成方法
        """
        if self.config.num_modes == 1:
            print("🔄 使用多波长单模式标签生成")
            return self.generate_labels_multi_wavelength_single_mode()
        else:
            print("🔄 使用多模式多波长标签生成")
            return self.generate_labels_by_wavelength()

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """对输入图像进行预处理（添加填充）"""
        # 输入形状: [num_wavelengths, H, W]
        # 输出形状: [num_wavelengths, layer_size, layer_size]
        padding_size = (self.config.layer_size - self.config.field_size) // 2
        padding = (padding_size, padding_size, padding_size, padding_size)
        
        # 对每个波长通道应用填充
        padded_channels = []
        for i in range(image.shape[0]):
            padded = torch.nn.functional.pad(image[i:i+1], padding)
            padded_channels.append(padded)
        
        return torch.cat(padded_channels, dim=0)

    def create_dataloader(self) -> DataLoader:
        """
        创建数据加载器 - 支持单模式和多模式
        """
        # 生成输入数据和标签
        image_data = self.generate_input_data()  
        label_data = self.generate_labels()      
        
        print(f"📊 数据加载器信息:")
        print(f"   输入数据形状: {image_data.shape}")
        print(f"   标签数据形状: {label_data.shape}")
        
        # 创建数据集
        train_dataset = []
        
        if self.config.num_modes == 1:
            # 单模式情况：image_data [1, num_wavelengths, H, W], label_data [1, num_wavelengths, layer_size, layer_size]
            img_pad = self._preprocess_image(image_data[0])  # [num_wavelengths, layer_size, layer_size]
            lbl = label_data[0]                             # [num_wavelengths, layer_size, layer_size]
            train_dataset.append((img_pad, lbl))
        else:
            # 多模式情况：保持原有逻辑
            for i in range(self.config.num_modes):
                img_pad = self._preprocess_image(image_data[i])  
                lbl = label_data[i]                             
                train_dataset.append((img_pad, lbl))
        
        # 创建TensorDataset
        train_tensor_data = TensorDataset(*[torch.stack(tensors) for tensors in zip(*train_dataset)])
        
        # 创建DataLoader
        return DataLoader(train_tensor_data, batch_size=self.config.batch_size, shuffle=False)

    def visualize_data(self, save_path=None, save_individual=False):
        """
        便捷的数据可视化方法 - 直接调用label_utils中的函数
        """
        print("🎨 开始数据可视化...")
        
        # 生成标签
        labels = self.generate_labels()
        
        # 调用统一的可视化函数
        visualize_labels(
            labels, 
            wavelengths=self.config.wavelengths,
            save_path=save_path,
            style='clean_black',
            save_individual=save_individual
        )
        
        return labels