import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from ODNN_functions import generate_fields_ts, create_labels
import matplotlib.pyplot as plt
from light_propagation_simulation_qz import propagation
from label_utils import create_labels_mode_wavelength, create_evaluation_regions_mode_wavelength

class Config:
    def __init__(self, num_modes=3, wavelengths=None, field_size=50, layer_size=100, **kwargs):
        self.num_modes = num_modes
        self.wavelengths = wavelengths if wavelengths is not None else np.array([450e-9, 550e-9, 650e-9])
        self.field_size = field_size
        self.layer_size = layer_size
        self.batch_size = kwargs.get('batch_size', 1)
        self.focus_radius = kwargs.get('focus_radius', 5)  # 标签中聚焦点的半径
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

    def generate_labels(self):
        """生成与evaluation regions对应的多模式多波长标签"""
        labels = torch.zeros(self.config.num_modes, len(self.config.wavelengths), 
                            self.config.layer_size, self.config.layer_size)
        
        # 🆕 直接使用evaluation regions的位置生成标签
        from label_utils import create_evaluation_regions_mode_wavelength
        
        regions = create_evaluation_regions_mode_wavelength(
            self.config.layer_size, 
            self.config.layer_size, 
            self.config.focus_radius, 
            detectsize=self.config.detectsize,
            offsets=self.config.offsets
        )
        
        region_idx = 0
        for mode_idx in range(self.config.num_modes):
            for wl_idx in range(len(self.config.wavelengths)):
                # 从evaluation region获取中心位置
                x_start, x_end, y_start, y_end = regions[region_idx]
                center_x = (x_start + x_end) / 2
                center_y = (y_start + y_end) / 2
                
                # 确保位置在有效范围内
                center_x = max(0, min(self.config.layer_size-1, center_x))
                center_y = max(0, min(self.config.layer_size-1, center_y))
                
                # 创建高斯焦点
                y, x = torch.meshgrid(torch.arange(self.config.layer_size), 
                                    torch.arange(self.config.layer_size), indexing='ij')
                
                distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
                sigma = self.config.focus_radius / 2
                gaussian = torch.exp(-distance**2 / (2 * sigma**2))
                
                labels[mode_idx, wl_idx] = gaussian / gaussian.max()
                region_idx += 1
        
        return labels


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
        """创建数据加载器"""
        # 生成输入数据和标签
        image_data = self.generate_input_data()  # [num_modes, num_wavelengths, H, W]
        label_data = self.generate_labels()      # [num_modes, num_wavelengths, layer_size, layer_size]
        
        # 创建数据集
        train_dataset = []
        for i in range(self.config.num_modes):
            # 对每个模式的所有波长通道进行预处理
            img_pad = self._preprocess_image(image_data[i])  # [num_wavelengths, layer_size, layer_size]
            lbl = label_data[i]                             # [num_wavelengths, layer_size, layer_size]
            train_dataset.append((img_pad, lbl))
        
        # 创建TensorDataset
        train_tensor_data = TensorDataset(*[torch.stack(tensors) for tensors in zip(*train_dataset)])
        
        # 创建DataLoader
        return DataLoader(train_tensor_data, batch_size=self.config.batch_size, shuffle=False)
    
    def visualize_labels(self):
        """可视化九个标签的布局"""
        import matplotlib.pyplot as plt
        
        labels = self.generate_labels()
        
        fig, axes = plt.subplots(self.config.num_modes, len(self.config.wavelengths), 
                                figsize=(12, 12))
        
        for mode_idx in range(self.config.num_modes):
            for wl_idx, wl in enumerate(self.config.wavelengths):
                ax = axes[mode_idx, wl_idx]
                ax.imshow(labels[mode_idx, wl_idx].numpy(), cmap='viridis')
                ax.set_title(f'模式 {mode_idx+1}, 波长 {int(wl*1e9)}nm')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('labels_visualization.png')
        plt.show()
        
        # 合成所有标签到一个图像以检查空间分布
        combined_label = torch.zeros((self.config.layer_size, self.config.layer_size))
        for m in range(self.config.num_modes):
            for w in range(len(self.config.wavelengths)):
                combined_label += labels[m, w]
        
        plt.figure(figsize=(8, 8))
        plt.imshow(combined_label.numpy(), cmap='viridis')
        plt.title('所有标签的合成视图')
        plt.colorbar()
        plt.axis('off')
        plt.savefig('combined_labels.png')
        plt.show()

def generate_fields_ts(complex_weights, MMF_data, num_data, num_modes, image_size,
                       wavelength=None, z0=40e-6, dx=1e-6, device='cpu'):
    """
    生成场分布并从光纤输出传播到第一个相位屏。

    参数:
        complex_weights (Tensor): [num_data, num_modes], complex64.
        MMF_data       (Tensor): [num_modes, H, W], complex64 at fiber output.
        num_data (int), num_modes (int), image_size (int)
        wavelength (float): chosen lambda (m)
        z0 (float): fiber→first-screen distance (m)
        dx (float): pixel pitch (m)
        device (str): 'cpu' or 'cuda:0'

    返回:
        image_data: [num_data,1,H,W], complex64, field on first screen.
    """
    MMF_data = MMF_data.to(device)
    image_data = torch.zeros([num_data, 1, image_size, image_size],
                             dtype=torch.complex64, device=device)

    for idx in range(num_data):
        # 1) 叠加模式
        w = complex_weights[idx].view(num_modes,1,1).to(device)
        field0 = torch.sum(w * MMF_data, dim=0)  # [H,W], at fiber output

        if wavelength is not None:
            # 2) 真实自由空间传播到第一个相位屏
            #    propagation(E, z_start, z_prop, N, dx, device, wavelength)
            field1 = propagation(field0, z0, wavelength, image_size, dx, device)
        else:
            field1 = field0

        image_data[idx,0] = field1

    return image_data


