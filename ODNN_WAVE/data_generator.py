import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from ODNN_functions import generate_fields_ts, create_labels
import matplotlib.pyplot as plt
from light_propagation_simulation_qz import propagation
from label_utils import create_evaluation_regions_by_wavelength

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
        按波长分列生成标签 - 修复版本
        """
        labels = torch.zeros(self.config.num_modes, len(self.config.wavelengths), 
                            self.config.layer_size, self.config.layer_size)
        
        # 使用按波长分列的评估区域
        regions = create_evaluation_regions_by_wavelength(
            self.config.layer_size, 
            self.config.layer_size, 
            self.config.focus_radius, 
            detectsize=self.config.detectsize,
            offsets=self.config.offsets
        )
        
        print(f"创建标签 - 区域数: {len(regions)}, 标签形状: {labels.shape}")
        
        # 🔧 关键修复：确保区域索引与标签索引的对应关系正确
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
                    
                    # 🔧 关键修复：标签索引要与区域创建顺序对应
                    # 区域按 (wl_idx, mode_idx) 创建，标签按 [mode_idx, wl_idx] 存储
                    labels[mode_idx, wl_idx] = gaussian / gaussian.max()
                    
                    print(f"  区域{region_idx}: wl{wl_idx+1}_mode{mode_idx+1} -> 标签[{mode_idx}, {wl_idx}]")
                    region_idx += 1
                else:
                    print(f"  ⚠️ 区域索引{region_idx}超出范围")
        
        return labels

    def generate_labels(self):
        """生成标签的主函数"""
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
        """创建数据加载器"""
        # 生成输入数据和标签
        image_data = self.generate_input_data()  # [num_modes, num_wavelengths, H, W]
        label_data = self.generate_labels_by_wavelength()      # [num_modes, num_wavelengths, layer_size, layer_size]
        
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

# 🔧 将可视化函数移到类外部作为独立函数
def visualize_labels_by_wavelength(labels, wavelengths, save_path=None, show_colorbar=False):
    """
    按波长分列的标签可视化
    
    参数:
        labels: torch.Tensor, shape [modes, wavelengths, H, W]
        wavelengths: list or array, 波长列表
        save_path: str, 保存路径（可选）
        show_colorbar: bool, 是否显示颜色条（默认False）
    """
    # 转换为numpy数组
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    num_modes = labels.shape[0]
    num_wl = labels.shape[1]
    
    print(f"可视化标签: {num_modes} 个模式, {num_wl} 个波长")
    print(f"标签形状: {labels.shape}")
    
    # 创建图像，列对应波长，行对应模式
    fig, axes = plt.subplots(num_modes, num_wl, figsize=(num_wl*4, num_modes*4))
    
    # 处理单行或单列的情况
    if num_modes == 1 and num_wl == 1:
        axes = np.array([[axes]])
    elif num_modes == 1:
        axes = axes.reshape(1, -1)
    elif num_wl == 1:
        axes = axes.reshape(-1, 1)
    
    # 设置列标题（波长）
    for wl_idx in range(num_wl):
        wl_nm = int(wavelengths[wl_idx] * 1e9)
        axes[0, wl_idx].set_title(f'λ = {wl_nm}nm', fontsize=14, fontweight='bold', pad=20)
    
    # 设置行标题（模式）
    for mode_idx in range(num_modes):
        axes[mode_idx, 0].set_ylabel(f'MODE {mode_idx+1}', fontsize=12, fontweight='bold')
    
    # 绘制每个标签
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wl):
            label_data = labels[mode_idx, wl_idx]
            
            # 检查数据范围
            vmin, vmax = label_data.min(), label_data.max()
            print(f"  模式 {mode_idx+1}, 波长 {wl_idx+1}: 数据范围 [{vmin:.3f}, {vmax:.3f}]")
            
            # 绘制图像
            im = axes[mode_idx, wl_idx].imshow(label_data, 
                                             cmap='plasma', 
                                             vmin=0, vmax=1,
                                             interpolation='bilinear')
            axes[mode_idx, wl_idx].axis('off')
            
            # 🔧 只有在需要时才添加颜色条
            if show_colorbar:
                plt.colorbar(im, ax=axes[mode_idx, wl_idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path is not None and isinstance(save_path, str):
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图像已保存到: {save_path}")
        except Exception as e:
            print(f"⚠️  保存图像失败: {e}")
    
    plt.show()

def visualize_labels(labels, wavelengths=None, save_path=None, show_colorbar=False):
    """
    通用标签可视化函数
    
    参数:
        labels: 标签数据
        wavelengths: 波长列表（可选）
        save_path: 保存路径（可选）
        show_colorbar: bool, 是否显示颜色条（默认False）
    """
    print(f"标签可视化 - 输入形状: {labels.shape}")
    
    if wavelengths is not None and not isinstance(wavelengths, str):
        if labels.ndim == 4:  # [modes, wavelengths, H, W]
            print(f"使用4D标签可视化，波长: {[f'{wl*1e9:.0f}nm' for wl in wavelengths]}")
            visualize_labels_by_wavelength(labels, wavelengths, save_path, show_colorbar)
            return
    
    # 默认处理
    if labels.ndim == 4:  # [modes, wavelengths, H, W]
        default_wavelengths = np.array([1310e-9, 1550e-9])
        print(f"使用默认波长: {[f'{wl*1e9:.0f}nm' for wl in default_wavelengths]}")
        visualize_labels_by_wavelength(labels, default_wavelengths, save_path, show_colorbar)
        
    elif labels.ndim == 3:  # [channels, H, W]
        print("使用3D标签可视化")
        num_channels = labels.shape[0]
        
        fig, axes = plt.subplots(1, num_channels, figsize=(num_channels*4, 4))
        if num_channels == 1:
            axes = [axes]
        
        for i in range(num_channels):
            data = labels[i].numpy() if torch.is_tensor(labels) else labels[i]
            im = axes[i].imshow(data, cmap='plasma', vmin=0, vmax=1)
            axes[i].set_title(f'Channel {i+1}')
            axes[i].axis('off')
            
            if show_colorbar:
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    else:  # 2D 或其他
        print("使用2D标签可视化")
        plt.figure(figsize=(8, 6))
        data = labels.numpy() if torch.is_tensor(labels) else labels
        
        if data.ndim > 2:
            data = np.sum(data, axis=tuple(range(data.ndim-2)))
        
        im = plt.imshow(data, cmap='plasma')
        plt.title('标签可视化')
        if show_colorbar:
            plt.colorbar(im)
        plt.axis('off')
        
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
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
