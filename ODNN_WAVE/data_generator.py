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
        
        # 🔧 多波长情况保持原有逻辑
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
                    
                    print(f"  区域{region_idx}: wl{wl_idx+1}_mode{mode_idx+1} -> 标签[{mode_idx}, {wl_idx}], 中心({center_x:.1f}, {center_y:.1f})")
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

def visualize_labels_by_wavelength(labels, wavelengths, save_path=None, show_colorbar=False):
    """
    按波长分列的标签可视化 - 黑底纯净版本
    """
    # 转换为numpy数组
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    num_modes = labels.shape[0]
    num_wl = labels.shape[1]
    
    print(f"可视化标签: {num_modes} 个模式, {num_wl} 个波长")
    
    # 🔧 创建黑底图像 - 设置黑色背景
    fig = plt.figure(figsize=(num_wl*3, num_modes*3), facecolor='black')
    
    # 创建子图网格
    axes = []
    for i in range(num_modes * num_wl):
        ax = fig.add_subplot(num_modes, num_wl, i+1, facecolor='black')
        axes.append(ax)
    
    # 重新整理axes为2D数组
    axes = np.array(axes).reshape(num_modes, num_wl)
    
    # 🔧 绘制每个标签，使用黑底配色
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wl):
            label_data = labels[mode_idx, wl_idx]
            
            # 🔧 使用热图配色方案，黑色为背景
            im = axes[mode_idx, wl_idx].imshow(label_data, 
                                             cmap='hot',  # 改为hot配色，黑底红黄色
                                             vmin=0, vmax=1,
                                             interpolation='bilinear')
            
            # 🔧 移除所有装饰，设置黑色背景
            axes[mode_idx, wl_idx].axis('off')
            axes[mode_idx, wl_idx].set_facecolor('black')
            
            # 🔧 添加模式和波长标识（白色文字）
            if wavelengths is not None and wl_idx < len(wavelengths):
                wl_nm = int(wavelengths[wl_idx] * 1e9)
                axes[mode_idx, wl_idx].text(0.02, 0.98, f'M{mode_idx+1}', 
                                          transform=axes[mode_idx, wl_idx].transAxes,
                                          color='white', fontsize=12, fontweight='bold',
                                          verticalalignment='top')
                
                # 只在第一行添加波长标识
                if mode_idx == 0:
                    axes[mode_idx, wl_idx].text(0.5, 0.98, f'λ = {wl_nm}nm', 
                                              transform=axes[mode_idx, wl_idx].transAxes,
                                              color='white', fontsize=10,
                                              horizontalalignment='center',
                                              verticalalignment='top')
    
    # 🔧 移除子图间的间距，保持黑色背景
    plt.subplots_adjust(wspace=0.02, hspace=0.02, 
                       left=0.01, right=0.99, 
                       top=0.99, bottom=0.01)
    
    # 🔧 保存为黑底图像
    if save_path is not None and isinstance(save_path, str):
        try:
            plt.savefig(save_path, dpi=300, 
                       bbox_inches='tight', 
                       pad_inches=0.02,
                       facecolor='black',  # 确保保存时背景为黑色
                       edgecolor='none')
            print(f"✓ 黑底标签图像已保存到: {save_path}")
        except Exception as e:
            print(f"⚠️  保存图像失败: {e}")
    
    plt.show()

def visualize_labels(labels, wavelengths=None, save_path=None, show_colorbar=False, 
                    style='clean_black'):
    """
    通用标签可视化函数 - 支持多种风格
    
    参数:
        labels: 标签数据
        wavelengths: 波长列表（可选）
        save_path: 保存路径（可选）
        show_colorbar: bool, 是否显示颜色条（默认False）
        style: str, 可视化风格 ('clean_black', 'default')
    """
    print(f"标签可视化 - 输入形状: {labels.shape}, 风格: {style}")
    
    if wavelengths is not None and not isinstance(wavelengths, str):
        if labels.ndim == 4:  # [modes, wavelengths, H, W]
            print(f"使用4D标签可视化，波长: {[f'{wl*1e9:.0f}nm' for wl in wavelengths]}")
            if style == 'clean_black':
                visualize_labels_by_wavelength(labels, wavelengths, save_path, show_colorbar)
            else:
                visualize_labels_by_wavelength_default(labels, wavelengths, save_path, show_colorbar)
            return
    
    # 默认处理
    if labels.ndim == 4:  # [modes, wavelengths, H, W]
        default_wavelengths = np.array([1310e-9, 1550e-9])
        print(f"使用默认波长: {[f'{wl*1e9:.0f}nm' for wl in default_wavelengths]}")
        visualize_labels_by_wavelength(labels, default_wavelengths, save_path, show_colorbar)
        
    elif labels.ndim == 3:  # [channels, H, W]
        print("使用3D标签可视化")
        _visualize_3d_labels_clean_black(labels, save_path, show_colorbar)
        
    else:  # 2D 或其他
        print("使用2D标签可视化")
        _visualize_2d_labels_clean_black(labels, save_path, show_colorbar)

def _visualize_3d_labels_clean_black(labels, save_path=None, show_colorbar=False):
    """3D标签的黑底可视化"""
    num_channels = labels.shape[0]
    
    # 🔧 创建黑底图像
    fig = plt.figure(figsize=(num_channels*4, 4), facecolor='black')
    
    axes = []
    for i in range(num_channels):
        ax = fig.add_subplot(1, num_channels, i+1, facecolor='black')
        axes.append(ax)
    
    for i in range(num_channels):
        data = labels[i].numpy() if torch.is_tensor(labels) else labels[i]
        
        # 🔧 使用热图配色
        im = axes[i].imshow(data, cmap='hot', vmin=0, vmax=1)
        
        # 🔧 白色标题，移除坐标轴
        axes[i].text(0.02, 0.98, f'M{i+1}', 
                    transform=axes[i].transAxes,
                    color='white', fontsize=14, fontweight='bold',
                    verticalalignment='top')
        axes[i].axis('off')
        axes[i].set_facecolor('black')
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), color='white')
    
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    if save_path is not None and isinstance(save_path, str):
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        print(f"✓ 黑底3D标签图像已保存到: {save_path}")
    
    plt.show()

def _visualize_2d_labels_clean_black(labels, save_path=None, show_colorbar=False):
    """2D标签的黑底可视化"""
    # 🔧 创建黑底图像
    fig = plt.figure(figsize=(8, 6), facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    
    data = labels.numpy() if torch.is_tensor(labels) else labels
    
    if data.ndim > 2:
        data = np.sum(data, axis=tuple(range(data.ndim-2)))
    
    # 🔧 使用热图配色
    im = ax.imshow(data, cmap='hot')
    
    # 🔧 白色标题，移除坐标轴
    ax.text(0.5, 0.98, '标签可视化', 
           transform=ax.transAxes,
           color='white', fontsize=16, fontweight='bold',
           horizontalalignment='center', verticalalignment='top')
    ax.axis('off')
    
    if show_colorbar:
        cbar = plt.colorbar(im)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), color='white')
    
    if save_path is not None and isinstance(save_path, str):
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        print(f"✓ 黑底2D标签图像已保存到: {save_path}")
    
    plt.show()

# 🔧 保留原版本作为备用
def visualize_labels_by_wavelength_default(labels, wavelengths, save_path=None, show_colorbar=False):
    """
    按波长分列的标签可视化 - 原版本（白底）
    """
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    num_modes = labels.shape[0]
    num_wl = labels.shape[1]
    
    fig, axes = plt.subplots(num_modes, num_wl, figsize=(num_wl*3, num_modes*3))
    
    if num_modes == 1 and num_wl == 1:
        axes = np.array([[axes]])
    elif num_modes == 1:
        axes = axes.reshape(1, -1)
    elif num_wl == 1:
        axes = axes.reshape(-1, 1)
    
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wl):
            label_data = labels[mode_idx, wl_idx]
            
            im = axes[mode_idx, wl_idx].imshow(label_data, 
                                             cmap='plasma', 
                                             vmin=0, vmax=1,
                                             interpolation='bilinear')
            axes[mode_idx, wl_idx].axis('off')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if save_path is not None and isinstance(save_path, str):
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"✓ 原版标签图像已保存到: {save_path}")
    
    plt.show()
