import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ODNN_functions import generate_fields_ts, create_labels
from light_propagation_simulation_qz import propagation


class MultiModeMultiWavelengthDataGenerator:
    """多模式多波长数据生成器"""
    
    def __init__(self, config):
        """
        初始化多模式多波长数据生成器
        
        参数:
            config: 配置对象，包含必要的参数
        """
        self.config = config
        self.modes = None  # 用于存储加载的MMF模式数据
        self.visibility_value = 0.0
        self.training_losses = []
        
        # 生成正交偏移并更新配置
        if not hasattr(config, 'offsets') or config.offsets is None or len(config.offsets) == 0:
            print("正在生成模式和波长的正交偏移...")
            config.offsets = self.generate_orthogonal_offsets()
        elif isinstance(config.offsets, list) and not isinstance(config.offsets[0], list):
            print("使用配置中的基本偏移，为每个模式添加正交偏移...")
            config.offsets = self.generate_combined_offsets(config.offsets)
        
        print(f"初始化多模式多波长数据生成器:")
        print(f"  - 模式数量: {self.config.num_modes}")
        print(f"  - 波长数量: {len(self.config.wavelengths)}")
        print(f"  - 波长列表: {[f'{wl*1e9:.0f}nm' for wl in self.config.wavelengths]}")
        print(f"  - 偏移设置: {self.config.offsets}")
    
    def _get_label_position(self, wl_idx, mode_idx):
        """
        获取特定波长和模式的标签位置
        
        参数:
            wl_idx: 波长索引
            mode_idx: 模式索引
                
        返回:
            tuple: (y_pos, x_pos) 标签在图像中的坐标
        """
        # 获取中心点
        center = self.config.layer_size // 2
        
        # 从配置的偏移中获取位置
        if isinstance(self.config.offsets[0], list):
            # 格式: offsets[mode_idx][wl_idx]
            offset = self.config.offsets[mode_idx][wl_idx]
        else:
            # 简单格式，可能只有波长偏移
            offset = self.config.offsets[wl_idx]
        
        # 计算最终位置
        y_pos = center + offset[1]  # Y坐标对应第二个元素
        x_pos = center + offset[0]  # X坐标对应第一个元素
        
        # 确保在有效范围内
        margin = 20
        y_pos = max(margin, min(y_pos, self.config.layer_size - margin))
        x_pos = max(margin, min(x_pos, self.config.layer_size - margin))
        
        return int(y_pos), int(x_pos)


    def generate_orthogonal_offsets(self):
        """
        生成模式和波长的正交偏移
        - 波长沿水平方向(X轴)偏移
        - 模式沿垂直方向(Y轴)偏移
        
        返回:
            list: 每个模式和波长组合的偏移列表
        """
        num_modes = self.config.num_modes
        num_wavelengths = len(self.config.wavelengths)
        
        # 检查是否有波长偏移配置
        wl_offsets = None
        if hasattr(self.config, 'wavelength_offsets') and self.config.wavelength_offsets is not None:
            wl_offsets = self.config.wavelength_offsets
        else:
            # 默认波长偏移
            x_offset_step = 20
            wl_offsets = []
            for wl_idx in range(num_wavelengths):
                x_offset = wl_idx * x_offset_step - ((num_wavelengths - 1) * x_offset_step) / 2
                wl_offsets.append((int(x_offset), 0))
        
        # 检查是否有模式偏移配置
        mode_offsets = None
        if hasattr(self.config, 'mode_offsets') and self.config.mode_offsets is not None:
            mode_offsets = self.config.mode_offsets
        else:
            # 默认模式偏移
            y_offset_step = 20
            mode_offsets = []
            for mode_idx in range(num_modes):
                y_offset = mode_idx * y_offset_step - ((num_modes - 1) * y_offset_step) / 2
                mode_offsets.append((0, int(y_offset)))
        
        # 生成所有模式和波长组合的偏移
        offsets = []
        for mode_idx in range(num_modes):
            mode_y_offset = mode_offsets[mode_idx][1]
            mode_offsets_list = []
            
            for wl_idx in range(num_wavelengths):
                wl_x_offset = wl_offsets[wl_idx][0]
                # 组合偏移：波长在X轴，模式在Y轴
                combined_offset = (wl_x_offset, mode_y_offset)
                mode_offsets_list.append(combined_offset)
            
            offsets.append(mode_offsets_list)
        
        print(f"生成的正交偏移: {offsets}")
        return offsets
    
    def generate_combined_offsets(self, base_offsets):
        """
        基于波长基础偏移和模式偏移，生成组合偏移
        
        参数:
            base_offsets: 每个波长的基础偏移
        
        返回:
            list: 每个模式和波长组合的偏移列表
        """
        num_modes = self.config.num_modes
        num_wavelengths = len(self.config.wavelengths)
        
        # 检查波长偏移数量是否匹配
        if len(base_offsets) != num_wavelengths:
            print(f"警告: 提供的波长偏移数量({len(base_offsets)})与波长数量({num_wavelengths})不匹配")
            # 如果不匹配，截断或扩展列表
            if len(base_offsets) > num_wavelengths:
                base_offsets = base_offsets[:num_wavelengths]
            else:
                # 扩展列表，复制最后一个偏移
                last_offset = base_offsets[-1] if base_offsets else (0, 0)
                base_offsets.extend([last_offset] * (num_wavelengths - len(base_offsets)))
        
        # 检查是否有模式偏移配置
        mode_offsets = None
        if hasattr(self.config, 'mode_offsets') and self.config.mode_offsets is not None:
            mode_offsets = self.config.mode_offsets
            # 检查模式偏移数量是否匹配
            if len(mode_offsets) != num_modes:
                print(f"警告: 提供的模式偏移数量({len(mode_offsets)})与模式数量({num_modes})不匹配")
                # 如果不匹配，截断或扩展列表
                if len(mode_offsets) > num_modes:
                    mode_offsets = mode_offsets[:num_modes]
                else:
                    # 扩展列表，复制最后一个偏移
                    last_offset = mode_offsets[-1] if mode_offsets else (0, 0)
                    mode_offsets.extend([last_offset] * (num_modes - len(mode_offsets)))
        else:
            # 默认模式偏移
            y_offset_step = 20
            mode_offsets = []
            for mode_idx in range(num_modes):
                y_offset = mode_idx * y_offset_step - ((num_modes - 1) * y_offset_step) / 2
                mode_offsets.append((0, int(y_offset)))
        
        # 生成组合偏移
        combined_offsets = []
        for mode_idx in range(num_modes):
            mode_offset = mode_offsets[mode_idx]
            mode_combined_offsets = []
            
            for wl_idx in range(num_wavelengths):
                wl_offset = base_offsets[wl_idx]
                # 组合偏移：波长在X轴，模式在Y轴
                combined_offset = (wl_offset[0], mode_offset[1])
                mode_combined_offsets.append(combined_offset)
            
            combined_offsets.append(mode_combined_offsets)
        
        print(f"生成的组合偏移: {combined_offsets}")
        return combined_offsets
    
    def generate_radial_offsets(self):
        """
        生成模式和波长径向偏移的标签位置
        波长决定径向距离，模式决定角度
        
        返回:
            dict: 每个波长的位置，格式为{wl_idx: [(y1,x1), (y2,x2), ...]}
        """
        center = self.config.layer_size // 2
        margin = 20
        max_radius = center - margin
        
        # 波长映射到径向距离（短波长在内，长波长在外）
        radii = []
        for wl_idx in range(len(self.config.wavelengths)):
            # 计算半径 - 从内到外
            radius = max_radius * (0.3 + 0.7 * (wl_idx + 1) / len(self.config.wavelengths))
            radii.append(radius)
        
        # 模式映射到角度
        angles = []
        for mode_idx in range(self.config.num_modes):
            # 计算角度 - 均匀分布在圆周上
            angle = 2 * np.pi * mode_idx / self.config.num_modes
            angles.append(angle)
        
        # 组合偏移量
        offsets = {}
        for wl_idx in range(len(self.config.wavelengths)):
            offsets[wl_idx] = []
            for mode_idx in range(self.config.num_modes):
                # 使用极坐标计算偏移
                radius = radii[wl_idx]
                angle = angles[mode_idx]
                
                x_offset = int(radius * np.cos(angle))
                y_offset = int(radius * np.sin(angle))
                
                offsets[wl_idx].append((y_offset, x_offset))
                print(f"波长 {wl_idx}, 模式 {mode_idx}: 位置 ({y_offset},{x_offset}), 半径 {radius:.1f}, 角度 {angle:.2f}rad")
        
        return offsets
    
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
            print(f"模式 {i+1} 振幅范围: {np.min(mode_amp):.4f} - {np.max(mode_amp):.4f}")
        
        # 对每个模式单独归一化
        normalized_data = np.zeros_like(MMF_data, dtype=np.complex128)
        for i in range(MMF_data.shape[0]):
            mode_data = MMF_data[i]
            mode_amp = np.abs(mode_data)
            max_amp = np.max(mode_amp)
            if max_amp > 0:
                normalized_data[i] = mode_data / max_amp
        
        # 转换为PyTorch张量
        return torch.tensor(normalized_data, dtype=torch.complex64)

    def generate_input_data(self):
        """生成输入数据"""
        import numpy as np  # 确保导入 numpy
        
        # 如果模式数据尚未加载，则加载它
        if self.modes is None:
            try:
                self.modes = self.load_mmf_data()
                print(f"已加载 {self.modes.shape[0]} 个模式")
            except FileNotFoundError:
                print("警告: 找不到MMF数据文件，使用随机生成的模式")
                # 随机生成模式
                self.modes = torch.randn(self.config.num_modes, 
                                        self.config.field_size, 
                                        self.config.field_size, 
                                        dtype=torch.complex64)
                # 归一化
                for i in range(self.modes.shape[0]):
                    mode_amp = torch.abs(self.modes[i])
                    max_amp = torch.max(mode_amp)
                    if max_amp > 0:
                        self.modes[i] = self.modes[i] / max_amp
        
        # 检查模式数据形状
        if self.modes.shape[0] < self.config.num_modes:
            raise ValueError(f"需要至少 {self.config.num_modes} 个模式，但只有 {self.modes.shape[0]} 个")
        
        # 调整大小以匹配配置中的场大小
        if self.modes.shape[1] != self.config.field_size or self.modes.shape[2] != self.config.field_size:
            print(f"调整模式大小从 {self.modes.shape[1]}x{self.modes.shape[2]} 到 {self.config.field_size}x{self.config.field_size}")
            # ... 调整大小代码 ...
        
        # 预先创建输出张量
        input_data = torch.zeros(
            (self.config.num_modes, len(self.config.wavelengths), 
            self.config.field_size, self.config.field_size),
            dtype=torch.complex64
        )
        
        # 填充张量
        for mode_idx in range(self.config.num_modes):
            for wl_idx, wavelength in enumerate(self.config.wavelengths):
                # 使用模式数据作为输入场
                field = self.modes[mode_idx].clone()
                
                # 添加波长相关的相位调制 - 使用 NumPy 计算相位因子
                phase_angle = 2 * np.pi * wl_idx / len(self.config.wavelengths)
                phase_factor = np.exp(1j * phase_angle)
                field = field * phase_factor
                
                # 确保输入数据的大小与配置匹配
                if field.shape[0] != self.config.field_size or field.shape[1] != self.config.field_size:
                    raise ValueError(f"输入场大小 {field.shape} 与配置中的场大小 {self.config.field_size}x{self.config.field_size} 不匹配")
                
                input_data[mode_idx, wl_idx] = field
        
        return input_data


    def generate_labels(self):
        """生成标签数据"""
        labels = []
        
        for mode_idx in range(self.config.num_modes):
            mode_labels = []
            for wl_idx, wl in enumerate(self.config.wavelengths):
                # 创建空白标签
                label = torch.zeros((self.config.layer_size, self.config.layer_size))
                
                # 获取标签位置并设置点
                y_pos, x_pos = self._get_label_position(wl_idx, mode_idx)
                label[y_pos, x_pos] = 1.0
                
                mode_labels.append(label)
            labels.append(torch.stack(mode_labels))
        
        return torch.stack(labels)

    
    def create_dataset(self, num_samples=None):
        """创建数据集"""
        if num_samples is None:
            num_samples = self.config.batch_size
        
        # 生成输入数据和标签
        inputs = self.generate_input_data()
        labels = self.generate_labels()
        
        # 创建包含多个样本的数据集
        input_samples = []
        label_samples = []
        
        for _ in range(num_samples):
            # 添加一些随机变化以创建不同的样本
            # 例如，随机相位或小的位置变化
            modified_inputs = []
            for mode_idx in range(self.config.num_modes):
                mode_inputs = []
                for wl_idx in range(len(self.config.wavelengths)):
                    # 添加随机相位
                    random_phase = torch.exp(1j * torch.rand(1) * 2 * torch.pi)
                    modified_input = inputs[mode_idx, wl_idx] * random_phase
                    mode_inputs.append(modified_input)
                modified_inputs.append(torch.stack(mode_inputs))
            
            # 将输入和标签添加到样本列表
            input_samples.append(torch.stack(modified_inputs))
            label_samples.append(labels.clone())  # 标签保持不变
        
        # 创建数据集
        dataset = TensorDataset(torch.stack(input_samples), torch.stack(label_samples))
        return dataset
    
    def create_dataloader(self, num_samples=None, batch_size=None, shuffle=True):
        """创建数据加载器"""
        if num_samples is None:
            num_samples = self.config.batch_size * 10  # 默认生成10个批次的数据
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # 创建数据集
        dataset = self.create_dataset(num_samples)
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader
    
    def visualize_data(self, save_path=None):
        """可视化生成的输入数据和标签"""
        # 生成数据
        image_data = self.generate_input_data()
        
        # 创建图像网格
        fig, axes = plt.subplots(self.config.num_modes, len(self.config.wavelengths) * 2, 
                            figsize=(len(self.config.wavelengths) * 6, self.config.num_modes * 3))
        
        # 遍历每个模式和波长
        for mode_idx in range(self.config.num_modes):
            for wl_idx, wl in enumerate(self.config.wavelengths):
                # 输入场振幅
                ax_input = axes[mode_idx, wl_idx * 2]
                input_amp = torch.abs(image_data[mode_idx, wl_idx])
                im_input = ax_input.imshow(input_amp.numpy(), cmap='viridis')
                ax_input.set_title(f'模式 {mode_idx+1}, 波长 {int(wl*1e9)}nm\n输入场')
                ax_input.axis('off')
                plt.colorbar(im_input, ax=ax_input, fraction=0.046, pad=0.04)
                
                # 标签 - 使用纯黑背景
                ax_label = axes[mode_idx, wl_idx * 2 + 1]
                ax_label.set_facecolor('black')
                
                # 创建黑色背景图像
                black_bg = np.zeros((self.config.layer_size, self.config.layer_size))
                im_label = ax_label.imshow(black_bg, cmap='viridis', vmin=0, vmax=1)
                
                # 获取标签位置并绘制
                y_pos, x_pos = self._get_label_position(wl_idx, mode_idx)
                ax_label.scatter(x_pos, y_pos, s=50, c='yellow', zorder=10)
                
                ax_label.set_title(f'模式 {mode_idx+1}, 波长 {int(wl*1e9)}nm\n标签')
                ax_label.axis('off')
                
                # 为标签图添加颜色条
                cbar = plt.colorbar(im_label, ax=ax_label, fraction=0.046, pad=0.04)
                cbar.set_ticks([0, 0.5, 1.0])
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_label_positions(self, save_path=None):
        """
        可视化标签位置
        
        参数:
            save_path: 保存路径
        
        返回:
            fig, ax: 图形和坐标轴对象
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 定义模式的标记和颜色
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '8']
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # 绘制所有模式和波长组合的位置
        for mode_idx, mode_offsets in enumerate(self.config.offsets):
            for wl_idx, offset in enumerate(mode_offsets):
                x, y = offset
                wavelength = self.config.wavelengths[wl_idx] * 1e9  # 转换为nm
                
                # 绘制点
                marker = markers[mode_idx % len(markers)]
                color = colors[wl_idx % len(colors)]
                ax.scatter(x, y, marker=marker, color=color, s=100, 
                           label=f'Mode {mode_idx}, λ={wavelength:.0f}nm' if wl_idx == 0 else "")
                
                # 添加标签
                ax.text(x, y+5, f'M{mode_idx}, λ={wavelength:.0f}', 
                        ha='center', va='bottom', fontsize=8)
                
                # 添加圆圈突出显示
                circle = Circle((x, y), radius=10, fill=False, 
                                linestyle='--', linewidth=1, color='black')
                ax.add_patch(circle)
        
        # 设置图形属性
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.set_title('Label Positions for Different Modes and Wavelengths')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 创建自定义图例：模式
        mode_handles = []
        mode_labels = []
        for i in range(self.config.num_modes):
            marker = markers[i % len(markers)]
            handle = plt.Line2D([0], [0], marker=marker, color='gray', 
                               markersize=10, linestyle='None')
            mode_handles.append(handle)
            mode_labels.append(f'Mode {i}')
        
        # 创建自定义图例：波长
        wl_handles = []
        wl_labels = []
        for i, wl in enumerate(self.config.wavelengths):
            color = colors[i % len(colors)]
            handle = plt.Line2D([0], [0], marker='o', color=color, 
                               markersize=10, linestyle='None')
            wl_handles.append(handle)
            wl_labels.append(f'λ={wl*1e9:.0f}nm')
        
        # 添加图例
        ax.legend(mode_handles, mode_labels, loc='upper left', 
                 title='Modes', framealpha=0.7)
        ax.legend(wl_handles, wl_labels, loc='upper right', 
                 title='Wavelengths', framealpha=0.7)
        
        # 调整坐标轴范围，确保所有点都可见
        x_values = [offset[0] for mode_offsets in self.config.offsets for offset in mode_offsets]
        y_values = [offset[1] for mode_offsets in self.config.offsets for offset in mode_offsets]
        
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        
        # 添加一些边距
        margin = 30
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"标签位置图已保存至: {save_path}")
        
        return fig, ax

