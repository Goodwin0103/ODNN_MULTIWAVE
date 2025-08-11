import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def place_square_by_index(index, array_size=100, square_size=5, distance=15):
    """
    在指定位置放置方块
    
    参数:
        index: 方块索引
        array_size: 数组大小
        square_size: 方块大小
        distance: 方块间距
        
    返回:
        包含方块的二维数组
    """
    # 创建空数组
    array = np.zeros((array_size, array_size))
    
    # 计算布局
    if index < 0 or index > 9:
        raise ValueError(f"索引必须在0-9之间，当前值: {index}")
    
    # 确定行和列位置
    if index < 3:  # 第一行
        row = 0
        col = index
    elif index < 7:  # 第二行
        row = 1
        col = index - 3
    else:  # 第三行
        row = 2
        col = index - 7
    
    # 计算方块中心坐标
    center_y = array_size // 2 - distance + row * distance
    center_x = array_size // 2 - (3 * distance) // 2 + col * distance
    
    # 放置方块
    y_start = center_y - square_size // 2
    y_end = center_y + square_size // 2 + (1 if square_size % 2 else 0)
    x_start = center_x - square_size // 2
    x_end = center_x + square_size // 2 + (1 if square_size % 2 else 0)
    
    array[y_start:y_end, x_start:x_end] = 1.0
    
    return array

def generate_fields_ts(num_samples, num_wavelengths=3, array_size=100):
    """
    生成测试场
    
    参数:
        num_samples: 样本数量
        num_wavelengths: 波长数量
        array_size: 数组大小
        
    返回:
        测试场张量
    """
    fields = []
    
    for _ in range(num_samples):
        index = np.random.randint(0, 10)
        field = place_square_by_index(index, array_size)
        fields.append(field)
    
    # 转换为张量 [N, H, W]
    fields_tensor = torch.tensor(np.array(fields), dtype=torch.float32)
    
    # 添加通道维度 [N, 1, H, W]
    fields_tensor = fields_tensor.unsqueeze(1)
    
    return fields_tensor

def create_labels(fields_tensor, num_classes=10):
    """
    为场创建标签
    
    参数:
        fields_tensor: 场张量 [N, 1, H, W]
        num_classes: 类别数量
        
    返回:
        标签张量 [N, num_classes]
    """
    batch_size = fields_tensor.shape[0]
    labels = torch.zeros(batch_size, num_classes)
    
    for i in range(batch_size):
        field = fields_tensor[i, 0].numpy()  # [H, W]
        
        # 通过计算方块位置确定类别
        for j in range(num_classes):
            template = place_square_by_index(j, field.shape[0])
            if np.sum(field * template) > 0:
                labels[i, j] = 1.0
                break
    
    return labels

class DataGenerator:
    """数据生成器"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
    
    def load_mode_data(self, file_path='eigenmodes_OM4.npy'):
        """
        从文件加载模式数据
        
        参数:
            file_path: 模式数据文件路径
            
        返回:
            模式数据张量
        """
        try:
            eigenmodes = np.load(file_path)
            print(f"已加载模式数据，形状: {eigenmodes.shape}")
            
            # 选择指定数量的模式
            num_modes = min(self.config.num_modes, eigenmodes.shape[2])
            modes_data = eigenmodes[:, :, :num_modes].transpose(2, 0, 1)
            
            # 转换为张量
            modes_tensor = torch.tensor(modes_data, dtype=torch.complex64)
            
            # 归一化
            for i in range(modes_tensor.shape[0]):
                mode = modes_tensor[i]
                modes_tensor[i] = mode / torch.max(torch.abs(mode))
            
            return modes_tensor
            
        except Exception as e:
            print(f"加载模式数据失败: {e}")
            raise
    
    def create_training_dataset(self):
        """
        创建训练数据集
        
        返回:
            数据加载器
        """
        # 加载模式数据
        modes = self.load_mode_data()
        num_modes = modes.shape[0]
        num_wavelengths = len(self.config.wavelengths)
        
        # 创建输入数据和标签
        inputs = []
        labels = []
        
        for mode_idx in range(num_modes):
            for wl_idx in range(num_wavelengths):
                # 准备输入数据
                input_field = modes[mode_idx].unsqueeze(0)  # [1, H, W]
                inputs.append(input_field)
                
                # 创建标签
                label = torch.zeros(num_modes, num_wavelengths)
                label[mode_idx, wl_idx] = 1.0
                labels.append(label)
        
        # 转换为张量
        inputs = torch.stack(inputs)  # [N, 1, H, W]
        labels = torch.stack(labels)  # [N, num_modes, num_wavelengths]
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        return dataloader
