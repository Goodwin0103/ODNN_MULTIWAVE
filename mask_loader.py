# mask_loader.py
import numpy as np

class MaskLoader:
    """简化的相位掩码加载器"""
    
    def __init__(self, config):
        """
        初始化掩码加载器
        
        参数:
            config: 配置对象
        """
        self.config = config
        print("🔧 初始化掩码加载器...")
        print(f"  - 默认层数: {config.default_num_layers}")
    
    def create_fallback_masks(self, num_layers=None):
        """
        创建备用聚焦掩码
        
        参数:
            num_layers: 层数，如果为None则使用配置中的默认值
        
        返回:
            List[List[np.ndarray]]: 掩码列表 [num_layers][num_wavelengths]
        """
        if num_layers is None:
            num_layers = self.config.default_num_layers
            
        print(f"⚠ 创建备用聚焦掩码...")
        print(f"  - 层数: {num_layers}")
        print(f"  - 波长数: {len(self.config.wavelengths)}")
        
        def create_focusing_mask(size, wavelength, focal_length, pixel_size):
            center = size // 2
            y, x = np.ogrid[:size, :size]
            r_squared = ((x - center) * pixel_size) ** 2 + ((y - center) * pixel_size) ** 2
            k = 2 * np.pi / wavelength
            phase = -k * r_squared / (2 * focal_length)
            return np.mod(phase, 2 * np.pi)
        
        masks = []
        
        for layer_idx in range(num_layers):
            layer_masks = []
            focal_length = self.config.fallback_focal_lengths[layer_idx % len(self.config.fallback_focal_lengths)]
            
            for wl_idx, wavelength in enumerate(self.config.wavelengths):
                mask = create_focusing_mask(
                    self.config.layer_size, 
                    wavelength, 
                    focal_length, 
                    self.config.pixel_size
                )
                layer_masks.append(mask)
            
            masks.append(layer_masks)
        
        print(f"✓ 创建了 {len(masks)} 层备用掩码")
        return masks
    
    def get_masks_for_simulation(self, trained_masks=None, num_layers=None):
        """
        获取用于仿真的掩码
        
        参数:
            trained_masks: 训练好的掩码
            num_layers: 层数，如果为None则使用配置中的默认值
        
        返回:
            List[List[np.ndarray]]: 掩码列表
        """
        if num_layers is None:
            num_layers = self.config.default_num_layers
            
        if trained_masks is not None:
            print("✓ 使用训练好的相位掩码进行仿真")
            return trained_masks
        else:
            print("⚠ 使用备用掩码进行仿真")
            return self.create_fallback_masks(num_layers)
