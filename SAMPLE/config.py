import torch
import os

class Config:
    def __init__(self, field_size=None, layer_size=None, wavelengths=None, offsets=None, 
                 detectsize=10, num_layers=3, epochs=200, batch_size=1, 
                 learning_rate=0.01, save_dir='results', num_epochs=None,
                 # 🔥 新增改进配置选项
                 use_dispersion=False,  # 是否使用色散效应
                 use_differential_detection=True,  # 是否使用差分检测
                 loss_weights=None,  # 自定义损失权重
                 optimizer_type='adamw',  # 优化器类型
                 scheduler_type='cosine',  # 学习率调度器类型
                 early_stopping_patience=100,  # 早停耐心值
                 gradient_clip_norm=1.0,  # 梯度裁剪范数
                 warmup_epochs=None):  # 预热轮数
        
        # 🔥 首先设置基础参数 - 修复顺序问题
        self.epochs = epochs if num_epochs is None else num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.detectsize = detectsize
        
        # 设置保存目录
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 波长配置
        if wavelengths is None:
            self.wavelengths = [450e-9, 650e-9]  # 默认蓝光和红光
        else:
            self.wavelengths = wavelengths
        
        # 物理参数
        self.pixel_size = 1e-6  # 1微米像素
        
        # 场尺寸自动计算
        if field_size is None:
            # 根据波长和检测区域自动计算合适的场尺寸
            min_wavelength = min(self.wavelengths)
            # 确保有足够的空间进行衍射分离
            self.field_size = max(128, detectsize * 8)
        else:
            self.field_size = field_size
        
        # 层尺寸
        if layer_size is None:
            self.layer_size = self.field_size
        else:
            self.layer_size = layer_size
        
        # 检测区域偏移
        if offsets is None:
            # 自动计算偏移，使检测区域分离
            separation = self.field_size // 4
            if len(self.wavelengths) == 2:
                self.offsets = [(-separation, 0), (separation, 0)]
            else:
                # 多波长情况下的圆形分布
                import numpy as np
                angles = np.linspace(0, 2*np.pi, len(self.wavelengths), endpoint=False)
                self.offsets = [(int(separation * np.cos(angle)), 
                               int(separation * np.sin(angle))) 
                               for angle in angles]
        else:
            self.offsets = offsets
        
        # 验证配置
        assert len(self.offsets) == len(self.wavelengths), "偏移数量必须与波长数量匹配"
        
        # 计算检测区域边界
        self.detect_size = detectsize
        
        # 🔥 新增改进配置参数
        self.use_dispersion = use_dispersion
        self.use_differential_detection = use_differential_detection
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_norm = gradient_clip_norm
        
        # 🔥 修复：预热轮数设置（现在 self.epochs 已经存在）
        if warmup_epochs is None:
            self.warmup_epochs = min(50, self.epochs // 10)
        else:
            self.warmup_epochs = warmup_epochs
        
        # 🔥 损失权重设置
        if loss_weights is None:
            self.loss_weights = {
                'efficiency': 2.0,
                'separation': 1.5,
                'crosstalk': 1.0,
                'concentration': 0.8,
                'smoothing': 0.1
            }
        else:
            self.loss_weights = loss_weights
    
    def print_config(self):
        """打印配置信息（扩展版）"""
        print("=" * 60)
        print("📋 IMPROVED CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"🔲 Field size: {self.field_size} × {self.field_size} pixels")
        print(f"🔲 Layer size: {self.layer_size} × {self.layer_size} pixels")
        print(f"📏 Pixel size: {self.pixel_size*1e6:.1f} μm")
        print(f"🌈 Wavelengths: {[int(wl*1e9) for wl in self.wavelengths]} nm")
        print(f"🎯 Detection offsets: {self.offsets}")
        print(f"🔍 Detection size: {self.detectsize} × {self.detectsize} pixels")
        print(f"🏗️  Number of layers: {self.num_layers}")
        print(f"🔄 Training epochs: {self.epochs}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"📈 Learning rate: {self.learning_rate}")
        print(f"💻 Device: {self.device}")
        print(f"💾 Save directory: {self.save_dir}")
        
        # 🔥 新增改进配置显示
        print(f"\n🔥 改进配置:")
        print(f"  色散效应: {'启用' if self.use_dispersion else '禁用'}")
        print(f"  差分检测: {'启用' if self.use_differential_detection else '禁用'}")
        print(f"  优化器: {self.optimizer_type.upper()}")
        print(f"  学习率调度: {self.scheduler_type}")
        print(f"  早停耐心: {self.early_stopping_patience}")
        print(f"  梯度裁剪: {self.gradient_clip_norm}")
        print(f"  预热轮数: {self.warmup_epochs}")
        
        print(f"\n📊 损失权重:")
        for key, weight in self.loss_weights.items():
            print(f"  {key}: {weight}")
        
        print("=" * 60)

    def get_detection_regions(self):
        """获取所有检测区域的坐标"""
        regions = []
        center = self.field_size // 2
        
        for offset in self.offsets:
            # 计算检测区域中心
            detect_center_x = center + offset[0]
            detect_center_y = center + offset[1]
            
            # 计算检测区域边界
            half_size = self.detect_size // 2
            x_start = max(0, detect_center_x - half_size)
            x_end = min(self.field_size, detect_center_x + half_size)
            y_start = max(0, detect_center_y - half_size)
            y_end = min(self.field_size, detect_center_y + half_size)
            
            regions.append((x_start, x_end, y_start, y_end))
        
        return regions
    
    def validate_config(self):
        """验证配置的有效性"""
        errors = []
        
        # 检查场尺寸
        if self.field_size < 64:
            errors.append("场尺寸太小，建议至少64×64")
        
        # 检查检测区域是否重叠
        regions = self.get_detection_regions()
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions[i+1:], i+1):
                if self._regions_overlap(region1, region2):
                    errors.append(f"检测区域 {i} 和 {j} 重叠")
        
        # 检查训练参数
        if self.epochs < 10:
            errors.append("训练轮数太少")
        
        if self.learning_rate <= 0 or self.learning_rate > 1:
            errors.append("学习率应在(0, 1]范围内")
        
        if errors:
            print("⚠️  配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("✅ 配置验证通过")
            return True
    
    def _regions_overlap(self, region1, region2):
        """检查两个区域是否重叠"""
        x1_start, x1_end, y1_start, y1_end = region1
        x2_start, x2_end, y2_start, y2_end = region2
        
        return not (x1_end <= x2_start or x2_end <= x1_start or 
                   y1_end <= y2_start or y2_end <= y1_start)
