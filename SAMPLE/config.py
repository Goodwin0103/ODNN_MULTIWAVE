# 在原有config.py基础上添加新的配置选项

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
        
        # ... 原有初始化代码 ...
        
        # 🔥 新增配置参数
        self.use_dispersion = use_dispersion
        self.use_differential_detection = use_differential_detection
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_norm = gradient_clip_norm
        
        # 预热轮数设置
        if warmup_epochs is None:
            self.warmup_epochs = min(50, self.epochs // 10)
        else:
            self.warmup_epochs = warmup_epochs
        
        # 损失权重设置
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
        
        # ... 原有验证代码 ...
    
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
