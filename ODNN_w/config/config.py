import numpy as np

class ODNNConfig:
    """ODNN配置类"""
    
    # 数据参数
    FIELD_SIZE = 50
    LAYER_SIZE = 200
    NUM_MODES = 3
    FOCUS_RADIUS = 5
    DETECT_SIZE = 15
    
    # 光学参数
    WAVELENGTHS = np.array([450e-9, 550e-9, 650e-9])  # RGB波长
    PIXEL_SIZE = 1e-6
    Z_LAYERS = 40e-6      # 层间距离
    Z_PROP = 300e-6       # 最后传播距离
    
    # 训练参数
    BATCH_SIZE = 3
    EPOCHS = 400
    LEARNING_RATE = 1.99
    GAMMA = 0.99
    
    # 模型参数
    NUM_LAYER_OPTIONS = [1, 2, 3]
    PHASE_OPTION = 4
    PRED_CASE = 1
    
    # 设备设置
    DEVICE = 'cuda:0'
    
    # 保存设置
    SAVE_DIR = "./results_MC_wavelength/"
    CHECKPOINT_DIR = "./checkpoints/"
    
    @classmethod
    def get_wavelength_names(cls):
        return [f"{wl*1e9:.0f}nm" for wl in cls.WAVELENGTHS]
