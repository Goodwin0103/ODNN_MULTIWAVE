# main.py
import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

import torch
from config.config import ODNNConfig

# 修复导入路径 - 注意拼写
try:
    from training.trainer import ODNNTrainer  # 不是 trainning
except ImportError:
    print("Error: training module not found. Please check directory name.")
    sys.exit(1)

try:
    from model.odnn_models import WavelengthDependentODNNModel
except ImportError:
    try:
        from models.odnn_model import WavelengthDependentODNNModel
    except ImportError:
        print("Error: model module not found")
        sys.exit(1)

try:
    from data.data_loader import ODNNDataLoader
    from utils.visualization import plot_training_results, plot_visibility_vs_layers
    from utils.metrics import calculate_visibility
    from utils.save_utils import create_experiment_summary
except ImportError as e:
    print(f"Import error: {e}")
    print("请确保所有模块都已创建")
    sys.exit(1)

# 确保能找到ODNN_functions
try:
    from ODNN_functions import create_evaluation_regions
except ImportError:
    print("Warning: ODNN_functions not found, using local implementation")
    try:
        from utils.odnn_functions import create_evaluation_regions
    except ImportError:
        print("Error: No ODNN_functions found")
        sys.exit(1)

def main():
    """主函数"""
    # 初始化配置
    config = ODNNConfig()
    
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device}')
    
    # 创建保存目录
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    try:
        # 加载数据
        print("正在加载数据...")
        data_loader = ODNNDataLoader(config)
        train_loader, train_dataset = data_loader.create_dataloader()
        test_loader = train_loader  # 使用相同的数据进行测试
        
        # 创建检测区域
        evaluation_regions = create_evaluation_regions(
            config.LAYER_SIZE, config.LAYER_SIZE, 
            config.NUM_MODES, config.FOCUS_RADIUS, config.DETECT_SIZE
        )
        
        print("数据加载成功！")
        print(f"训练数据形状: {len(train_dataset)}")
        print(f"检测区域数量: {len(evaluation_regions)}")
        
        # 存储结果
        all_losses = []
        all_phase_masks = []
        all_weights_pred = []
        all_models = []
        visibility_list = []
        
        # 训练不同层数的模型
        for num_layer in config.NUM_LAYER_OPTIONS:
            print(f"\n{'='*50}")
            print(f"训练 {num_layer} 层 ODNN")
            print(f"{'='*50}")
            
            # 创建模型
            model = WavelengthDependentODNNModel(
                num_layers=num_layer,
                layer_size=config.LAYER_SIZE,
                z_layers=config.Z_LAYERS,
                z_prop=config.Z_PROP,
                pixel_size=config.PIXEL_SIZE,
                wavelengths=config.WAVELENGTHS
            )
            
            # 创建训练器
            trainer = ODNNTrainer(model, config)
            
            # 训练模型
            losses = trainer.train(train_loader, num_layer)
            
            # 评估模型
            weights_pred = trainer.evaluate(test_loader, evaluation_regions)
            
            # 计算可见度
            visibility = calculate_visibility(weights_pred, config.NUM_MODES)
            
            # 保存结果
            all_losses.append(losses)
            all_phase_masks.append(model.get_phase_masks())
            all_weights_pred.append(weights_pred)
            all_models.append(model)
            visibility_list.append(visibility)
            
            print(f"模型 {num_layer} 层训练完成，可见度: {visibility:.4f}")
            
            # 清理GPU内存
            torch.cuda.empty_cache()
        
        # 保存所有结果
        results_dict = {
            'visibility_list': visibility_list,
            'num_layers': config.NUM_LAYER_OPTIONS,
            'final_losses': [losses[-1] for losses in all_losses]
        }
        
        # 创建实验总结
        create_experiment_summary(config, results_dict)
        
        # 可视化结果
        try:
            plot_training_results(all_losses, config.NUM_LAYER_OPTIONS)
            plot_visibility_vs_layers(visibility_list, config.NUM_LAYER_OPTIONS, config.NUM_MODES)
        except Exception as e:
            print(f"可视化出错: {e}")
        
        print("\n🎉 训练完成！")
        print("结果已保存到:", config.SAVE_DIR)
        
    except Exception as e:
        print(f"运行时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
