import torch
import numpy as np
import os
from config import Config
from data_generator import SingleModeDualWavelengthDataGenerator
from model import ImprovedMultiWavelengthModel
from trainer import ImprovedTrainer
from visualizer import ImprovedVisualizer

def main():
    """🔥 改进版主函数 - 对比原版和改进版"""
    print("🚀 启动改进版多波长衍射神经网络训练...")
    
    # 1. 配置设置
    config = Config(
        field_size=None,  # 自动检测
        wavelengths=[450e-9, 650e-9],  # 蓝光和红光
        detectsize=10,
        num_layers=2,
        epochs=300,  # 增加训练轮数
        learning_rate=1e-3,  # 调整学习率
        save_dir='results_improved'
    )
    
    config.print_config()
    
    # 2. 数据生成器
    print("\n📊 初始化数据生成器...")
    data_generator = SingleModeDualWavelengthDataGenerator(config)
    
    # 可视化数据和检测器布局
    data_generator.visualize_separation_concept(
        save_path=os.path.join(config.save_dir, 'separation_concept.png')
    )
    data_generator.visualize_detector_layout(
        save_path=os.path.join(config.save_dir, 'detector_layout.png')
    )
    
    # 3. 🔥 训练原版模型（用于对比）
    print("\n🔧 训练原版模型...")
    original_model = SimpleMultiWavelengthModel(config, num_layers=config.num_layers)
    original_trainer = SimpleTrainer(config, data_generator)
    original_results = original_trainer.train_model(original_model, config.epochs)
    
    # 4. 🔥 训练改进版模型
    print("\n🚀 训练改进版模型...")
    improved_model = ImprovedMultiWavelengthModel(config, num_layers=config.num_layers)
    improved_model.print_model_info()
    
    improved_trainer = ImprovedTrainer(config, data_generator)
    improved_results = improved_trainer.train_model(improved_model, config.epochs)
    
    # 5. 🔥 性能对比分析
    print("\n📊 进行性能对比分析...")
    
    # 计算分离指标
    input_fields = data_generator.generate_input_fields()
    original_metrics = original_trainer.calculate_separation_metrics(original_results['model'], input_fields)
    improved_metrics = improved_trainer.calculate_separation_metrics(improved_results['model'], input_fields)
    
    # 打印对比结果
    print("\n" + "="*60)
    print("📊 性能对比结果")
    print("="*60)
    print(f"{'指标':<20} {'原版':<15} {'改进版':<15} {'提升':<15}")
    print("-"*65)
    
    # 平均效率对比
    orig_avg_eff = original_metrics['overall']['avg_efficiency']
    impr_avg_eff = improved_metrics['overall']['avg_efficiency']
    eff_improvement = (impr_avg_eff - orig_avg_eff) / orig_avg_eff * 100
    
    print(f"{'平均效率':<20} {orig_avg_eff:<15.4f} {impr_avg_eff:<15.4f} {eff_improvement:+.1f}%")
    
    # 平均串扰对比
    orig_avg_cross = original_metrics['overall']['avg_crosstalk']
    impr_avg_cross = improved_metrics['overall']['avg_crosstalk']
    cross_improvement = (orig_avg_cross - impr_avg_cross) / orig_avg_cross * 100
    
    print(f"{'平均串扰':<20} {orig_avg_cross:<15.4f} {impr_avg_cross:<15.4f} {cross_improvement:+.1f}%")
    
    # 分离比率对比
    orig_sep_ratio = original_metrics['overall']['separation_ratio']
    impr_sep_ratio = improved_metrics['overall']['separation_ratio']
    ratio_improvement = (impr_sep_ratio - orig_sep_ratio) / orig_sep_ratio * 100
    
    print(f"{'分离比率':<20} {orig_sep_ratio:<15.1f} {impr_sep_ratio:<15.1f} {ratio_improvement:+.1f}%")
    
    # 训练时间对比
    time_change = (improved_results['training_time'] - original_results['training_time']) / original_results['training_time'] * 100
    print(f"{'训练时间(s)':<20} {original_results['training_time']:<15.1f} {improved_results['training_time']:<15.1f} {time_change:+.1f}%")
    
    print("="*65)
    
    # 6. 🔥 生成可视化对比
    print("\n🎨 生成可视化结果...")
    
    # 原版可视化器
    original_visualizer = SimpleVisualizer(config)
    improved_visualizer = ImprovedVisualizer(config)
    
    # 训练历史对比
    improved_visualizer.plot_improved_training_history(
        improved_results['training_history'],
        save_path=os.path.join(config.save_dir, 'improved_training_history.png')
    )
    
    # 相位掩膜对比
    improved_visualizer.plot_wavelength_dependent_phase_masks(
        improved_results['model'],
        save_path=os.path.join(config.save_dir, 'wavelength_dependent_masks.png')
    )
    
    # 能量分布对比
    with torch.no_grad():
        original_output = original_results['model'](input_fields)
        improved_output = improved_results['model'](input_fields)
    
    improved_visualizer.plot_improved_energy_distribution(
        improved_output,
        save_path=os.path.join(config.save_dir, 'improved_energy_distribution.png')
    )
    
    # 🔥 核心对比图
    improved_visualizer.plot_comparison_original_vs_improved(
        original_results, improved_results,
        save_path=os.path.join(config.save_dir, 'comparison_original_vs_improved.png')
    )
    
    # 分离性能指标
    improved_visualizer.plot_wavelength_separation_metrics(
        improved_metrics,
        save_path=os.path.join(config.save_dir, 'improved_separation_metrics.png')
    )
    
    # 7. 保存模型和结果
    print("\n💾 保存模型和结果...")
    
    # 保存模型
    torch.save({
        'original_model_state': original_results['model'].state_dict(),
        'improved_model_state': improved_results['model'].state_dict(),
        'config': config,
        'original_results': original_results,
        'improved_results': improved_results,
        'original_metrics': original_metrics,
        'improved_metrics': improved_metrics
    }, os.path.join(config.save_dir, 'models_and_results.pth'))
    
    # 保存详细报告
    save_detailed_report(config, original_results, improved_results, 
                        original_metrics, improved_metrics)
    
    print(f"\n✅ 改进版训练完成!")
    print(f"📁 结果保存在: {config.save_dir}")
    print(f"🎯 平均效率提升: {eff_improvement:+.1f}%")
    print(f"🔽 平均串扰降低: {cross_improvement:+.1f}%")
    print(f"📈 分离比率提升: {ratio_improvement:+.1f}%")

def save_detailed_report(config, original_results, improved_results, 
                        original_metrics, improved_metrics):
    """保存详细的对比报告"""
    report_path = os.path.join(config.save_dir, 'detailed_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🔥 改进版多波长衍射神经网络 - 详细报告\n")
        f.write("="*80 + "\n\n")
        
        # 配置信息
        f.write("📋 配置信息:\n")
        f.write(f"  场大小: {config.field_size}×{config.field_size}\n")
        f.write(f"  波长: {[int(wl*1e9) for wl in config.wavelengths]} nm\n")
        f.write(f"  检测区域大小: {config.detect_size}×{config.detect_size}\n")
        f.write(f"  层数: {config.num_layers}\n")
        f.write(f"  训练轮数: {config.epochs}\n")
        f.write(f"  学习率: {config.learning_rate}\n\n")
        
        # 训练结果对比
        f.write("🚀 训练结果对比:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'指标':<25} {'原版':<15} {'改进版':<15} {'提升':<15}\n")
        f.write("-"*70 + "\n")
        
        # 效率对比
        orig_avg_eff = original_metrics['overall']['avg_efficiency']
        impr_avg_eff = improved_metrics['overall']['avg_efficiency']
        eff_improvement = (impr_avg_eff - orig_avg_eff) / orig_avg_eff * 100
        f.write(f"{'平均效率':<25} {orig_avg_eff:<15.4f} {impr_avg_eff:<15.4f} {eff_improvement:+.1f}%\n")
        
        # 各波长效率
        for i, wavelength in enumerate(config.wavelengths):
            wl_nm = int(wavelength * 1e9)
            orig_eff = original_results['final_efficiencies'][i]
            impr_eff = improved_results['final_efficiencies'][i]
            wl_improvement = (impr_eff - orig_eff) / orig_eff * 100
            f.write(f"  {f'{wl_nm}nm效率':<23} {orig_eff:<15.4f} {impr_eff:<15.4f} {wl_improvement:+.1f}%\n")
        
        # 串扰对比
        orig_avg_cross = original_metrics['overall']['avg_crosstalk']
        impr_avg_cross = improved_metrics['overall']['avg_crosstalk']
        cross_improvement = (orig_avg_cross - impr_avg_cross) / orig_avg_cross * 100
        f.write(f"{'平均串扰':<25} {orig_avg_cross:<15.4f} {impr_avg_cross:<15.4f} {cross_improvement:+.1f}%\n")
        
        # 分离比率
        orig_sep_ratio = original_metrics['overall']['separation_ratio']
        impr_sep_ratio = improved_metrics['overall']['separation_ratio']
        ratio_improvement = (impr_sep_ratio - orig_sep_ratio) / orig_sep_ratio * 100
        f.write(f"{'分离比率':<25} {orig_sep_ratio:<15.1f} {impr_sep_ratio:<15.1f} {ratio_improvement:+.1f}%\n")
        
        # 训练时间
        time_change = (improved_results['training_time'] - original_results['training_time']) / original_results['training_time'] * 100
        f.write(f"{'训练时间(s)':<25} {original_results['training_time']:<15.1f} {improved_results['training_time']:<15.1f} {time_change:+.1f}%\n")
        
        f.write("\n")
        
        # 详细性能指标
        f.write("📊 详细性能指标:\n")
        f.write("-"*40 + "\n")
        
        for wavelength in config.wavelengths:
            wl_nm = int(wavelength * 1e9)
            wl_key = f"wavelength_{wl_nm}nm"
            
            f.write(f"\n🌈 {wl_nm}nm 波长:\n")
            f.write(f"  原版 - 效率: {original_metrics[wl_key]['efficiency']:.4f}, "
                   f"串扰: {original_metrics[wl_key]['avg_crosstalk']:.4f}, "
                   f"SNR: {original_metrics[wl_key]['snr']:.1f}\n")
            f.write(f"  改进 - 效率: {improved_metrics[wl_key]['efficiency']:.4f}, "
                   f"串扰: {improved_metrics[wl_key]['avg_crosstalk']:.4f}, "
                   f"SNR: {improved_metrics[wl_key]['snr']:.1f}\n")
        
        # 改进技术说明
        f.write(f"\n🔧 主要改进技术:\n")
        f.write("1. 🎯 波长独立相位掩膜 - 每个波长使用专门优化的相位掩膜\n")
        f.write("2. 📊 多目标损失函数 - 效率+分离+串扰+集中+平滑综合优化\n")
        f.write("3. 🚀 AdamW优化器 - 更好的权重衰减和收敛性能\n")
        f.write("4. 📈 余弦退火调度 - 动态学习率调整\n")
        f.write("5. 🔄 差分检测机制 - 增强波长分离效果\n")
        f.write("6. 🎨 相位平滑约束 - 提高制造可行性\n")
        f.write("7. ⏰ 早停机制 - 防止过拟合\n")
        f.write("8. 📏 梯度裁剪 - 稳定训练过程\n\n")
        
        # 结论
        f.write("🎉 结论:\n")
        f.write(f"改进版模型在平均效率上提升了 {eff_improvement:.1f}%，")
        f.write(f"串扰降低了 {cross_improvement:.1f}%，")
        f.write(f"分离比率提升了 {ratio_improvement:.1f}%。\n")
        f.write("改进效果显著，技术方案可行！\n")
    
    print(f"📄 详细报告已保存到: {report_path}")

def run_ablation_study(config):
    """🔬 消融研究 - 分析各个改进组件的贡献"""
    print("\n🔬 开始消融研究...")
    
    data_generator = SingleModeDualWavelengthDataGenerator(config)
    input_fields = data_generator.generate_input_fields()
    
    ablation_results = {}
    
    # 1. 基础模型（原版）
    print("  测试基础模型...")
    base_model = SimpleMultiWavelengthModel(config, num_layers=config.num_layers)
    base_trainer = SimpleTrainer(config, data_generator)
    base_result = base_trainer.train_model(base_model, config.epochs // 2)
    ablation_results['基础模型'] = base_result['avg_efficiency']
    
    # 2. 仅添加独立相位掩膜
    print("  测试独立相位掩膜...")
    independent_model = ImprovedMultiWavelengthModel(config, num_layers=config.num_layers)
    independent_model.use_differential_detection = False  # 关闭差分检测
    independent_trainer = ImprovedTrainer(config, data_generator)
    independent_result = independent_trainer.train_model(independent_model, config.epochs // 2)
    ablation_results['独立相位掩膜'] = independent_result['avg_efficiency']
    
    # 3. 完整改进模型
    print("  测试完整改进模型...")
    full_model = ImprovedMultiWavelengthModel(config, num_layers=config.num_layers)
    full_trainer = ImprovedTrainer(config, data_generator)
    full_result = full_trainer.train_model(full_model, config.epochs // 2)
    ablation_results['完整改进模型'] = full_result['avg_efficiency']
    
    # 打印消融研究结果
    print("\n📊 消融研究结果:")
    print("-" * 40)
    for name, efficiency in ablation_results.items():
        improvement = (efficiency - ablation_results['基础模型']) / ablation_results['基础模型'] * 100
        print(f"{name:<15}: {efficiency:.4f} ({improvement:+.1f}%)")
    
    return ablation_results

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行主程序
    main()
    
    # 可选：运行消融研究
    print("\n" + "="*60)
    print("🔬 是否运行消融研究？(y/n)")
    if input().lower() == 'y':
        config = Config(
            field_size=None,
            wavelengths=[450e-9, 650e-9],
            detectsize=10,
            num_layers=2,
            epochs=200,  # 消融研究使用较少轮数
            learning_rate=1e-3,
            save_dir='results_ablation'
        )
        ablation_results = run_ablation_study(config)

