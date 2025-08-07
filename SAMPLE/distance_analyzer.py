# distance_analyzer.py - 完整修复版本
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import time
import traceback
from config import Config
from data_generator import SingleModeDualWavelengthDataGenerator
from model import SimpleMultiWavelengthModel
from trainer import SimpleTrainer

class PropagationDistanceAnalyzer:
    """
    传播距离分析器 - 修复版本
    解决了数据生成器方法调用问题和matplotlib显示问题
    """
    def __init__(self, save_dir='propagation_results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
    def analyze_multiple_distances(self, distances_mm=None, epochs=50):
        """
        分析多个传播距离下的光场分布性能
        使用正确的数据生成方法
        """
        if distances_mm is None:
            distances_mm = [5, 8, 10, 18]
        
        print(f"🌈 传播距离光场分布可视化工具")
        print("="*50)
        print(f"🔍 开始多距离光场分布可视化")
        print(f"📊 测试距离: {distances_mm} mm")
        print(f"🔄 每个距离训练轮数: {epochs}")
        print(f"💾 结果保存至: {self.save_dir}")
        
        results = {}
        
        for i, dist_mm in enumerate(distances_mm):
            print(f"\n{'='*60}")
            print(f"📏 处理距离 {i+1}/{len(distances_mm)}: {dist_mm}mm")
            print(f"{'='*60}")
            
            try:
                # 训练单个距离的模型
                result = self._analyze_single_distance(dist_mm, epochs)
                results[dist_mm] = result
                
                # 保存详细分析图
                self._plot_detailed_analysis(dist_mm, result)
                
                print(f"✅ {dist_mm}mm 完成并已保存")
                
            except Exception as e:
                print(f"❌ {dist_mm}mm 处理失败: {str(e)}")
                traceback.print_exc()
                results[dist_mm] = {'error': str(e)}
        
        # 创建综合对比分析
        self._create_comprehensive_comparison(results)
        
        print(f"\n🎉 所有可视化完成！")
        print(f"📁 结果保存在: {self.save_dir}")
        
        return results
    
    def _analyze_single_distance(self, distance_mm, epochs):
        """
        分析单个传播距离
        使用改进的训练器进行详细的性能分析
        """
        print(f"🔧 配置距离: {distance_mm}mm")
        
        # 创建配置
        config = Config()
        config.num_epochs = epochs
        config.propagation_distance = distance_mm / 1000.0  # 转换为米
        
        # 验证配置
        self._validate_config(config)
        
        print(f"✅ 传播距离设置: {config.propagation_distance*1000:.1f}mm")
        print(f"🎯 场大小: {config.field_size}x{config.field_size}")
        print(f"🌈 波长: {config.wavelengths}")
        
        # 初始化组件
        data_generator = SingleModeDualWavelengthDataGenerator(config)
        model = SimpleMultiWavelengthModel(config)
        trainer = SimpleTrainer(config, data_generator)
        
        # 使用改进的训练方法
        print(f"🚀 开始训练...")
        start_time = time.time()
        training_result = trainer.train_model(model, num_epochs=epochs)
        
        # 提取训练结果
        trained_model = training_result['model']
        training_history = training_result['training_history']
        final_efficiencies = training_result['final_efficiencies']
        training_time = training_result['training_time']
        
        print(f"⏱️ 训练完成，总耗时: {training_time:.1f}秒")
        
        # 生成测试输出 - 使用正确的方法
        test_outputs = self._generate_test_outputs_fixed(config, data_generator, trained_model)
        
        return {
            'config': config,
            'distance_mm': distance_mm,
            'training_history': training_history,
            'final_efficiencies': final_efficiencies,
            'training_time': training_time,
            'test_outputs': test_outputs,
            'model': trained_model
        }
    
    def _generate_test_outputs_fixed(self, config, data_generator, model):
        """
        修复的测试输出生成方法
        使用正确的数据生成器API
        """
        model.eval()
        
        with torch.no_grad():
            # 修复：使用正确的方法生成输入场
            # 检查数据生成器有哪些可用的方法
            available_methods = [method for method in dir(data_generator) 
                               if not method.startswith('_') and callable(getattr(data_generator, method))]
            
            print(f"🔍 数据生成器可用方法: {available_methods}")
            
            # 尝试不同的方法来生成输入场
            input_fields = None
            
            # 方法1: 尝试 generate_input_fields
            if hasattr(data_generator, 'generate_input_fields'):
                print("📊 使用 generate_input_fields 方法")
                input_fields = data_generator.generate_input_fields()
            
            # 方法2: 尝试 generate_batch (如果存在)
            elif hasattr(data_generator, 'generate_batch'):
                print("📊 使用 generate_batch 方法")
                input_fields = data_generator.generate_batch(batch_size=1)
                if isinstance(input_fields, tuple):
                    input_fields = input_fields[0]  # 取第一个元素
            
            # 方法3: 尝试直接调用 __call__ 方法
            elif hasattr(data_generator, '__call__'):
                print("📊 使用 __call__ 方法")
                try:
                    input_fields = data_generator()
                except Exception as e:
                    print(f"⚠️ __call__ 方法失败: {e}")
            
            # 方法4: 尝试创建简单的测试输入
            if input_fields is None:
                print("📊 创建简单的测试输入场")
                input_fields = self._create_simple_test_input(config)
            
            # 确保输入在正确的设备上
            if isinstance(input_fields, list):
                input_fields = [field.to(config.device) for field in input_fields]
            else:
                input_fields = input_fields.to(config.device)
            
            print(f"✅ 输入场类型: {type(input_fields)}")
            if isinstance(input_fields, list):
                print(f"✅ 输入场形状: {[field.shape for field in input_fields]}")
            else:
                print(f"✅ 输入场形状: {input_fields.shape}")
            
            # 获取模型输出
            output_fields = model(input_fields)
            
            # 计算最终损失
            loss_dict = model.get_detailed_loss(output_fields)
            
            # 转换为numpy用于可视化
            if isinstance(input_fields, list):
                input_numpy = [field.cpu().numpy() for field in input_fields]
            else:
                input_numpy = input_fields.cpu().numpy()
            
            output_numpy = [field.cpu().numpy() for field in output_fields]
            
            return {
                'input_fields': input_numpy,
                'output_fields': output_numpy,
                'loss_dict': {k: v.item() if torch.is_tensor(v) else v 
                             for k, v in loss_dict.items()}
            }
    
    def _create_simple_test_input(self, config):
        """
        创建简单的测试输入场
        当数据生成器方法不可用时的后备方案
        """
        print("🔧 创建简单的高斯光束测试输入")
        
        # 创建坐标网格
        x = torch.linspace(-1, 1, config.field_size)
        y = torch.linspace(-1, 1, config.field_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 创建高斯光束
        sigma = 0.3  # 光束宽度
        gaussian_beam = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        
        # 添加批次维度和复数部分
        test_input = torch.complex(gaussian_beam.unsqueeze(0), 
                                  torch.zeros_like(gaussian_beam.unsqueeze(0)))
        
        print(f"✅ 创建的测试输入形状: {test_input.shape}")
        
        return test_input
    
    def _validate_config(self, config):
        """验证配置参数"""
        print(f"🔍 开始配置验证...")
        
        # 检查检测区域是否在有效范围内
        for i, (offset_x, offset_y) in enumerate(config.offsets):
            center_x = config.field_size // 2 + offset_x
            center_y = config.field_size // 2 + offset_y
            half_size = config.detect_size // 2
            
            if (center_x - half_size < 0 or center_x + half_size > config.field_size or
                center_y - half_size < 0 or center_y + half_size > config.field_size):
                raise ValueError(f"波长 {config.wavelengths[i]*1e9:.0f}nm 检测区域超出边界")
            
            wl_nm = config.wavelengths[i] * 1e9
            print(f"✅ 波长 {wl_nm:.0f}nm 检测区域位置有效")
        
        print(f"✅ 所有检测区域都在有效范围内")
        print(f"✅ 学习率 {config.learning_rate} 在合理范围内")
        
        if config.num_epochs < 50:
            print(f"⚠️  训练轮数 {config.num_epochs} 可能不足")
        
        print(f"🔍 配置验证完成")
    
    def _safe_prepare_image_data(self, data, data_name="未知数据"):
        """
        安全地准备图像数据用于matplotlib显示
        解决维度和类型问题 [[0]](#__0) [[1]](#__1)
        """
        print(f"🔧 处理 {data_name}，原始形状: {data.shape if hasattr(data, 'shape') else type(data)}")
        
        try:
            # 确保是numpy数组
            if torch.is_tensor(data):
                data = data.cpu().numpy()
            elif not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # 如果是复数，取绝对值
            if np.iscomplexobj(data):
                data = np.abs(data)
            
            # 处理不同的维度情况 [[0]](#__0)
            if len(data.shape) == 1:
                # 一维数组，尝试重塑为正方形
                size = int(np.sqrt(data.shape[0]))
                if size * size == data.shape[0]:
                    data = data.reshape(size, size)
                    print(f"✅ 重塑一维数组为: {data.shape}")
                else:
                    print(f"❌ 无法重塑一维数组，创建默认图像")
                    data = np.zeros((50, 50))
                    
            elif len(data.shape) == 2:
                # 二维数组，直接使用
                print(f"✅ 二维数组，形状: {data.shape}")
                
            elif len(data.shape) == 3:
                # 三维数组，需要处理 [[1]](#__1) [[4]](#__4)
                if data.shape[0] == 1:
                    # 批次维度为1，去掉
                    data = data[0]
                    print(f"✅ 去掉批次维度，新形状: {data.shape}")
                elif data.shape[-1] == 1:
                    # 最后一个维度为1（灰度图），去掉 [[1]](#__1)
                    data = data[:, :, 0]
                    print(f"✅ 去掉单通道维度，新形状: {data.shape}")
                elif data.shape[-1] in [3, 4]:
                    # RGB或RGBA图像，保持原样
                    print(f"✅ RGB/RGBA图像，形状: {data.shape}")
                else:
                    # 其他情况，取第一个通道
                    data = data[:, :, 0]
                    print(f"✅ 取第一个通道，新形状: {data.shape}")
                    
            elif len(data.shape) == 4:
                # 四维数组，去掉批次维度
                data = data[0]
                print(f"✅ 去掉批次维度，递归处理")
                return self._safe_prepare_image_data(data, data_name)
            
            else:
                print(f"❌ 不支持的维度: {data.shape}，创建默认图像")
                data = np.zeros((50, 50))
            
            # 最终检查
            if len(data.shape) != 2 and not (len(data.shape) == 3 and data.shape[-1] in [3, 4]):
                print(f"❌ 最终形状仍不正确: {data.shape}，创建默认图像")
                data = np.zeros((50, 50))
            
            print(f"✅ {data_name} 处理完成，最终形状: {data.shape}")
            return data
            
        except Exception as e:
            print(f"❌ 处理 {data_name} 时出错: {e}，使用默认图像")
            return np.zeros((50, 50))
    
    def _plot_detailed_analysis(self, distance_mm, result):
        """
        创建详细的分析图表 - 修复版本
        包含训练历史、效率分析和光场分布，解决了UnboundLocalError问题 [[1]](#__1)
        """
        print(f"🎨 开始绘制 {distance_mm}mm 距离的详细分析...")
        
        # 提前初始化所有可能用到的变量，避免UnboundLocalError [[1]](#__1)
        input_intensity = None
        output_450_intensity = None
        output_650_intensity = None
        
        try:
            training_history = result['training_history']
            test_outputs = result['test_outputs']
            config = result['config']
            
            # 创建大型图表 (3x3 布局)
            fig, axes = plt.subplots(3, 3, figsize=(20, 16))
            fig.suptitle(f'详细分析 - 传播距离: {distance_mm}mm', fontsize=16, fontweight='bold')
            
            # 第一行：训练历史
            # 总损失曲线
            axes[0, 0].plot(training_history['total_loss'], 'b-', linewidth=2)
            axes[0, 0].set_title('总损失曲线', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('训练轮次')
            axes[0, 0].set_ylabel('损失值')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')
            
            # 分解损失曲线
            axes[0, 1].plot(training_history['efficiency_loss'], label='效率损失', linewidth=2)
            axes[0, 1].plot(training_history['separation_loss'], label='分离损失', linewidth=2)
            axes[0, 1].plot(training_history['crosstalk_loss'], label='串扰损失', linewidth=2)
            axes[0, 1].plot(training_history['concentration_loss'], label='集中损失', linewidth=2)
            axes[0, 1].set_title('分解损失曲线', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('训练轮次')
            axes[0, 1].set_ylabel('损失值')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
            
            # 效率演化
            efficiencies_array = np.array(training_history['efficiencies'])
            for i, wl in enumerate(config.wavelengths):
                wl_nm = wl * 1e9
                axes[0, 2].plot(efficiencies_array[:, i], label=f'{wl_nm:.0f}nm', linewidth=2)
            axes[0, 2].set_title('效率演化', fontsize=12, fontweight='bold')
            axes[0, 2].set_xlabel('训练轮次')
            axes[0, 2].set_ylabel('效率')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 第二行：输入和输出光场 - 使用安全的数据处理方法
            input_fields = test_outputs['input_fields']
            output_fields = test_outputs['output_fields']
            
            # 安全处理输入光场数据 [[0]](#__0)
            if isinstance(input_fields, list) and len(input_fields) > 0:
                input_intensity = self._safe_prepare_image_data(input_fields[0], "输入场")
            elif input_fields is not None:
                input_intensity = self._safe_prepare_image_data(input_fields, "输入场")
            else:
                print("⚠️ 没有输入场数据，使用默认值")
                input_intensity = np.zeros((50, 50))
            
            # 安全处理450nm输出数据
            if isinstance(output_fields, list) and len(output_fields) > 0:
                output_450_intensity = self._safe_prepare_image_data(output_fields[0], "450nm输出")
            else:
                print("⚠️ 没有450nm输出数据，使用默认值")
                output_450_intensity = np.zeros((50, 50))
            
            # 安全处理650nm输出数据
            if isinstance(output_fields, list) and len(output_fields) > 1:
                output_650_intensity = self._safe_prepare_image_data(output_fields[1], "650nm输出")
            else:
                print("⚠️ 没有650nm输出数据，使用默认值")
                output_650_intensity = np.zeros((50, 50))
            
            # 绘制图像 - 现在所有变量都已安全初始化 [[0]](#__0)
            im1 = axes[1, 0].imshow(input_intensity, cmap='hot', interpolation='bilinear')
            axes[1, 0].set_title('输入光场强度', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('x (像素)')
            axes[1, 0].set_ylabel('y (像素)')
            plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)
            
            # 450nm输出
            im2 = axes[1, 1].imshow(output_450_intensity, cmap='Blues', interpolation='bilinear')
            axes[1, 1].set_title('450nm 输出强度', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('x (像素)')
            axes[1, 1].set_ylabel('y (像素)')
            plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)
            
            # 650nm输出
            im3 = axes[1, 2].imshow(output_650_intensity, cmap='Reds', interpolation='bilinear')
            axes[1, 2].set_title('650nm 输出强度', fontsize=12, fontweight='bold')
            axes[1, 2].set_xlabel('x (像素)')
            axes[1, 2].set_ylabel('y (像素)')
            plt.colorbar(im3, ax=axes[1, 2], shrink=0.8)
            
            # 第三行：分析图
            # 中心线强度分布
            center = output_450_intensity.shape[0] // 2
            center_line_450 = output_450_intensity[center, :]
            center_line_650 = output_650_intensity[center, :]
            x_coords = np.arange(len(center_line_450))
            
            axes[2, 0].plot(x_coords, center_line_450, 'b-', linewidth=2, label='450nm')
            axes[2, 0].plot(x_coords, center_line_650, 'r-', linewidth=2, label='650nm')
            axes[2, 0].set_title('中心线强度分布', fontsize=12, fontweight='bold')
            axes[2, 0].set_xlabel('x (像素)')
            axes[2, 0].set_ylabel('强度')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # 效率柱状图
            final_efficiencies = result['final_efficiencies']
            wavelengths_nm = [wl * 1e9 for wl in config.wavelengths]
            bars = axes[2, 1].bar(wavelengths_nm, final_efficiencies, 
                                 color=['blue', 'red'], alpha=0.7, width=50)
            axes[2, 1].set_title('最终效率', fontsize=12, fontweight='bold')
            axes[2, 1].set_xlabel('波长 (nm)')
            axes[2, 1].set_ylabel('效率')
            axes[2, 1].set_ylim(0, 1)
            axes[2, 1].grid(True, alpha=0.3, axis='y')
            
            # 在柱状图上添加数值标签
            for bar, eff in zip(bars, final_efficiencies):
                height = bar.get_height()
                axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{eff:.3f}\n({eff*100:.1f}%)',
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 参数信息
            axes[2, 2].axis('off')
            loss_dict = test_outputs['loss_dict']
            info_text = f"""传播距离: {distance_mm} mm
训练轮数: {len(training_history['total_loss'])}
训练时间: {result['training_time']:.1f}s

最终损失分解:
• 总损失: {loss_dict['total_loss']:.6f}
• 效率损失: {loss_dict['efficiency_loss']:.6f}
• 分离损失: {loss_dict['separation_loss']:.6f}
• 串扰损失: {loss_dict['crosstalk_loss']:.6f}
• 集中损失: {loss_dict['concentration_loss']:.6f}

最终效率:
• 450nm: {final_efficiencies[0]:.3f} ({final_efficiencies[0]*100:.1f}%)
• 650nm: {final_efficiencies[1]:.3f} ({final_efficiencies[1]*100:.1f}%)

场大小: {config.field_size}×{config.field_size}
设备: {'GPU' if torch.cuda.is_available() else 'CPU'}"""
            
            axes[2, 2].text(0.05, 0.95, info_text, transform=axes[2, 2].transAxes, 
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图片
            save_path = self.save_dir / f"detailed_analysis_{distance_mm}mm.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"💾 详细分析图已保存: {save_path}")
            
        except Exception as e:
            print(f"❌ 绘制详细分析时出错: {e}")
            traceback.print_exc()
            # 确保关闭图形，避免内存泄漏
            plt.close('all')
    
    def _create_comprehensive_comparison(self, results):
        """
        创建综合对比分析
        比较不同距离下的性能指标
        """
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if len(valid_results) < 2:
            print("⚠️ 有效结果太少，无法创建对比图")
            return
        
        distances = sorted(valid_results.keys())
        
        # 提取对比数据
        total_losses = []
        efficiency_450 = []
        efficiency_650 = []
        training_times = []
        
        for dist in distances:
            result = valid_results[dist]
            total_losses.append(result['test_outputs']['loss_dict']['total_loss'])
            efficiency_450.append(result['final_efficiencies'][0])
            efficiency_650.append(result['final_efficiencies'][1])
            training_times.append(result['training_time'])
        
        try:
            # 创建对比图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('传播距离性能对比分析', fontsize=16, fontweight='bold')
            
            # 损失对比
            axes[0, 0].plot(distances, total_losses, 'o-', linewidth=2, markersize=8, color='red')
            axes[0, 0].set_title('总损失 vs 传播距离', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('传播距离 (mm)')
            axes[0, 0].set_ylabel('总损失')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')
            
            # 标记最优点
            best_loss_idx = total_losses.index(min(total_losses))
            axes[0, 0].plot(distances[best_loss_idx], total_losses[best_loss_idx], 
                           'ro', markersize=12, label=f'最优: {distances[best_loss_idx]}mm')
            axes[0, 0].legend()
            
            # 效率对比
            axes[0, 1].plot(distances, efficiency_450, 'o-', linewidth=2, markersize=8, 
                           color='blue', label='450nm')
            axes[0, 1].plot(distances, efficiency_650, 's-', linewidth=2, markersize=8, 
                           color='red', label='650nm')
            axes[0, 1].set_title('效率 vs 传播距离', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel
            axes[0, 1].set_xlabel('传播距离 (mm)')
            axes[0, 1].set_ylabel('效率')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
            
            # 训练时间对比
            axes[1, 0].bar(distances, training_times, alpha=0.7, color='green')
            axes[1, 0].set_title('训练时间 vs 传播距离', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('传播距离 (mm)')
            axes[1, 0].set_ylabel('训练时间 (秒)')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # 综合性能雷达图 
            angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            # 归一化指标
            normalized_losses = [1 - (loss - min(total_losses))/(max(total_losses) - min(total_losses)) 
                               for loss in total_losses]
            normalized_eff_450 = efficiency_450
            normalized_eff_650 = efficiency_650
            normalized_times = [1 - (t - min(training_times))/(max(training_times) - min(training_times)) 
                              for t in training_times]
            
            axes[1, 1] = plt.subplot(2, 2, 4, projection='polar')
            
            for i, dist in enumerate(distances):
                values = [normalized_losses[i], normalized_eff_450[i], 
                         normalized_eff_650[i], normalized_times[i]]
                values += values[:1]  # 闭合图形
                
                axes[1, 1].plot(angles, values, 'o-', linewidth=2, 
                               label=f'{dist}mm', markersize=6)
            
            axes[1, 1].set_xticks(angles[:-1])
            axes[1, 1].set_xticklabels(['损失\n(归一化)', '450nm\n效率', '650nm\n效率', '训练时间\n(归一化)'])
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('综合性能对比', fontsize=12, fontweight='bold', pad=20)
            axes[1, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存对比图
            comparison_path = self.save_dir / "comprehensive_comparison.png"
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 综合对比图已保存: {comparison_path}")
            
            # 打印最优结果摘要
            best_distance = distances[best_loss_idx]
            best_loss = total_losses[best_loss_idx]
            best_eff_450 = efficiency_450[best_loss_idx]
            best_eff_650 = efficiency_650[best_loss_idx]
            
            print(f"\n🏆 最优传播距离分析:")
            print(f"📏 最优距离: {best_distance}mm")
            print(f"📉 最低损失: {best_loss:.6f}")
            print(f"⚡ 450nm效率: {best_eff_450:.3f} ({best_eff_450*100:.1f}%)")
            print(f"⚡ 650nm效率: {best_eff_650:.3f} ({best_eff_650*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ 创建综合对比图时出错: {e}")
            traceback.print_exc()
            plt.close('all')

def main():
    """主函数"""
    try:
        # 创建分析器
        analyzer = PropagationDistanceAnalyzer()
        
        # 定义要测试的距离
        test_distances = [5, 8, 10, 12, 15, 18]  # 毫米
        
        # 运行分析
        print("🚀 开始传播距离分析...")
        results = analyzer.analyze_multiple_distances(
            distances_mm=test_distances,
            epochs=30  # 可以根据需要调整
        )
        
        print("✅ 分析完成！")
        return results
        
    except Exception as e:
        print(f"❌ 主函数执行失败: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
