import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def convert_npy_to_png(npy_file_path, output_dir=None):
    """
    将单个 .npy 文件转换为 .png 图像 - 只显示强度分布
    
    参数:
        npy_file_path: .npy 文件路径
        output_dir: 输出目录（默认与原文件同目录）
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(npy_file_path)
    
    # 加载数据
    try:
        data = np.load(npy_file_path, allow_pickle=True)
        print(f"✅ 加载文件: {os.path.basename(npy_file_path)}")
        print(f"   数据形状: {data.shape}")
        print(f"   数据类型: {data.dtype}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None
    
    # 处理数据维度
    if data.ndim > 2:
        print(f"   降维处理: {data.shape} -> ", end="")
        # 如果是多维数据，取最后两个维度作为空间维度
        data = data.squeeze()
        if data.ndim > 2:
            data = data.reshape(-1, data.shape[-2], data.shape[-1])
            data = np.sum(data, axis=0)  # 对其他维度求和
        print(f"{data.shape}")
    
    # 计算强度
    if np.iscomplexobj(data):
        intensity = np.abs(data)**2
    else:
        intensity = np.abs(data)**2
    
    # 翻转Y轴（匹配原代码的显示方式）
    intensity_flipped = np.flipud(intensity)
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(npy_file_path))[0]
    
    # 创建单个强度图
    plt.figure(figsize=(8, 6))  # 调整为单图尺寸
    
    # 绘制强度分布（不添加色标）
    plt.imshow(intensity_flipped, cmap='hot', origin='lower')
    
    # 移除坐标轴标签和刻度
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()
    
    print(f"✅ 保存图像: {output_path}")
    return output_path


def batch_convert_npy_to_png(input_dir, output_dir=None, pattern="*.npy"):
    """
    批量转换目录中的所有 .npy 文件 - 只生成强度图
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录（默认为输入目录下的 'png_results'）
        pattern: 文件匹配模式
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "png_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有 .npy 文件
    npy_files = glob.glob(os.path.join(input_dir, pattern))
    
    if not npy_files:
        print(f"❌ 在 {input_dir} 中未找到匹配的 .npy 文件")
        return
    
    print(f"🔍 找到 {len(npy_files)} 个 .npy 文件")
    
    successful_conversions = 0
    
    for npy_file in npy_files:
        try:
            result = convert_npy_to_png(npy_file, output_dir)
            if result:
                successful_conversions += 1
        except Exception as e:
            print(f"❌ 转换失败 {os.path.basename(npy_file)}: {e}")
    
    print(f"\n✅ 批量转换完成！")
    print(f"   成功转换: {successful_conversions}/{len(npy_files)} 个文件")
    print(f"   输出目录: {output_dir}")

# 使用示例
batch_convert_npy_to_png("ODNN_WAVE/results/1_wl_basewl_8.5e-07_z_prop_8e-05_focus_10")
