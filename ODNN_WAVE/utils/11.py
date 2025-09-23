import numpy as np
import matplotlib.pyplot as plt
import os

# 创建输出目录
output_dir = "first_three_modes"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
eigenmodes = np.load('ODNN_WAVE/eigenmodes_OM4.npy')
print(f"数据形状: {eigenmodes.shape}")

# 处理数据维度
if eigenmodes.ndim == 3:
    # 如果是 (height, width, modes) 格式
    if eigenmodes.shape[2] >= 3:
        modes_data = eigenmodes[:, :, 0:3].transpose(2, 0, 1)
    else:
        print(f"警告：只有 {eigenmodes.shape[2]} 个模式，将保存所有可用模式")
        modes_data = eigenmodes.transpose(2, 0, 1)
elif eigenmodes.ndim == 4:
    # 如果是 4D 数据，取前3个
    modes_data = eigenmodes[0:3]
else:
    # 如果是 2D，当作单个模式处理
    modes_data = eigenmodes[np.newaxis, :, :]

print(f"处理后形状: {modes_data.shape}")

# 保存前三个模式
num_modes = min(3, modes_data.shape[0])
for i in range(num_modes):
    mode_data = modes_data[i]
    
    # 计算强度
    if np.iscomplexobj(mode_data):
        intensity = np.abs(mode_data)**2
    else:
        intensity = np.abs(mode_data)**2
    
    # 翻转Y轴
    intensity_flipped = np.flipud(intensity)
    
    # 创建图像
    plt.figure(figsize=(8, 6))
    plt.imshow(intensity_flipped, cmap='hot', origin='lower')
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f"mode_{i}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()
    
    print(f"✅ 保存模式 {i}: {output_path}")

print(f"🎉 完成！共保存了 {num_modes} 个模式到: {output_dir}")
