import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# 设置中文字体和图形参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 加载数据
eigenmodes = np.load('ODNN_WAVE/eigenmodes_OM4.npy')
print(f"数据形状: {eigenmodes.shape}")

# 选择前三个模式 (索引 0, 1, 2)
mode_indices = [0, 1, 2]
mode_labels = ['LP₀₁', 'LP₁₁ᵃ', 'LP₁₁ᵇ']  # 根据典型OM4光纤模式顺序

# 创建图形
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 设置颜色映射
# 振幅用热力图
amp_cmap = 'hot'
# 相位用自定义颜色映射（蓝-青-黄）
phase_colors = ['#000080', '#008080', '#FFFF00']
phase_cmap = LinearSegmentedColormap.from_list('phase', phase_colors, N=256)

for i, mode_idx in enumerate(mode_indices):
    # 获取复数场分布
    mode_field = eigenmodes[:, :, mode_idx]
    
    # 计算振幅和相位
    amplitude = np.abs(mode_field)
    phase = np.angle(mode_field)
    
    # 归一化振幅
    amplitude_norm = amplitude / np.max(amplitude) if np.max(amplitude) > 0 else amplitude
    
    # 上排：振幅分布
    ax_amp = axes[0, i]
    im_amp = ax_amp.imshow(amplitude_norm, cmap=amp_cmap, 
                          extent=[-25, 25, -25, 25], origin='lower')
    
    # 添加纤芯边界圆圈
    circle_amp = patches.Circle((0, 0), 25, linewidth=2, edgecolor='white', 
                               facecolor='none', linestyle='-', alpha=0.8)
    ax_amp.add_patch(circle_amp)
    
    ax_amp.set_aspect('equal')
    ax_amp.set_xlim(-30, 30)
    ax_amp.set_ylim(-30, 30)
    
    # 移除坐标轴标签和刻度（与参考图一致）
    ax_amp.set_xticks([])
    ax_amp.set_yticks([])
    
    # 添加颜色条
    cbar_amp = plt.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)
    cbar_amp.ax.tick_params(labelsize=10)
    # 设置科学计数法格式
    cbar_amp.formatter.set_powerlimits((0, 0))
    cbar_amp.update_ticks()
    
    # 下排：相位分布
    ax_phase = axes[1, i]
    im_phase = ax_phase.imshow(phase, cmap=phase_cmap, 
                              extent=[-25, 25, -25, 25], origin='lower',
                              vmin=-np.pi, vmax=np.pi)
    
    # 添加纤芯边界圆圈
    circle_phase = patches.Circle((0, 0), 25, linewidth=2, edgecolor='black', 
                                 facecolor='none', linestyle='-', alpha=0.8)
    ax_phase.add_patch(circle_phase)
    
    ax_phase.set_aspect('equal')
    ax_phase.set_xlim(-30, 30)
    ax_phase.set_ylim(-30, 30)
    
    # 移除坐标轴标签和刻度
    ax_phase.set_xticks([])
    ax_phase.set_yticks([])
    
    # 添加相位颜色条
    cbar_phase = plt.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)
    cbar_phase.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar_phase.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    cbar_phase.ax.tick_params(labelsize=10)

# 添加列标签 (a), (b), (c)
for i, label in enumerate(['(a)', '(b)', '(c)']):
    axes[1, i].text(0.5, -0.15, label, transform=axes[1, i].transAxes,
                   ha='center', va='top', fontsize=14, fontweight='bold')

# 添加行标签
fig.text(0.02, 0.75, 'Amplitude', rotation=90, va='center', ha='center', 
         fontsize=14, fontweight='bold')
fig.text(0.02, 0.25, 'Phase', rotation=90, va='center', ha='center', 
         fontsize=14, fontweight='bold')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(left=0.08, bottom=0.1, top=0.95, wspace=0.3, hspace=0.1)

# 保存为多种格式
print("正在保存图像文件...")

# 保存为SVG格式（矢量图）
plt.savefig('LP_modes_012_amplitude_phase.svg', format='svg', dpi=300, bbox_inches='tight')
print("✅ 已保存: LP_modes_012_amplitude_phase.svg")

# 保存为PDF格式（矢量图）
plt.savefig('LP_modes_012_amplitude_phase.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("✅ 已保存: LP_modes_012_amplitude_phase.pdf")

# 保存为PNG格式（位图）
plt.savefig('LP_modes_012_amplitude_phase.png', format='png', dpi=300, bbox_inches='tight')
print("✅ 已保存: LP_modes_012_amplitude_phase.png")

# 保存为EPS格式（矢量图，适合LaTeX）
plt.savefig('LP_modes_012_amplitude_phase.eps', format='eps', dpi=300, bbox_inches='tight')
print("✅ 已保存: LP_modes_012_amplitude_phase.eps")

plt.show()

# 打印模式信息
print("\n=== 前三个模式信息 ===")
for i, mode_idx in enumerate(mode_indices):
    mode_field = eigenmodes[:, :, mode_idx]
    amplitude = np.abs(mode_field)
    phase = np.angle(mode_field)
    max_amp = np.max(amplitude)
    print(f"模式 {mode_idx} ({mode_labels[i]}): 最大振幅 = {max_amp:.6e}")
    print(f"  - 振幅范围: [{np.min(amplitude):.6e}, {max_amp:.6e}]")
    print(f"  - 相位范围: [{np.min(phase):.3f}, {np.max(phase):.3f}] rad")
    print()

print("\n=== 保存的文件格式说明 ===")
print("📄 SVG: 可缩放矢量图形，适合网页和演示")
print("📄 PDF: 高质量矢量图，适合论文和印刷")
print("📄 PNG: 高分辨率位图，适合一般用途")
print("📄 EPS: 封装PostScript，适合LaTeX文档")
