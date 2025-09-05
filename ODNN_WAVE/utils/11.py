import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 设置更专业的绘图参数
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# 创建复杂的多子图布局
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], width_ratios=[1, 1], 
                      hspace=0.3, wspace=0.3)

# =================== 主图：3D色散传播可视化 ===================
ax_main = fig.add_subplot(gs[0, :], projection='3d')

# 定义精确的波长和物理参数
wavelengths = np.array([400, 450, 500, 550, 600, 650, 700, 750, 800])  # nm
n_base = 1.45  # 基础折射率
dn_dlambda = -0.01e-3  # 色散系数 dn/dλ

# 计算每个波长的折射率（Cauchy方程）
def cauchy_dispersion(wavelength):
    A = 1.4580
    B = 0.00354  # μm²
    return A + B / (wavelength/1000)**2

# 生成颜色映射
def wavelength_to_rgb(wavelength):
    """将波长转换为RGB颜色"""
    if wavelength < 380:
        return (0.5, 0, 0.5)
    elif wavelength < 440:
        return ((440 - wavelength) / 60, 0, 1)
    elif wavelength < 490:
        return (0, (wavelength - 440) / 50, 1)
    elif wavelength < 510:
        return (0, 1, (510 - wavelength) / 20)
    elif wavelength < 580:
        return ((wavelength - 510) / 70, 1, 0)
    elif wavelength < 645:
        return (1, (645 - wavelength) / 65, 0)
    elif wavelength < 780:
        return (1, 0, 0)
    else:
        return (0.5, 0, 0)

# 3D传播路径
z_prop = np.linspace(0, 50, 500)  # 传播距离 (mm)
fiber_radius = 5  # 光纤半径 (μm)

for i, wl in enumerate(wavelengths):
    n = cauchy_dispersion(wl)
    color = wavelength_to_rgb(wl)
    
    # 计算群速度和相位速度
    beta = 2 * np.pi * n / (wl * 1e-6)  # 传播常数
    
    # 模拟模式色散和材料色散
    delta_beta = (n - cauchy_dispersion(600)) * 2 * np.pi / (wl * 1e-6)
    
    # 3D螺旋路径模拟模式传播
    theta = delta_beta * z_prop / 10
    x_path = fiber_radius * 0.3 * np.cos(theta) * np.exp(-z_prop/100)
    y_path = fiber_radius * 0.3 * np.sin(theta) * np.exp(-z_prop/100)
    
    # 添加色散延迟效应
    z_delayed = z_prop + delta_beta * z_prop**2 / 1000
    
    ax_main.plot(x_path, y_path, z_delayed, color=color, linewidth=3, 
                label=f'{wl:.0f}nm', alpha=0.8)
    
    # 添加起始点
    ax_main.scatter([x_path[0]], [y_path[0]], [z_delayed[0]], 
                   color=color, s=50, alpha=1.0)

# 绘制光纤截面
theta_fiber = np.linspace(0, 2*np.pi, 100)
x_fiber = fiber_radius * np.cos(theta_fiber)
y_fiber = fiber_radius * np.sin(theta_fiber)
z_fiber = np.zeros_like(x_fiber)
ax_main.plot(x_fiber, y_fiber, z_fiber, 'k-', linewidth=2, alpha=0.5)

# 添加光纤芯层和包层
core_radius = fiber_radius * 0.6
x_core = core_radius * np.cos(theta_fiber)
y_core = core_radius * np.sin(theta_fiber)
ax_main.plot(x_core, y_core, z_fiber, 'b--', linewidth=1, alpha=0.7, label='Core')

ax_main.set_xlabel('X (μm)', fontsize=12)
ax_main.set_ylabel('Y (μm)', fontsize=12)
ax_main.set_zlabel('Propagation Distance (mm)', fontsize=12)
ax_main.set_title('3D Wavelength Dispersion in Optical Fiber', fontsize=14, fontweight='bold')
ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# =================== 左下：色散曲线详细分析 ===================
ax_disp = fig.add_subplot(gs[1, 0])

# 扩展波长范围用于色散分析
lambda_extended = np.linspace(300, 1700, 2000)
n_extended = cauchy_dispersion(lambda_extended)

# 计算群折射率和色散参数
dn_dlambda_extended = np.gradient(n_extended, lambda_extended)
d2n_dlambda2_extended = np.gradient(dn_dlambda_extended, lambda_extended)

# 群速度色散参数
c = 299792458  # 光速 m/s
D = -(lambda_extended * 1e-9) / c * d2n_dlambda2_extended * 1e12  # ps/(nm·km)

# 绘制色散曲线
ax_disp.plot(lambda_extended, D, 'b-', linewidth=2.5, label='Material Dispersion')

# 添加零色散波长
zero_disp_idx = np.argmin(np.abs(D))
zero_disp_wavelength = lambda_extended[zero_disp_idx]
ax_disp.axvline(zero_disp_wavelength, color='red', linestyle='--', linewidth=2, 
                label=f'Zero Dispersion: {zero_disp_wavelength:.0f}nm')

# 标注可见光区域
visible_range = (380, 780)
ax_disp.axvspan(visible_range[0], visible_range[1], alpha=0.2, color='yellow', 
                label='Visible Range')

# 标注通信窗口
comm_windows = [(1260, 1360, 'O-band'), (1530, 1565, 'C-band'), (1565, 1625, 'L-band')]
colors_comm = ['lightgreen', 'lightcoral', 'lightblue']
for (start, end, name), color in zip(comm_windows, colors_comm):
    ax_disp.axvspan(start, end, alpha=0.3, color=color, label=name)

# 在可见光波长处标注点
for wl in wavelengths:
    if 380 <= wl <= 780:
        idx = np.argmin(np.abs(lambda_extended - wl))
        color = wavelength_to_rgb(wl)
        ax_disp.plot(wl, D[idx], 'o', color=color, markersize=8, 
                    markeredgecolor='black', markeredgewidth=1)

ax_disp.set_xlabel('Wavelength (nm)', fontsize=12)
ax_disp.set_ylabel('Dispersion D (ps/nm/km)', fontsize=12)
ax_disp.set_title('Material Dispersion Curve', fontsize=12, fontweight='bold')
ax_disp.grid(True, alpha=0.3)
ax_disp.legend(fontsize=8, loc='upper right')
ax_disp.set_xlim(300, 1700)

# =================== 右下：相位和群延迟 ===================
ax_delay = fig.add_subplot(gs[1, 1])

# 计算相位延迟和群延迟
L = 1000  # 光纤长度 1km
phase_delay = 2 * np.pi * n_extended * L / (lambda_extended * 1e-6)  # 相位延迟
group_delay = L * (n_extended - lambda_extended * dn_dlambda_extended) / c * 1e9  # 群延迟 (ns)

# 归一化到参考波长
ref_idx = np.argmin(np.abs(lambda_extended - 1550))
phase_delay_norm = phase_delay - phase_delay[ref_idx]
group_delay_norm = group_delay - group_delay[ref_idx]

# 双y轴绘图
ax_delay2 = ax_delay.twinx()

line1 = ax_delay.plot(lambda_extended, phase_delay_norm/1e6, 'b-', linewidth=2, 
                     label='Phase Delay (×10⁶ rad)')
line2 = ax_delay2.plot(lambda_extended, group_delay_norm, 'r-', linewidth=2, 
                      label='Group Delay (ns)')

# 标注可见光波长
for wl in wavelengths:
    if 380 <= wl <= 780:
        idx = np.argmin(np.abs(lambda_extended - wl))
        color = wavelength_to_rgb(wl)
        ax_delay.plot(wl, phase_delay_norm[idx]/1e6, 's', color=color, markersize=6,
                     markeredgecolor='black', markeredgewidth=1)
        ax_delay2.plot(wl, group_delay_norm[idx], '^', color=color, markersize=6,
                      markeredgecolor='black', markeredgewidth=1)

ax_delay.set_xlabel('Wavelength (nm)', fontsize=12)
ax_delay.set_ylabel('Phase Delay (×10⁶ rad)', color='blue', fontsize=12)
ax_delay2.set_ylabel('Group Delay (ns)', color='red', fontsize=12)
ax_delay.tick_params(axis='y', labelcolor='blue')
ax_delay2.tick_params(axis='y', labelcolor='red')

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax_delay.legend(lines, labels, loc='upper left', fontsize=10)

ax_delay.set_title('Phase and Group Delay vs Wavelength', fontsize=12, fontweight='bold')
ax_delay.grid(True, alpha=0.3)
ax_delay.set_xlim(300, 1700)

# =================== 底部：脉冲展宽效应 ===================
ax_pulse = fig.add_subplot(gs[2, :])

# 模拟脉冲展宽
t = np.linspace(-50, 50, 1000)  # 时间 ps
sigma_0 = 10  # 初始脉冲宽度 ps

# 不同传播距离的脉冲形状
distances = [0, 10, 50, 100]  # km
colors_pulse = ['black', 'blue', 'green', 'red']

for dist, color in zip(distances, colors_pulse):
    # 计算脉冲展宽（简化模型）
    D_eff = np.interp(1550, lambda_extended, D)  # 1550nm处的色散
    sigma_z = sigma_0 * np.sqrt(1 + (D_eff * dist * 1e-3)**2)  # 展宽后的脉冲宽度
    
    # 高斯脉冲
    pulse = np.exp(-t**2 / (2 * sigma_z**2))
    ax_pulse.plot(t, pulse, color=color, linewidth=2.5, 
                 label=f'L = {dist} km, σ = {sigma_z:.1f} ps')

# 添加频谱成分
ax_pulse_freq = ax_pulse.twinx()
freq = np.linspace(-0.5, 0.5, 1000)  # 归一化频率
spectrum = np.exp(-2 * (np.pi * freq * sigma_0)**2)
ax_pulse_freq.plot(freq*100, spectrum, 'orange', linestyle='--', linewidth=2, 
                  alpha=0.7, label='Spectrum')

ax_pulse.set_xlabel('Time (ps)', fontsize=12)
ax_pulse.set_ylabel('Pulse Amplitude', fontsize=12)
ax_pulse_freq.set_ylabel('Spectral Amplitude', color='orange', fontsize=12)
ax_pulse_freq.tick_params(axis='y', labelcolor='orange')

ax_pulse.set_title('Pulse Broadening Due to Chromatic Dispersion', 
                  fontsize=14, fontweight='bold')
ax_pulse.legend(loc='upper left', fontsize=10)
ax_pulse_freq.legend(loc='upper right', fontsize=10)
ax_pulse.grid(True, alpha=0.3)

# 添加数学公式
formula_text = r'$\sigma_z(L) = \sigma_0\sqrt{1 + \left(\frac{D \cdot L \cdot \Delta\lambda}{2\pi c \sigma_0}\right)^2}$'
ax_pulse.text(0.02, 0.95, formula_text, transform=ax_pulse.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

# 保存高质量图形
plt.savefig('advanced_wavelength_dispersion.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig('advanced_wavelength_dispersion.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('advanced_wavelength_dispersion.png', format='png', dpi=300, bbox_inches='tight')

print("✅ 高级波长色散效应图已保存:")
print("  - advanced_wavelength_dispersion.svg")
print("  - advanced_wavelength_dispersion.pdf")
print("  - advanced_wavelength_dispersion.png")

plt.tight_layout()
plt.show()
