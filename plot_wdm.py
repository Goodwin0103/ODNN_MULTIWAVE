import matplotlib.pyplot as plt
import numpy as np

# 设置学术风格
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(10, 6))

# 数据
years = np.array([1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
cband_channels = np.array([4, 8, 32, 40, 80, 96, 110, 110])
lband_channels = np.array([0, 0, 0, 0, 8, 20, 40, 96])

# 绘制
ax.plot(years, cband_channels, 'b-o', linewidth=2.5, markersize=7, 
        label='C-band Channels (1530-1565 nm)', markerfacecolor='lightblue')
ax.plot(years, lband_channels, 'r-s', linewidth=2.5, markersize=7, 
        label='L-band Channels (1565-1625 nm)', markerfacecolor='lightcoral')
ax.plot(years, cband_channels + lband_channels, 'g--^', linewidth=2, 
        label='Total WDM Channels', alpha=0.8)

# 饱和区域
ax.axvspan(2020, 2025, alpha=0.1, color='red', label='Saturation Region')
ax.axhline(y=110, color='blue', linestyle=':', alpha=0.7, linewidth=1.5)
ax.axhline(y=96, color='red', linestyle=':', alpha=0.7, linewidth=1.5)

# 标注
ax.text(2018, 115, 'C-band Saturation (~110 channels)', fontsize=10, color='blue')
ax.text(2018, 101, 'L-band Capacity (~96 channels)', fontsize=10, color='red')

# 格式设置
ax.set_xlabel('Year', fontsize=12, weight='bold')
ax.set_ylabel('Number of WDM Channels', fontsize=12, weight='bold')
ax.set_title('WDM Technology Development Approaching Spectral Saturation', 
             fontsize=14, weight='bold')
ax.legend(fontsize=10)
ax.set_xlim(1988, 2027)
ax.set_ylim(0, 220)

plt.tight_layout()
plt.savefig('figure_1_wdm_saturation.pdf', bbox_inches='tight', dpi=300)
plt.show()
