# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
# from matplotlib.lines import Line2D

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# # ===== 上半部分：物理光学视角 =====
# ax1.set_xlim(0, 12)
# ax1.set_ylim(0, 4)
# ax1.set_aspect('equal')

# # 输入光场
# input_rect1 = patches.Rectangle((0.5, 1.5), 0.3, 1, linewidth=2, 
#                                edgecolor='blue', facecolor='lightblue', alpha=0.7)
# ax1.add_patch(input_rect1)
# ax1.text(0.65, 1, r'$U_{in}$', fontsize=12, ha='center', va='center')

# # 衍射层（物理视角）
# layer_positions = [2.5, 4.5, 6.5, 8.5]
# layer_colors = ['red', 'green', 'orange', 'purple']
# layer_labels = [r'Layer 1', r'Layer 2', r'Layer 3', r'Layer L']
# phase_labels = [r'$\phi_1(x,y)$', r'$\phi_2(x,y)$', r'$\phi_3(x,y)$', r'$\phi_L(x,y)$']

# for i, (pos, color, label, phase) in enumerate(zip(layer_positions, layer_colors, layer_labels, phase_labels)):
#     # 衍射层
#     layer_rect = patches.Rectangle((pos-0.1, 1), 0.2, 2, linewidth=2,
#                                   edgecolor=color, facecolor=color, alpha=0.3)
#     ax1.add_patch(layer_rect)
    
#     # 相位图案
#     for y in np.linspace(1.2, 2.8, 6):
#         for x in np.linspace(pos-0.08, pos+0.08, 3):
#             intensity = np.sin(4*np.pi*y + i*np.pi/2)
#             color_intensity = plt.cm.viridis(0.5 + 0.5*intensity)
#             small_rect = patches.Rectangle((x-0.02, y-0.1), 0.04, 0.2,
#                                          facecolor=color_intensity, edgecolor='none')
#             ax1.add_patch(small_rect)
    
#     ax1.text(pos, 0.5, label, fontsize=10, ha='center', va='center', weight='bold')
#     ax1.text(pos, 3.5, phase, fontsize=10, ha='center', va='center')

# # 输出光场
# output_rect1 = patches.Rectangle((10.2, 1.5), 0.3, 1, linewidth=2,
#                                 edgecolor='blue', facecolor='lightblue', alpha=0.7)
# ax1.add_patch(output_rect1)
# ax1.text(10.35, 1, r'$U_{out}$', fontsize=12, ha='center', va='center')

# # 传播箭头
# propagation_pairs = [(1, 2.3), (2.7, 4.3), (4.7, 6.3), (6.7, 8.3), (8.7, 10)]
# distance_labels = [r'$d_1$', r'$d_2$', r'$d_3$', r'$d_{L-1}$', '']

# for (start, end), label in zip(propagation_pairs, distance_labels):
#     ax1.annotate('', xy=(end, 2), xytext=(start, 2),
#                 arrowprops=dict(arrowstyle='->', lw=2, color='red'))
#     if label:
#         ax1.text((start+end)/2, 2.3, label, fontsize=10, ha='center', va='center')

# ax1.text(5.5, 0.1, 'Physical Optics View: Free-space Propagation', fontsize=12, ha='center', weight='bold')
# ax1.set_xticks([])
# ax1.set_yticks([])
# for spine in ax1.spines.values():
#     spine.set_visible(False)

# # ===== 下半部分：网络计算视角 =====
# ax2.set_xlim(0, 12)
# ax2.set_ylim(0, 4)
# ax2.set_aspect('equal')

# # 网络节点
# node_positions = [1, 2.5, 4.5, 6.5, 8.5, 10]
# node_counts = [3, 5, 5, 5, 5, 3]  # 每层节点数
# node_labels = ['Input', r'$N_1$', r'$N_2$', r'$N_3$', r'$N_L$', 'Output']

# all_nodes = []
# for i, (x_pos, n_nodes, label) in enumerate(zip(node_positions, node_counts, node_labels)):
#     y_positions = np.linspace(0.5, 3.5, n_nodes)
#     layer_nodes = []
    
#     for j, y_pos in enumerate(y_positions):
#         # 节点颜色基于相位值
#         if i == 0 or i == len(node_positions)-1:
#             color = 'lightblue'
#         else:
#             phase_val = np.sin(2*np.pi*j/n_nodes + i*np.pi/4)
#             color = plt.cm.RdYlBu(0.5 + 0.5*phase_val)
        
#         circle = patches.Circle((x_pos, y_pos), 0.1, facecolor=color, 
#                                edgecolor='black', linewidth=1)
#         ax2.add_patch(circle)
#         layer_nodes.append((x_pos, y_pos))
    
#     all_nodes.append(layer_nodes)
#     ax2.text(x_pos, -0.2, label, fontsize=10, ha='center', va='center', weight='bold')

# # 连接线（只显示部分以避免过于复杂）
# for i in range(len(all_nodes)-1):
#     current_layer = all_nodes[i]
#     next_layer = all_nodes[i+1]
    
#     # 只画部分连接线
#     for j, (x1, y1) in enumerate(current_layer):
#         for k, (x2, y2) in enumerate(next_layer):
#             if j % 2 == 0 or k % 2 == 0:  # 只显示部分连接
#                 alpha = 0.3 if (j + k) % 2 == 0 else 0.1
#                 ax2.plot([x1, x2], [y1, y2], 'k-', alpha=alpha, linewidth=0.5)

# # 色散传播标注
# ax2.text(5.5, 3.8, 'Dispersive Propagation', fontsize=11, ha='center', 
#          style='italic', color='red')

# # 相位颜色条
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlBu'), cax=cbar_ax)
# cbar.set_label('Phase (units of π)', rotation=270, labelpad=15)
# cbar.set_ticks([0, 0.5, 1])
# cbar.set_ticklabels(['-1', '0', '1'])

# ax2.text(5.5, 0.1, 'Network Computation View: Node Connections', fontsize=12, ha='center', weight='bold')
# ax2.set_xticks([])
# ax2.set_yticks([])
# for spine in ax2.spines.values():
#     spine.set_visible(False)

# # 总标题
# fig.suptitle('Optical Diffractive Neural Network (ODNN): Combined Architecture View', 
#              fontsize=16, weight='bold', y=0.95)

# plt.tight_layout()
# plt.subplots_adjust(top=0.9, right=0.9)
# plt.savefig('ODNN_combined_architecture.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('ODNN_combined_architecture.png', dpi=300, bbox_inches='tight')
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 创建竖直色标
fig, ax = plt.subplots(figsize=(0.5, 3))

# 相位从0到2π
phase = np.linspace(0, 2*np.pi, 256).reshape(-1, 1)

# 使用HSV色彩映射
im = ax.imshow(phase, cmap='hsv', aspect='auto')

# 完全去掉所有轴和标签
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')  # 这一行就够了，会去掉所有轴元素

plt.tight_layout()
plt.savefig('phase_colorbar_simple.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
