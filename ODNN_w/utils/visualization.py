import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(all_losses, num_layer_options):
    """绘制训练损失曲线"""
    plt.figure(figsize=(12, 8))
    
    for i, (losses, num_layer) in enumerate(zip(all_losses, num_layer_options)):
        plt.subplot(2, 2, i+1)
        plt.plot(losses)
        plt.title(f'{num_layer} Layers - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_visibility_vs_layers(visibility_list, num_layer_options, num_modes):
    """绘制可见度随层数变化图"""
    visibility_array = np.array(visibility_list)
    
    plt.figure(figsize=(8, 6))
    plt.plot(num_layer_options, visibility_array, marker='o', linestyle='-', color='orange')
    plt.ylim(0, 1)
    plt.xlabel("Number of Layers")
    plt.ylabel("Visibility")
    plt.title(f"{num_modes} modes")
    plt.xticks(num_layer_options)
    
    # 标注数据点
    for x, y in zip(num_layer_options, visibility_array):
        plt.text(x, y - 0.05, f"{y:.3f}", ha='center', va='top', fontsize=9)
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_energy_distribution(weights_pred, num_modes):
    """绘制能量分布图"""
    normalized_weights = weights_pred / np.sum(weights_pred, axis=1, keepdims=True)
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(normalized_weights, cmap='Oranges', interpolation='nearest', vmin=0, vmax=1)
    plt.xlabel('Input Mode Index')
    plt.ylabel('Detector Regions')
    plt.title('Energy Distribution (Percentage)')
    plt.colorbar(im)
    plt.xticks(ticks=np.arange(num_modes), labels=np.arange(1, num_modes + 1))
    plt.yticks(ticks=np.arange(num_modes), labels=np.arange(1, num_modes + 1))
    
    # 添加百分比标注
    num_rows, num_cols = weights_pred.shape
    for i in range(num_rows):
        for j in range(num_cols):
            value = normalized_weights[i, j] * 100
            plt.text(j, i, f"{value:.1f}", ha='center', va='center', 
                    color='black', fontsize=8)
    
    plt.tight_layout()
    plt.show()
