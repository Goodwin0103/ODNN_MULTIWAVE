import numpy as np

def calculate_visibility(weights_pred, num_modes):
    """计算可见度"""
    energy_distribute = np.zeros((num_modes, num_modes))
    energy_sum = np.sum(weights_pred, axis=1) + 1e-12  # 避免除以零
    
    for i in range(num_modes):
        for j in range(num_modes):
            energy_distribute[j, i] = weights_pred[i][j] / energy_sum[i]
    
    visibility = np.average(np.diag(energy_distribute))
    return visibility

def calculate_crosstalk(weights_pred, num_modes):
    """计算串扰"""
    normalized_weights = weights_pred / np.sum(weights_pred, axis=1, keepdims=True)
    
    # 对角线元素是目标信号
    diagonal_sum = np.sum(np.diag(normalized_weights))
    
    # 非对角线元素是串扰
    total_sum = np.sum(normalized_weights)
    crosstalk = (total_sum - diagonal_sum) / total_sum
    
    return crosstalk
