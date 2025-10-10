import torch
import numpy as np
import matplotlib.pyplot as plt

def verify_wavelength_training(model, train_loader, criterion, optimizer, device):
    model.train()

    # 1. 打印相位掩膜形状，确认只有一个掩膜
    print("Phase mask shapes per layer:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i+1}: {layer.phase.shape}")

    # 2. 训练一步，记录参数前后变化和梯度
    phase_before = [layer.phase.detach().cpu().numpy().copy() for layer in model.layers]

    images, labels = next(iter(train_loader))
    images = images.to(device, dtype=torch.complex64)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    phase_after = [layer.phase.detach().cpu().numpy().copy() for layer in model.layers]

    print("\n参数变化和梯度范数（训练一步后）:")
    for layer_idx, layer in enumerate(model.layers):
        grad = layer.phase.grad.detach().cpu().numpy()
        change_norm = np.linalg.norm(phase_after[layer_idx] - phase_before[layer_idx])
        grad_norm = np.linalg.norm(grad)
        print(f" 层 {layer_idx+1}: 变化范数={change_norm:.6f}, 梯度范数={grad_norm:.6f}")

    # 3. 可视化示例：某层训练前后相位对比
    layer_to_plot = 0  # 选择第1层
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].imshow(phase_before[layer_to_plot], cmap='jet')
    axes[0].set_title(f'Layer {layer_to_plot+1} Before')
    axes[1].imshow(phase_after[layer_to_plot], cmap='jet')
    axes[1].set_title(f'Layer {layer_to_plot+1} After')
    plt.show()



# ====== 使用示例 ======
if __name__ == "__main__":
    from config import Config
    from data_generator import DataGenerator
    from model import WavelengthDependentD2NNModel

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = Config()
    data_generator = DataGenerator(config)
    train_loader = data_generator.create_dataloader()

    model = WavelengthDependentD2NNModel(config, num_layers=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    verify_wavelength_training(model, train_loader, criterion, optimizer, device)
