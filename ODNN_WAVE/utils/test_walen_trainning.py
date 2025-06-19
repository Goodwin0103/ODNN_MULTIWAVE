import torch
import numpy as np

def test_wavelength_training_step(model, train_loader, criterion, optimizer, device):
    model.train()

    # 1. 记录训练前参数
    phase_before = []
    for layer in model.layers:
        phase_before.append(layer.phase.detach().cpu().numpy().copy())

    # 2. 取一个batch训练一步
    images, labels = next(iter(train_loader))
    images = images.to(device, dtype=torch.complex64)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # 3. 记录训练后参数
    phase_after = []
    for layer in model.layers:
        phase_after.append(layer.phase.detach().cpu().numpy().copy())

    # 4. 计算参数变化范数 & 梯度范数
    for layer_idx, layer in enumerate(model.layers):
        print(f"\nLayer {layer_idx+1}:")
        grad = layer.phase.grad.detach().cpu().numpy()
        for w_idx in range(grad.shape[0]):  # 波长数
            change_norm = np.linalg.norm(phase_after[layer_idx][w_idx] - phase_before[layer_idx][w_idx])
            grad_norm = np.linalg.norm(grad[w_idx])
            print(f"  Wavelength {w_idx}: parameter change norm = {change_norm:.6f}, grad norm = {grad_norm:.6f}")

if __name__ == "__main__":
    from config import Config
    from data_generator import DataGenerator
    from model import WavelengthDependentD2NNModel

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 初始化配置和数据
    config = Config()
    data_generator = DataGenerator(config)
    train_loader = data_generator.create_dataloader()

    # 初始化模型，损失，优化器
    model = WavelengthDependentD2NNModel(config, num_layers=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 运行测试
    test_wavelength_training_step(model, train_loader, criterion, optimizer, device)
