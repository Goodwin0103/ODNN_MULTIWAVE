import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import time
import numpy as np
from config.config import ODNNConfig
from utils.metrics import calculate_visibility
from utils.save_utils import save_model_checkpoint

class ODNNTrainer:
    """ODNN训练器"""
    
    def __init__(self, model, config: ODNNConfig):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        
        # 优化器和调度器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = ExponentialLR(self.optimizer, gamma=config.GAMMA)
        
        # 训练历史
        self.losses = []
        self.best_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        
        for images, labels in train_loader:
            images = images.to(self.device, dtype=torch.complex64)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        self.losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, num_layers):
        """完整训练过程"""
        print(f"\n训练{num_layers}层的ODNN...\n")
        
        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            
            avg_loss = self.train_epoch(train_loader)
            self.scheduler.step()
            
            # 保存最佳模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                save_model_checkpoint(self.model, epoch, avg_loss, num_layers, self.config)
            
            elapsed_time = time.time() - start_time
            
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{self.config.EPOCHS}], '
                      f'Loss: {avg_loss:.18f}, Time: {elapsed_time*100:.2f} seconds')
        
        return self.losses
    
    def evaluate(self, test_loader, evaluation_regions):
        """评估模型"""
        self.model.eval()
        all_weights_pred = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device, dtype=torch.complex64)
                predictions = self.model(images)
                pred_energy = predictions.sum(dim=1)
                predictions_np = pred_energy.cpu().numpy()
                
                for i, prediction in enumerate(predictions_np):
                    weights_pred = []
                    for j, (x_start, x_end, y_start, y_end) in enumerate(evaluation_regions):
                        region_mean = np.mean(prediction[y_start:y_end, x_start:x_end])
                        weights_pred.append(region_mean)
                    all_weights_pred.append(weights_pred)
        
        return np.array(all_weights_pred)
