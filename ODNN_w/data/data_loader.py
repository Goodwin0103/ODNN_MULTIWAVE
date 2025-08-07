import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ODNN_functions import generate_fields_ts, create_labels
from config.config import ODNNConfig

class ODNNDataLoader:
    """ODNN数据加载器"""
    
    def __init__(self, config: ODNNConfig):
        self.config = config
        self.mmf_data = None
        self.complex_weights = None
        
    def load_mmf_data(self, file_path='eigenmodes_OM4.npy'):
        """加载多模光纤数据"""
        eigenmodes_OM4 = np.load(file_path)
        mmf_data = eigenmodes_OM4[:, :, 0:self.config.NUM_MODES].transpose(2, 0, 1)
        
        # 振幅归一化但保持相位
        mmf_data_amp_norm = (np.abs(mmf_data) - np.min(np.abs(mmf_data))) / \
                           (np.max(np.abs(mmf_data)) - np.min(np.abs(mmf_data)))
        self.mmf_data = mmf_data_amp_norm * np.exp(1j * np.angle(mmf_data))
        
        return torch.from_numpy(self.mmf_data)
    
    def generate_complex_weights(self):
        """生成复权重"""
        if self.config.PHASE_OPTION == 4:
            amplitudes = np.eye(self.config.NUM_MODES)
            phases = np.eye(self.config.NUM_MODES)
        
        self.complex_weights = amplitudes * np.exp(1j * phases)
        return torch.from_numpy(self.complex_weights)
    
    def generate_input_data(self):
        """生成多波长输入数据"""
        mmf_data_ts = self.load_mmf_data()
        complex_weights_ts = self.generate_complex_weights()
        
        image_data_multi = []
        for i in range(self.config.NUM_MODES):
            fields = []
            for wl in self.config.WAVELENGTHS:
                cw_batch = complex_weights_ts[i].unsqueeze(0)
                field = generate_fields_ts(
                    cw_batch, mmf_data_ts, num_data=1,
                    num_modes=self.config.NUM_MODES,
                    image_size=self.config.FIELD_SIZE,
                    wavelength=wl
                )
                f2d = field.squeeze(0).squeeze(0)
                fields.append(f2d)
            image_data_multi.append(torch.stack(fields, dim=0))
        
        return torch.stack(image_data_multi, dim=0)
    
    def generate_labels(self):
        """生成标签数据"""
        amplitudes = np.eye(self.config.NUM_MODES)
        phases = np.eye(self.config.NUM_MODES)
        amplitudes_phases = np.hstack((amplitudes[:, :], phases[:, 1:] / (2 * np.pi)))
        
        label_data = torch.zeros([self.config.NUM_MODES, 1, 
                                 self.config.LAYER_SIZE, self.config.LAYER_SIZE])
        
        MMF_Label_data = torch.zeros([self.config.LAYER_SIZE, self.config.LAYER_SIZE, 
                                     self.config.NUM_MODES])
        
        for index in range(self.config.NUM_MODES):
            MMF_Label_data[:, :, index] = torch.from_numpy(
                create_labels(self.config.LAYER_SIZE, self.config.LAYER_SIZE,
                             self.config.NUM_MODES, self.config.FOCUS_RADIUS, index + 1)
            )
        
        for index in range(self.config.NUM_MODES):
            label_data[index, :, :, :] = (
                torch.from_numpy(amplitudes_phases[index, 0:self.config.NUM_MODES]) * 
                MMF_Label_data
            ).sum(dim=2)
        
        return label_data
    
    def preprocess(self, images, label):
        """数据预处理"""
        padding_size = (self.config.LAYER_SIZE - self.config.FIELD_SIZE) // 2
        padding = (padding_size, padding_size, padding_size, padding_size)
        img_pad = F.pad(images, padding)
        return img_pad, label
    
    def create_dataloader(self):
        """创建数据加载器"""
        image_data_multi = self.generate_input_data()
        label_data = self.generate_labels()
        
        train_dataset = []
        for i in range(len(label_data)):
            img_pad, _ = self.preprocess(image_data_multi[i], None)
            lbl = label_data[i, 0]
            lbl_3chan = torch.stack([lbl] * 3, dim=0)
            train_dataset.append((img_pad, lbl_3chan))
        
        train_tensor_data = TensorDataset(*[torch.stack(tensors) for tensors in zip(*train_dataset)])
        train_loader = DataLoader(train_tensor_data, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, train_dataset
