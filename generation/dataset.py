import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def halfframe(data):
    return data[::2,:]

def remove_r(data):
    return data[:, [0, 1, 3, 4]]

class kofDataset(Dataset):
    def __init__(self, folder_path, transform=True, normalize=False):
        """
        参数:
        - folder_path: 存放 `.npy` 文件的文件夹路径
        - transform: 可选的转换操作（如归一化、数据增强等）
        """
        self.folder_path = folder_path
        self.file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
        self.normalize = normalize
        self.transform = transform
        if self.normalize:
            print("Apply Transformation")
        self.x = 1280
        self.y = 720
        self.r = 50

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        data = np.load(file_path)
        data = torch.tensor(data, dtype=torch.float32)
        
        if self.normalize:
            data = self.apply_normalize(data)
        if self.transform:
            data = self.apply_transform(data)
        data_rm_r = remove_r(data)  
        return data_rm_r
    
    def apply_transform(self, data):
        if np.random.rand() > 0.5: # flip x
            data[:, 0] = self.x - data[:, 0]
            data[:, 3] = self.x - data[:, 3]
        if np.random.rand() > 0.5: # flip y
            data[:, 1] = self.y - data[:, 1]
            data[:, 4] = self.y - data[:, 4]
        if np.random.rand() > 0.5: # swap
            data[:, 0:3], data[:, 3:6] = data[:, 3:6], data[:, 0:3]
        return data
    
    def apply_normalize(self, data):
        # scale data to [0, 1]
        data[:, 0] /= self.x
        data[:, 1] /= self.y
        data[:, 2] /= self.r
        data[:, 3] /= self.x
        data[:, 4] /= self.y
        data[:, 5] /= self.r
        # scale data to [-1, 1]
        # data = data * 2 - 1
        # data = torch.clamp(data, -1, 1)
        return data 
    
    def inv_normalize(self, data):
        # here do not use inplace operation for data
        output = torch.zeros_like(data)
        output[:, :, 0] = self.x * data[:, :, 0]
        output[:, :, 1] = self.y * data[:, :, 1]
        output[:, :, 2] = self.x * data[:, :, 2]
        output[:, :, 3] = self.y * data[:, :, 3]
        # output[:, :, 2] = torch.abs(self.r * data[:, :, 2])
        # output[:, :, 3] = self.x * data[:, :, 3]
        # output[:, :, 4] = self.y * data[:, :, 4]
        # output[:, :, 5] = torch.abs(self.r * data[:, :, 5])
        return output
    
    def vel_penalty(self, data, tolorance=200):
        # here do not use inplace operation for data
        # data shape: batch_size, frames, channels = 6
        if self.normalize: pos = self.inv_normalize(data)
        else: pos = data
        vel = pos[:, 1:] - pos[:, :-1]
        vel[abs(vel) < tolorance] = 0
        vel_a = torch.norm(vel[:, :, :2], dim=2)
        vel_b = torch.norm(vel[:, :, 2:], dim=2)
        # penalty = mean(vel ** 2) 
        penalty_a = torch.mean(vel_a ** 2, dim=1)
        penalty_b = torch.mean(vel_b ** 2, dim=1)
        penalty = (penalty_a.sum() + penalty_b.sum())
        return penalty * 1e-2
    
    def pos_penalty(self, data):
        # here do not use inplace operation for data
        # data shape: batch_size, frames, channels = 6
        if self.normalize: pos = self.inv_normalize(data)
        else: pos = data
        def out_of_x(pos):
            return torch.sum(- pos[pos<0]) + torch.sum(pos[pos>self.x]-self.x)
        def out_of_y(pos):
            return torch.sum(- pos[pos<0]) + torch.sum(pos[pos>self.y]-self.y)
        penalty = out_of_x(pos[:, :, 0]) + out_of_y(pos[:, :, 1]) + out_of_x(pos[:, :, 2]) + out_of_y(pos[:, :, 3])
        return penalty * 1e-1