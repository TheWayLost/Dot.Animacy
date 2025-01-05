import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def halfframe(data):
    return data[::2,:]

class kofDataset(Dataset):
    def __init__(self, folder_path, transform=True):
        """
        参数:
        - folder_path: 存放 `.npy` 文件的文件夹路径
        - transform: 可选的转换操作（如归一化、数据增强等）
        """
        self.folder_path = folder_path
        self.file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
        self.transform = transform
        if self.transform:
            print("Apply Transformation")
        self.x = 1280 / 2
        self.y = 720 / 2
        self.r = 30 / 2

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        data = np.load(file_path)
        data = torch.tensor(data, dtype=torch.float32)
        
        if self.transform:
            data = self.apply_transform(data)
        return data
    
    def apply_transform(self, data):
        data[:, 0] /= self.x
        data[:, 1] /= self.y
        data[:, 2] *= self.r / torch.max(data[:, 2])
        data[:, 3] /= self.x
        data[:, 4] /= self.y
        data[:, 5] *= self.r / torch.max(data[:, 5])
        data -= 1
        return data 
    
    def inv_transform(self, data):
        data += 1
        data[:, :, 0] *= self.x
        data[:, :, 1] *= self.y
        data[:, :, 2] *= self.r
        data[:, :, 3] *= self.x
        data[:, :, 4] *= self.y
        data[:, :, 5] *= self.r
        return data.cpu().numpy().astype(np.int32)