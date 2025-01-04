import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def halfframe(data):
    return data[::2,:]

class kofDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        参数:
        - folder_path: 存放 `.npy` 文件的文件夹路径
        - transform: 可选的转换操作（如归一化、数据增强等）
        """
        self.folder_path = folder_path
        self.file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        data = np.load(file_path)
        data = torch.tensor(data, dtype=torch.float32)
        
        if self.transform:
            data = self.transform(data)
        return data