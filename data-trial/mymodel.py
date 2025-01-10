import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
from torchdyn.core import NeuralODE
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm
import torchdiffeq
import torchcfm
import numpy as np
import math
from torchcfm.conditional_flow_matching import *
from npyVis import vis

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
        return len(self.file_names)*2

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        data = np.load(file_path)
        data = torch.tensor(data, dtype=torch.float32)

        if self.transform:
            data = self.transform(data)
        return data
    
data_folder = "/mnt/d/my_github/dotAnimacy/Dot.Animacy/kof_data/segmentation_split/"
train_dataset = kofDataset(data_folder, transform=halfframe)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


overfit_test_data = np.load(data_folder+"1.npy")



class my_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, args=None):
        if isinstance(x, tuple):
            data, cond = x
        else:
            data, cond = x, args
        return self.model(data, t.expand(data.shape[0]), cond)

class SimpleConv1d(nn.Module):
    def __init__(self):
        super(SimpleConv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, 6)
        self.condition_proj = nn.Linear(2 * 6, 18)
        self.relu = nn.ReLU()

    def time_embedding(self, times, embedding_dim=8):
        if not isinstance(embedding_dim, int):
            print(embedding_dim)
            raise ValueError("Embedding dimension must be an integer.")
        assert embedding_dim % 2 == 0, "Embedding dimension must be even."

        device = times.device  # Keep the tensor on the same device as the input
        times = times.unsqueeze(-1)  # Shape (batch_size, 1)
        
        # Define frequencies for the embedding
        frequencies = torch.pow(10000, torch.arange(0, embedding_dim, 2, device=device) / embedding_dim)
        
        # Calculate embeddings using sin and cos
        sinusoids = times / frequencies  # Broadcasting
        embeddings = torch.zeros(times.size(0), embedding_dim, device=device)
        embeddings[:, 0::2] = torch.sin(sinusoids)
        embeddings[:, 1::2] = torch.cos(sinusoids)
        
        return embeddings

    def forward(self, x, t, y):
        # x = torch.cat((x,t.reshape(-1,1,1).expand(-1,90,-1)),dim=2)
        cond = self.condition_proj(y.view(y.size(0),-1)).unsqueeze(1).repeat(1, x.size(1), 1)
        tt = self.time_embedding(t)
        x = torch.cat((x, tt.unsqueeze(1).repeat(1, 90,1), cond),dim=2)
        x = x.permute(0, 2, 1)  # (batch_size, feature_size, seq_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x



#################################
#            Simpleconv1d
#################################

sigma = 0.0
model = SimpleConv1d().to(device)
optimizer = torch.optim.Adam(model.parameters())
# FM = ConditionalFlowMatcher(sigma=sigma)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
node = NeuralODE(my_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
for epoch in range(400):
    with tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True) as pbar:
        for i, data in pbar:
            optimizer.zero_grad()
            x1 = data.to(device)
            y = x1[:,[0,89],:]
            x0 = torch.randn_like(x1)
            t, xt, ut,_,y1 = FM.guided_sample_location_and_conditional_flow(x0, x1,y1=y)
            # print(t)
            # print(t.shape)
            #xtt = torch.cat((xt,t.reshape(-1,1,1).expand(-1,90,-1)),dim=2)
            #print(xtt.shape)
            #vt = model(xtt)
            vt = model(xt,t,y1)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

# 训练完成后保存模型的 checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss.item(),
}

# 保存到文件
torch.save(checkpoint, 'SimpleConv1d_cond_222epoch.pth')

with torch.no_grad():
    # traj = node.trajectory(
    #     torch.randn(10, 90, 6, device=device),
    #     t_span=torch.linspace(0, 1, 2, device=device),
    # )
    traj = torchdiffeq.odeint(
            lambda t, x: model.forward(
                x,
                t.expand(x.shape[0]),
                torch.tensor(
                    overfit_test_data[[0,178],:],
                    device=device,  # Ensure the tensor is on the correct device
                    dtype=torch.float32  # Match model's expected dtype
                ).unsqueeze(0).repeat(x.shape[0], 1, 1)
            ),
            torch.randn(10, 90, 6, device=device),
            torch.linspace(0, 1, 2, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

result = traj[-1, :10].view([-1, 90, 6])

print("cond: ",overfit_test_data[[0,178],:])
print("type: ",type(result))
print("whole shape: ",result.shape)
result = result.cpu().numpy()
np.save("SimpleConv1d_cond_222e_10sample.npy",result)
print(result[0][[0,89],:])
print(result[1][[0,89],:])
vis(result[0])


#################################
#            Transformer
#################################
# 
# sigma = 0.0
# model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
# optimizer = torch.optim.Adam(model.parameters())
# # FM = ConditionalFlowMatcher(sigma=sigma)
# FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
# node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
# for epoch in range(n_epochs):
#     for i, data in tqdm(enumerate(train_loader)):
#         optimizer.zero_grad()
#         x1 = data[0].to(device)
#         x0 = torch.randn_like(x1)
#         t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
#         vt = model(t, xt)
#         loss = torch.mean((vt - ut) ** 2)
#         loss.backward()
#         optimizer.step()
# 
# with torch.no_grad():
#     traj = node.trajectory(
#         torch.randn(100, 1, 28, 28, device=device),
#         t_span=torch.linspace(0, 1, 2, device=device),
#     )
# grid = make_grid(
#     traj[-1, :100].view([-1, 1, 28, 28]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
# )
# img = ToPILImage()(grid)
# plt.imshow(img)