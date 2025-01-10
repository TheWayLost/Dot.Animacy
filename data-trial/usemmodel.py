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
import torchcfm
import numpy as np
import math
from torchcfm.conditional_flow_matching import *
from npyVis import vis
from model import SimpleConv1d, my_wrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigma = 0.0
model = SimpleConv1d().to(device)
node = NeuralODE(my_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

checkpoint = torch.load('/mnt/d/my_github/dotAnimacy/Dot.Animacy/SimpleConv1d_500epoch_smoData_swap.pth')

model.load_state_dict(checkpoint['model_state_dict'])

print("checkpoint loaded")

with torch.no_grad():
    traj = node.trajectory(
        torch.randn(10, 90, 6, device=device),
        t_span=torch.linspace(0, 1, 6, device=device),
    )

result = traj[:, :10].view([-1, 10, 90, 6])
print("type: ",type(result))
print("whole shape: ",result.shape)
result = result.cpu().numpy()

np.save("SimpleConv1d_500epoch_path_smoData_swap.npy",result)