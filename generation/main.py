import os
import torch
import torch.nn.functional as F
from torchdyn.core import NeuralODE
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchcfm.conditional_flow_matching import *
from utils import visulizer
from dataset import kofDataset, DataLoader
from model import transformer_wrapper, conv1d_wrapper, SimpleConv1d, BaseModel
from datetime import datetime


def train_conv1d():
    data_folder = "kof_data/segment_slice"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = os.path.join("logs", current_time)

    train_dataset = kofDataset(data_folder, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = SummaryWriter(log_dir)
    max_epoch = 500
    
    #################################
    #            Simpleconv1d
    #################################
    sigma = 0.0
    model = SimpleConv1d().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # FM = ConditionalFlowMatcher(sigma=sigma)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    node = NeuralODE(
        conv1d_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    try:
        for epoch in range(1, max_epoch+1):
            avg_epoch_loss = 0
            with tqdm(
                enumerate(train_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True
            ) as pbar:
                for i, data in pbar:
                    optimizer.zero_grad()
                    x1 = data.to(device)
                    x0 = torch.randn_like(x1)
                    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                    # print(t.shape)
                    xtt = torch.cat((xt, t.reshape(-1, 1, 1).expand(-1, 180, -1)), dim=2)
                    # print(xtt.shape)
                    vt = model(xtt)
                    loss = F.mse_loss(vt, ut)
                    avg_epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())
            avg_epoch_loss /= len(train_loader)
            logger.add_scalar("Loss/epoch", avg_epoch_loss, epoch)
            lr_scheduler.step()
    except KeyboardInterrupt:
        print("Training interrupted")
    # 训练完成后保存模型的 checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss.item(),
    }

    # 保存到文件
    torch.save(checkpoint, f"{log_dir}/SimpleConv1d_{epoch}epoch.pth")

    with torch.no_grad():
        traj = node.trajectory(
            torch.randn(10, 180, 6, device=device),
            t_span=torch.linspace(0, 1, 2, device=device),
        )

    result = traj[-1, :10].view([-1, 180, 6])

    print("type: ", type(result))
    print("whole shape: ", result.shape)
    result = result.cpu().numpy()
    np.save(f"{log_dir}/SimpleConv1d_{epoch}e_10sample.npy", result)
    
    vis = visulizer(720, 1280, 30, 6)
    vis.vis_as_video(result[0], f"{log_dir}/SimpleConv1d_{epoch}e_10sample.mp4", frame_rate=30)


def train_basemodel():
    data_folder = "kof_data/segment_slice"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = os.path.join("logs", current_time)

    train_dataset = kofDataset(data_folder)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = SummaryWriter(log_dir)
    max_epoch = 100
    
    #################################
    #            BaseModel
    #################################
    sigma = 0.0
    model = BaseModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # FM = ConditionalFlowMatcher(sigma=sigma)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    node = NeuralODE(
        transformer_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    try:
        for epoch in range(1, max_epoch+1):
            avg_epoch_loss = 0
            with tqdm(
                enumerate(train_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True
            ) as pbar:
                for i, data in pbar:
                    optimizer.zero_grad()
                    x1 = data.to(device)
                    x0 = torch.randn_like(x1)
                    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                    vt = model(xt, t)
                    loss = F.mse_loss(vt, ut)
                    avg_epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())
            avg_epoch_loss /= len(train_loader)
            logger.add_scalar("Loss/epoch", avg_epoch_loss, epoch)
            lr_scheduler.step()
    except KeyboardInterrupt:
        print("Training interrupted")
    # 训练完成后保存模型的 checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss.item(),
    }

    # 保存到文件
    torch.save(checkpoint, f"{log_dir}/BaseModel_{epoch}epoch.pth")

    with torch.no_grad():
        traj = node.trajectory(
            torch.randn(10, 180, 6, device=device),
            t_span=torch.linspace(0, 1, 2, device=device),
        )

    result = traj[-1, :10].view([-1, 180, 6])
    if train_dataset.transform:
        result = train_dataset.inv_transform(result)

    print("type: ", type(result))
    print("whole shape: ", result.shape)
    np.save(f"{log_dir}/BaseModel_{epoch}e_10sample.npy", result)
    
    vis = visulizer(720, 1280, 30, 6)
    for i in range(10):
        vis.vis_as_video(result[i], f"{log_dir}/BaseModel_{epoch}e_sample{i}.mp4", frame_rate=30)



if __name__ == "__main__":
    # fix seed
    torch.manual_seed(0)
    np.random.seed(0)
    
    # train_conv1d()
    train_basemodel()
    
    # data = np.load("logs/2025-01-04_21-40/BaseModel_200e_10sample.npy")
    # data = np.array(data[0], dtype=np.int64)
    # data = abs(data)
    # vis = visulizer(720, 1280, 30, 6)
    # vis.vis_as_video(data, "logs/2025-01-04_21-40/test.mp4", frame_rate=30)
    