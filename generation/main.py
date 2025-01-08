import os
import torch
import torch.nn.functional as F
from torchdyn.core import NeuralODE
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchcfm.conditional_flow_matching import *
from utils import visualizer
from dataset import kofDataset, DataLoader
from model import transformer_wrapper, conv1d_wrapper, SimpleConv1d, BaseModel
from datetime import datetime


def save(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_dir: str,
    node: NeuralODE,
    train_dataset: kofDataset,
    device: torch.device,
    encoder = 'H264'
    ):
    # 训练完成后保存模型的 checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    # 保存到文件
    torch.save(checkpoint, f"{log_dir}/ckpts/BaseModel_{epoch}epoch.pth")

    result = test(node, device)
    if train_dataset.normalize:
        result = train_dataset.inv_normalize(result)
    result = result.cpu().numpy()
    print("sample shape: ", result.shape)
    
    vis = visualizer(720, 1280, 30, 6)
    result = np.array(result, dtype=np.int64)
    r = 30
    data = np.ones((result.shape[0], result.shape[1], 6), dtype=np.int64) * r
    data[:, :, [0, 1, 3, 4]] = result
    os.makedirs(f"{log_dir}/vis/{epoch}", exist_ok=True)
    np.save(f"{log_dir}/vis/{epoch}/BaseModel_10sample.npy", data)
    for i in range(10):
        vis.vis_as_video(data[i], f"{log_dir}/vis/{epoch}/BaseModel_sample{i}.mp4", frame_rate=30, encoder=encoder)
        vis.vis_trajectory_with_line(data[i], f"{log_dir}/vis/{epoch}/BaseModel_sample{i}_line.png")

def test(node, device):
    with torch.no_grad():
        traj = node.trajectory(
            torch.randn(10, 180, 4, device=device),
            t_span=torch.linspace(0, 1, 2, device=device),
        )
        # print("traj shape: ", traj.shape)
    result = traj[-1].view([-1, 180, 4])
    return result

def train(
    max_epoch: int, 
    model, 
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    logger: SummaryWriter,
    FM: ConditionalFlowMatcher,
    node: NeuralODE,
    train_loader: DataLoader,
    train_dataset: kofDataset, 
    device):
    for epoch in range(1, max_epoch+1):
            avg_mse_loss = 0
            avg_penalty = 0
            with tqdm(
                enumerate(train_loader), desc=f"Epoch {epoch}", dynamic_ncols=True
            ) as pbar:
                for i, data in pbar:
                    optimizer.zero_grad()
                    x1 = data.to(device)
                    x0 = torch.randn_like(x1)
                    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                    vt = model(xt, t)
                    mes_loss = F.mse_loss(vt, ut) 
                    penalty_loss = train_dataset.vel_penalty(vt) + train_dataset.pos_penalty(vt)
                    avg_mse_loss += mes_loss.item()
                    avg_penalty += penalty_loss.item()
                    loss = mes_loss + penalty_loss
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())

            # lr_scheduler.step()1. model size 2. lr_scheduler 3. loss 4. batch_size
            
            avg_mse_loss /= len(train_loader)
            avg_penalty /= len(train_loader)
            logger.add_scalar("MSE_Loss/epoch", avg_mse_loss, epoch)
            logger.add_scalar("Velocity Penalty/epoch", avg_penalty, epoch)
            result = test(node, device)
            logger.add_scalar("Velocity Penalty/test", train_dataset.vel_penalty(result), epoch)
            
            if epoch % 10 == 0:
                save(model, optimizer, epoch, log_dir, node, train_dataset, device)

def train_conv1d(
    max_epoch: int,
    device: torch.device,
    log_dir: str,
    train_loader: DataLoader,
    train_dataset: kofDataset,
    logger: SummaryWriter
    ):
    #################################
    #            Simpleconv1d
    #################################
    sigma = 0.0
    model = SimpleConv1d().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max_epoch//5, gamma=0.5)
    # FM = ConditionalFlowMatcher(sigma=sigma)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    node = NeuralODE(
        conv1d_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    try:
        train(max_epoch, model, optimizer, lr_scheduler, logger, FM, node, train_loader, train_dataset, device)
    except KeyboardInterrupt:
        print("Training interrupted")
    
    save(model, optimizer, max_epoch, log_dir, node, train_dataset, device)

def train_basemodel(
    max_epoch: int,
    device: torch.device,
    log_dir: str,
    train_loader: DataLoader,
    train_dataset: kofDataset,
    logger: SummaryWriter
    ):
    #################################
    #            BaseModel
    #################################
    sigma = 0.0
    model = BaseModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max_epoch//3, gamma=0.1)
    # FM = ConditionalFlowMatcher(sigma=sigma)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    node = NeuralODE(
        transformer_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    try:
        train(max_epoch, model, optimizer, lr_scheduler, logger, FM, node, train_loader, train_dataset, device)
    except KeyboardInterrupt:
        print("Training interrupted")
        
    save(model, optimizer, max_epoch, log_dir, node, train_dataset, device)


if __name__ == "__main__":
    model = "BaseModel" 
    # model = "Conv1d"
    max_epoch = 200
    # fix seed
    torch.manual_seed(0)
    np.random.seed(0)
    
    data_folder = "kof_data/segment_slice"
    # exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = f"{model}_wo_Normalize_{max_epoch}epoch_linear_"+datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "vis"), exist_ok=True)

    train_dataset = kofDataset(data_folder, normalize=False, transform=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = SummaryWriter(log_dir)
    
    if model.lower() == "conv1d":
        train_conv1d(max_epoch, device, log_dir, train_loader, train_dataset, logger)
    elif model.lower() == "basemodel":
        train_basemodel(max_epoch, device, log_dir, train_loader, train_dataset, logger)
    else: raise ValueError("Model not found")
    
    