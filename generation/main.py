import os
import torch
import torch.nn.functional as F
from torchdyn.core import NeuralODE
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchcfm.conditional_flow_matching import *
from utils import vis
from dataset import kofDataset, DataLoader
from model import my_wrapper, SimpleConv1d
from datetime import datetime


if __name__ == "__main__":
    data_folder = "kof_data/segment_slice"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = os.path.join("logs", current_time)

    train_dataset = kofDataset(data_folder, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = SummaryWriter(log_dir)
    #################################
    #            Simpleconv1d
    #################################
    sigma = 0.0
    model = SimpleConv1d().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
    # FM = ConditionalFlowMatcher(sigma=sigma)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    node = NeuralODE(
        my_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    for epoch in range(20):
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

    # 训练完成后保存模型的 checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss.item(),
    }

    # 保存到文件
    torch.save(checkpoint, f"{log_dir}/SimpleConv1d_500epoch.pth")

    with torch.no_grad():
        traj = node.trajectory(
            torch.randn(10, 180, 6, device=device),
            t_span=torch.linspace(0, 1, 2, device=device),
        )

    result = traj[-1, :10].view([-1, 180, 6])

    print("type: ", type(result))
    print("whole shape: ", result.shape)
    result = result.cpu().numpy()
    np.save(f"{log_dir}/SimpleConv1d_500e_10sample.npy", result)
    vis(result[0])
