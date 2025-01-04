import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
from torchdyn.core import NeuralODE
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import math
from torchcfm.conditional_flow_matching import *

    
class my_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, args=None):
        return self.model(torch.cat((x,t.expand(x.shape[0],180,1)),dim=2))

class SimpleConv1d(nn.Module):
    def __init__(self):
        super(SimpleConv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = torch.cat((x,t.reshape(-1,1,1).expand(-1,90,-1)),dim=2)
        x = x.permute(0, 2, 1)  # (batch_size, feature_size, seq_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x

# Sinusoidal Positional Encoding
def get_positional_encoding(seq_len, d_model):
    position = torch.arange(0, seq_len).float().unsqueeze(1)  # shape (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # shape (d_model//2,)
    pos_embedding = torch.zeros(seq_len, d_model)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)  # Sinusoidal for even indices
    pos_embedding[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd indices
    
    print(pos_embedding.shape)
    return pos_embedding

# Sinusoidal Condition Embedding
def get_sinusoidal_condition_embedding(condition, d_model):
    """
    condition: 输入的flow matching的t，范围 [0, 1]
    d_model: 嵌入维度
    返回值：大小为 (1, d_model) 的 Sinusoidal Embedding
    """
    # 转换为对应的 "位置"
    condition = condition.unsqueeze(1)  # (batch_size, 1)
    
    # 计算每个维度的 sin 和 cos 嵌入
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # shape (d_model//2,)
    
    condition_embedding = torch.zeros(1, d_model)
    condition_embedding[:, 0::2] = torch.sin(condition * div_term)  # Sinusoidal for even indices
    condition_embedding[:, 1::2] = torch.cos(condition * div_term)  # Cosine for odd indices
    
    return condition_embedding



class Conv1dModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = get_positional_encoding(90, embed_dim)  # 90 time steps
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding[:x.size(1), :]
        for layer in self.layers:
            embedded = layer(embedded)
        return embedded


class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, num_heads, num_layers, ff_dim):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(output_dim, embed_dim)
        self.positional_encoding = get_positional_encoding(90, embed_dim)  # 90 time steps
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
            for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x, memory, condition_emb):
        # Add positional encoding to target sequence
        embedded = self.embedding(x) + self.positional_encoding[:x.size(1), :]
        # Apply the condition embedding to the decoder output at each step
        for layer in self.layers:
            embedded = layer(embedded, memory)
            embedded = embedded + condition_emb  # Adding condition to the decoder output
        logits = self.output_linear(embedded)
        return logits


class ConditionalTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, num_layers, ff_dim):
        super(ConditionalTransformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, embed_dim, num_heads, num_layers, ff_dim)
        self.decoder = TransformerDecoder(output_dim, embed_dim, num_heads, num_layers, ff_dim)

    def forward(self, src, trg, condition):
        # Generate Sinusoidal condition embedding
        condition_emb = get_sinusoidal_condition_embedding(condition, src.size(-1))  # Size of condition is (batch_size, 1)
        # Encoder processes the source sequence
        memory = self.encoder(src)
        # Decoder generates the output sequence based on the encoder's memory and the condition embedding
        output = self.decoder(trg, memory, condition_emb)
        return output



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


