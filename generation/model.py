import torch
import torch.nn as nn
import math
from torchcfm.conditional_flow_matching import *

FRAMES = 180
    
class conv1d_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, args=None):
        return self.model(x, t.expand(x.shape[0]))
    
class transformer_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, args=None):
        return self.model(x, t.expand(x.shape[0]))

class SimpleConv1d(nn.Module):
    def __init__(self):
        super(SimpleConv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, 4)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        x = torch.cat((x, t.unsqueeze(-1).repeat(1, FRAMES).unsqueeze(-1)),dim=2)
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
        self.positional_encoding = get_positional_encoding(FRAMES, embed_dim)  # FRAMES time steps
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
        self.positional_encoding = get_positional_encoding(FRAMES, embed_dim)  # FRAMES time steps
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


class BaseModel(nn.Module):
    def __init__(
        self, 
        in_dim=4, out_dim=4*FRAMES, 
        x_embed_dim=48, t_embed_dim=16,
        embed_dim=256, num_heads=4, num_layers=2, ff_dim=128):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proj_dim = int(x_embed_dim * FRAMES // 9) # noqa, only work when FRAMES % 9 == 0
        self.emb_x = nn.Sequential(
            nn.Conv1d(in_dim, x_embed_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(x_embed_dim//2),
            nn.AdaptiveAvgPool1d(FRAMES//3),
            nn.Conv1d(x_embed_dim//2, x_embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(x_embed_dim),
            nn.AdaptiveAvgPool1d(FRAMES//9),
        )
        self.emb_t = nn.Sequential(
            nn.Linear(1, t_embed_dim),
            nn.GELU()
        )
        self.proj1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(proj_dim, embed_dim),
            nn.GELU()
        )
        self.proj2 = nn.Sequential(
            nn.Linear(embed_dim+t_embed_dim, embed_dim),
            nn.GELU(),
        )
        self.positional_encoding = get_positional_encoding(FRAMES//9, x_embed_dim).unsqueeze(0).to(self.device)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
            for _ in range(num_layers)
        ])
        self.layers = nn.Sequential(*self.layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x, t):
        B, T, C = x.size() # batch_size, frames, channels = 6
        assert T == FRAMES
        x = self.emb_x(x.permute(0, 2, 1))
        x = self.proj1(x.permute(0, 2, 1) + self.positional_encoding) 
        t = self.emb_t(t.unsqueeze(-1))
        x = self.proj2(torch.cat([x, t], dim=1))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.mlp2(x)
        x = x.view(B, T, -1)
        return x