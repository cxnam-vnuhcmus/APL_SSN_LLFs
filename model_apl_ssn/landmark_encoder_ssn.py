import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling 1D
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.Sigmoid()
        )
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        x1 = self.gap(x)     # (B, d, 1)
        x1 = x1.permute(0,2,1)# (B, 1, d)
        x1 = self.fc(x1) # (B, 1, d)
        
        x2 = self.conv(x) # (B, d, N)
        x2 = x2.permute(0,2,1) # (B, N, d)

        out = x1 * x2 # (B, N, d)
        return out

class LandmarkEncoder(nn.Module):
    def __init__(self, input_dim=(131, 2), output_dim=(5,128), hidden_size=256):
        super(LandmarkEncoder, self).__init__()
        input_size = input_dim[0] * input_dim[1]
        output_size = output_dim[0] * output_dim[1]
        
        self.fc = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)  

        # Channel Attention Module
        self.channel_attention = ChannelAttention(input_dim=hidden_size, hidden_dim=512, output_dim=256)
        

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.fc(x)               # (B, N, d)
        x = x.permute(0,2,1)        # (B, d, N)
        x = self.bn(x)

        attn_w = self.channel_attention(x)  # (B, N, d)
        
        x = x.permute(0,2,1) 
        out = x * attn_w
        
        return out
