import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkDecoder(nn.Module):
    def __init__(self, input_dim=(5,256), hidden_dim=256, output_dim=(131,2)):
        super(LandmarkDecoder, self).__init__()
        
        input_size = input_dim[0] * input_dim[1]
        output_size = output_dim[0] * output_dim[1]
        
        # Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            # nn.Sigmoid()
        )
        
        # Normalization
        self.norm = nn.LayerNorm(output_size)
        
        
    def forward(self, audio_feature, landmark_feature, llfs_feature=None):
        assert audio_feature.shape == landmark_feature.shape, "audio_feature and landmark_feature must have the same shape"
        
        # Kết hợp các đặc trưng
        combined_feature = audio_feature + landmark_feature  # (B, N, 256)
        attn_weight = F.softmax(llfs_feature, dim=1)  # (B, 1, 256)
        #Old
        x, _ = self.attention(attn_weight, combined_feature, combined_feature)  # (B, N, hidden_dim)
        #New
        # x, _ = self.attention(combined_feature, llfs_feature, llfs_feature)  # (B, N, hidden_dim)

        x = x.reshape(x.shape[0], -1)
        
        output = self.fc(x)  # (B, output_dim)
        output = self.norm(output)
        output = output.reshape(output.shape[0], -1, 2)  # Reshape thành (B, 131, 2)
        
        return output
