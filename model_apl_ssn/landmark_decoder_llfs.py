import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkDecoder(nn.Module):
    def __init__(self, input_dim=(5,256), hidden_dim=512, output_dim=(131,2), emo_dim=11):
        super(LandmarkDecoder, self).__init__()
        
        input_size = input_dim[0] * input_dim[1]
        output_size = output_dim[0] * output_dim[1]
        
        self.linear1 = nn.Linear(input_size, hidden_dim)        
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)        
        self.shortcut = nn.Linear(input_size, hidden_dim)        
        self.projection = nn.Linear(hidden_dim, output_size + emo_dim)
        self.norm = nn.LayerNorm(output_size + emo_dim)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, audio_feature, landmark_feature, llfs_feature=None):
        assert audio_feature.shape == landmark_feature.shape, "audio_feature and landmark_feature must have the same shape"
        
        combined_feature = audio_feature + landmark_feature  # (B, N, 128)     
        if llfs_feature is not None:
            combined_feature = combined_feature + llfs_feature
        combined_feature = combined_feature.reshape(combined_feature.shape[0], -1)
        
        x = self.linear1(combined_feature)  # (B, hidden_dim)
        x = F.relu(x)  
        x = self.linear2(x)  # (B, output_dim)
        
        shortcut = self.shortcut(combined_feature)  # (B, output_dim)    
        output = x + shortcut  # (B, output_dim)
        output = self.projection(output)
        output = self.norm(output)        
        # output = self.sigmoid(output)        
        
        return output
