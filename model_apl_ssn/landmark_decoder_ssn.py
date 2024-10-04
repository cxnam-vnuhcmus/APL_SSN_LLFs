import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkDecoder(nn.Module):
    def __init__(self, input_dim=(5,256), hidden_dim=512, output_dim=(131,2)):
        super(LandmarkDecoder, self).__init__()

        input_size = input_dim[0] * input_dim[1]
        output_size = output_dim[0] * output_dim[1]  
        self.lstm = nn.LSTM(input_size=input_dim[1], hidden_size=hidden_dim, num_layers=2, bias=True, batch_first=False, dropout=0.1, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            # nn.Sigmoid()
        )

    def forward(self, audio_feature, landmark_feature, llfs_feature=None):
        assert audio_feature.shape == landmark_feature.shape, "audio_feature and landmark_feature must have the same shape"
        
        combined_feature = audio_feature + landmark_feature  # (B, N, d)     
        if llfs_feature is not None:
            combined_feature = combined_feature + llfs_feature
        
        x, _ = self.lstm(combined_feature) # (B, N, d)        
        x = x[:, -1, :] # (B, d)        
        x = self.fc(x)  # (B, 132*2)
            
        out = x.reshape(x.shape[0],-1,2)
        return out
