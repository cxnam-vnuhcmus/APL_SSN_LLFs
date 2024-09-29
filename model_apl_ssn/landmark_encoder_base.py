
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, N, d_model)
        N = x.size(1)
        return x + self.pe[:, :N, :].to(x.device)
    
class LandmarkEncoder(nn.Module):
    def __init__(self, input_dim=(131,2), hidden_dim=256, output_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super(LandmarkEncoder, self).__init__()
        
        input_size = input_dim[0] * input_dim[1]
        
        self.fc_in = nn.Linear(input_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
                
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.fc_in(x)  # (B, N, d)
        x = self.pos_encoder(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # (B, N, d)
        
        return x