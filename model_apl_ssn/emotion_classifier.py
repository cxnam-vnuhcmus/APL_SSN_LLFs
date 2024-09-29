import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=(5,256), output_dim=11):
        super(EmotionClassifier, self).__init__()
        input_size = input_dim[0] * input_dim[1]
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),    # Increased complexity with more neurons
            nn.BatchNorm1d(128),          # Batch normalization for better training stability
            nn.ReLU(),
            nn.Dropout(0.3),              # Dropout for regularization
            nn.Linear(128, 64),           # More layers
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)
