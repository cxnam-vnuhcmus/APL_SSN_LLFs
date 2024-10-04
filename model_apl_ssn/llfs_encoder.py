import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class LLFEncoder(nn.Module):
    def __init__(self, input_mfcc_dim=120, input_llf_dim=96, num_classes=11, pretrained=True):
        super(LLFEncoder, self).__init__()

        # Backbone: sử dụng ResNet18 đã được pre-trained
        self.resnet = models.resnet18(pretrained=pretrained)

        # Thay đổi đầu vào để phù hợp với chiều của MFCC
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.mfcc_fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Các lớp để xử lý LLFs
        self.llf_fc = nn.Sequential(
            nn.Linear(96*5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Lớp phân loại cảm xúc
        self.feat_out = nn.Linear(256 + 256, 256)
        self.emo_out = nn.Linear(256, num_classes)

    def forward(self, mfcc, llfs):
        # Chuyển đổi MFCC để phù hợp với định dạng đầu vào của ResNet
        mfcc = mfcc.unsqueeze(1)  # (B, 1, N, 120) -> thêm chiều kênh

        # Thông qua ResNet
        resnet_out = self.resnet(mfcc)  # (B, 1000)        
        resnet_out = self.mfcc_fc(resnet_out) # (B, 256)
        
        # Xử lý LLFs
        llfs = llfs.reshape(llfs.shape[0], -1)
        llf_out = self.llf_fc(llfs)  # (B, 256)
        
        # Kết hợp đầu ra từ ResNet và LLFs
        combined_features = torch.cat((resnet_out, llf_out), dim=1)  # (B, 256 + 256)

        # Đầu ra cảm xúc
        emotion_feature_output = self.feat_out(combined_features) # (B, 256)

        # Đầu ra đặc trưng cảm xúc
        emotion_output = self.emo_out(emotion_feature_output)  # (B, 11)

        return emotion_feature_output, emotion_output
