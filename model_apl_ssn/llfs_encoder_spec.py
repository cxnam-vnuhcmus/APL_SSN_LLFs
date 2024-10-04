import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import matplotlib.cm as cm
import os

class LLFEncoder(nn.Module):
    def __init__(self, input_mfcc_dim=120, input_llf_dim=96, num_classes=11, pretrained=True):
        super(LLFEncoder, self).__init__()

        # Backbone: sử dụng ResNet18 đã được pre-trained
        self.resnet = models.resnet50(pretrained=pretrained)
        
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Replace the fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        self.last_fc = nn.Linear(256, num_classes)

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        

    def forward(self, mfcc, llfs):
        # Chuyển đổi MFCC để phù hợp với định dạng đầu vào của ResNet
        mfcc_features = F.interpolate(mfcc.unsqueeze(1), size=(64, 128), mode='bilinear', align_corners=False)   #[32, 1, 32, 64]
        llfs_features = F.interpolate(llfs.unsqueeze(1), size=(64, 128), mode='bilinear', align_corners=False)  # [32, 1, 32, 64]
        combined_features = torch.cat((mfcc_features, llfs_features), axis=2)
        
        tensor = combined_features
        tensor_min, tensor_max = tensor.min(), tensor.max()
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        tensor_np = normalized_tensor.squeeze(1).detach().cpu().numpy()  
        cmap = cm.get_cmap('magma')
        colored_images = []
        for i in range(tensor_np.shape[0]):
            colored_image = cmap(tensor_np[i])[:, :, :3]  # (H, W, 3)
            colored_image = np.moveaxis(colored_image, -1, 0)  # (3, H, W)
            colored_images.append(colored_image)
        output_tensor = torch.tensor(np.stack(colored_images)).to(combined_features.device, dtype=combined_features.dtype)
        output_tensor = F.interpolate(output_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        if not os.path.exists("test_img.png"):
            from PIL import Image
            tensor = output_tensor[0] * 255.0
            tensor = tensor.byte()
            image_numpy = tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
            image_pil = Image.fromarray(image_numpy)
            image_pil.save(f"test_img.png")
        
        resnet_feature_out = self.resnet(output_tensor)  # (B, 1000)      
        resnet_out = self.last_fc (resnet_feature_out)
        
        return resnet_feature_out, resnet_out
