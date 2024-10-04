import torch
from torch import nn
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
from typing import Union


from .llfs_encoder import EmotionRecognitionModel

class Model(nn.Module):

    def __init__(self,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        
        self.emotion_model = EmotionRecognitionModel()
        self.emo_lb_ce_loss = nn.CrossEntropyLoss()
        self.emo_lv_ce_loss = nn.CrossEntropyLoss()
    
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def forward(self,
                audio,
                llfs,
                gt_emo
                ):
        emotion_feature_output, emotion_output = self.emotion_model(audio.to(self.device), llfs.to(self.device))
        
        loss = self.loss_fn(emotion_output, gt_emo)
        return (emotion_output), loss
        
    def loss_fn(self, y_pred, y):
        ce_loss_emotion = self.emo_lb_ce_loss(y_pred[:, :8], y[:, :8].argmax(dim=1))
        ce_loss_level = self.emo_lv_ce_loss(y_pred[:, 8:], y[:, 8:].argmax(dim=1))

        total_ce_loss = ce_loss_emotion + ce_loss_level
        
        return total_ce_loss

        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        audio, llfs, gt_emo = batch
        gt_emo = gt_emo.to(device)
        _, loss = self(
            audio = audio[:, 1:], 
            llfs = llfs[:, 1:],
            gt_emo = gt_emo
        )
        
        return loss

    def eval_step_imp(self, batch, device):
        with torch.no_grad():
            audio, llfs, gt_emo = batch
            gt_emo = gt_emo.to(device)
            pred_emo, loss = self(
                audio = audio[:, 1:], 
                llfs = llfs[:, 1:],
                gt_emo = gt_emo
            )
        return {"y_pred": pred_emo, "y": gt_emo}
        