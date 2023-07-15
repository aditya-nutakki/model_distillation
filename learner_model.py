import torch.nn as nn
import torch.nn.functional as F
from model_config import input_dims, train_path
import os
from torchvision import transforms
import cv2
import torch, torchvision
import model_config as c

# print(dataset_path)
class LearnerModel(nn.Module):
    def __init__(self, input_dims=c.input_dims) -> None:
        super().__init__()
        # expecting input dims to be a list or tuple in (C, H, W) mode

        self.nc = len(os.listdir(os.path.join(train_path)))
        assert self.nc > 1, "Dataset found on train with 1 or 0 classes"
        
        self.c, self.h, self.w = input_dims[0], input_dims[1], input_dims[2]

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=self.c, out_channels=128, kernel_size=5, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5),
            nn.Flatten()
        )
        self.seq_shape = self.get_shape(self.seq)[1]    
        self.linear1 = nn.Linear(in_features=self.seq_shape, out_features=32)
        self.fc = nn.Linear(in_features=32, out_features=self.nc)


    def get_shape(self, module):
        return module(torch.randn(c.input_dims, requires_grad=False).unsqueeze(0)).shape


    def forward(self, x):
        x = self.seq(x)
        # print(f"shape after flattening => {x.shape}")
        x = self.linear1(x)
        x = self.fc(x)
        # return F.softmax(x)
        return x
