import torch.nn as nn
import torch.nn.functional as F
from model_config import dataset_path, learner_model_input_dims
import os
from torchvision import transforms
import cv2
import torch, torchvision

# print(dataset_path)
class LearnerModel(nn.Module):
    def __init__(self, input_dims=(3, 512, 512)) -> None:
        super().__init__()
        # expecting input dims to be a list or tuple in (C, H, W) mode

        self.nc = len(os.listdir(os.path.join(dataset_path, "train")))
        self.c, self.h, self.w = input_dims[0], input_dims[1], input_dims[2]
        self.conv = nn.Conv2d(in_channels=self.c, out_channels=128, kernel_size=5, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=238144, out_features=32)

        assert self.nc > 1, "Dataset found to train on 1 or 0 classes"
        print(self.nc)
        self.fc = nn.Linear(in_features=32, out_features=self.nc)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.fc(x)
        return F.softmax(x)


lm = LearnerModel(learner_model_input_dims)
compose_transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize(size=(learner_model_input_dims[1], learner_model_input_dims[2])), 
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
)


# img = torch.Tensor(cv2.imread("./dog.jpg"))
img = cv2.imread("./dog.jpg")
print(img.shape)
img = compose_transforms(img)
# print(img)
print(img.shape)
img = img.unsqueeze(dim=0)
preds = lm(img)
print(preds)
