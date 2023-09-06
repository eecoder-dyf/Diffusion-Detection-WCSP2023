import torch
from torchvision import models
import torch.nn as nn 
import os
resnet18 = models.resnet18(pretrained=False)
num_feat = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_feat, 2)
resnet50 = models.resnet50(pretrained=False)
num_feat2 = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_feat2, 2)
