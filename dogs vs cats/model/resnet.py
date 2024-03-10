import torch
from torchvision.models import resnet34,ResNet34_Weights
import torch.nn as nn

#采用迁移学习的模型
net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)


cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
net.fc = nn.Linear(512,2)
net.to(device)