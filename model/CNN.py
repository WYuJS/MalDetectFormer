import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
class CNN(nn.Module):
    def __init__(self, feature_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1).to(device)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0).to(device)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1).to(device)
        
        # 计算经过两层卷积和池化层后的特征维度
        self.num_flatten = feature_size // 2 // 2 * 32
        
        self.fc1 = nn.Linear(self.num_flatten, 120).to(device)
        self.fc2 = nn.Linear(120, output_size).to(device)
        self.sigmod = nn.Sigmoid().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, x, is_multi=False):
        # 输入形状调整以包含通道维度 [batch_size, feature_size] -> [batch_size, 1, feature_size]
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # return (x) if is_multi else self.sigmod(x)
        return x
