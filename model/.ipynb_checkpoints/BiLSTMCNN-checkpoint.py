import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMCNN(nn.Module):
    def __init__(self, input_features, num_classes, hidden_size=128):
        super(BiLSTMCNN, self).__init__()
        self.bi_lstm = nn.LSTM(input_features, hidden_size, batch_first=True, bidirectional=True).to(device)
        self.conv = nn.Conv1d(in_channels=2*hidden_size, out_channels=64, kernel_size=3, stride=1, padding=1).to(device)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1).to(device)
        self.fc = nn.Linear(128, num_classes).to(device)  # Adjust depending on the output size after conv and pool
        self.sigmod = nn.Sigmoid().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, x, is_multi=False):
        # BiLSTM expects [batch, seq_len, feature], but seq_len=1 in our case
        x = x.unsqueeze(1)
        x, _ = self.bi_lstm(x)
        # Prepare for Conv1d [batch, channels, length]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x)
        # Flatten and pass through linear layer
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # return (x) if is_multi else self.sigmod(x)
        return x