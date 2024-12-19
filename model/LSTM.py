import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, feature_size, output_size, hidden_size=128, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        self.sigmod = nn.Sigmoid().to(device)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, is_multi=False):
        x = x.unsqueeze(1)
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        (hn, _) = self.lstm(x)
        x = hn[:,-1,:]
        # x = hn.squeeze(0)   # 移除 seq_len 维度，以匹配全连接层的输入期望
        out = self.fc(x)
        # return self.softmax(out) if is_multi else self.sigmod(out)
        return out