import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.nn import MultiheadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Embeddings(nn.Module):
    def __init__(self, d_model):
        '''d_model:词嵌入的维度（每个词转换成的大小）， vocab:词表的大小, word_len：字符型（进入nn.embedding）长度'''
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.numeric_transform = nn.Linear(in_features=1, out_features=d_model).float().to(device)

    def forward(self, x):
        x = (x.unsqueeze(-1)).to(device)  # 假设浮点型数据位于后部
        numeric_out = self.numeric_transform(x)
        return numeric_out * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(PositionEncoding, self).__init__()
        # 创建一个位置编码矩阵
        position = torch.arange(0, max_len).float().unsqueeze(1)  # 形状 (max_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos
        pe = pe.unsqueeze(0)  # 形状变为 (1, max_len, embedding_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入的嵌入上
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes,d_model=512, num_heads=4, num_encoder_layers=3, num_decoder_layers=3):
        super(TransformerModel, self).__init__()
        position = PositionEncoding(10000, d_model).to(device)
        self.embedding = nn.Sequential(Embeddings(d_model), position)
        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)
        self.fc_out1 = nn.Linear(d_model, 1)
        self.fc_out2 = nn.Linear(input_dim, num_classes)
        self.src_mask = None
        self.tgt_mask = None
        self.sigmoid = nn.Sigmoid().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, src, tgt, is_multi):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        out = self.transformer(src, tgt, src_mask=self.src_mask, tgt_mask=self.tgt_mask)
        out = self.fc_out1(out)
        out = out.squeeze()
        out = self.fc_out2(out)
        # return (out) if is_multi else self.sigmoid(out)
        return out