import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.nn import MultiheadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义Embeddings类来实现文本嵌入层，这里s说明两个一模一样的嵌入层共享参数
class Embeddings(nn.Module):
    def __init__(self, d_model, word_len,position, vocab=100000):
        '''d_model:词嵌入的维度（每个词转换成的大小）， vocab:词表的大小, word_len：字符型（进入nn.embedding）长度'''
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model).to(device)
        self.d_model = d_model
        # 浮点数数据的线性层，注意输入特征数在forward中动态确定
        self.numeric_transform = nn.Linear(in_features=1, out_features=d_model).float().to(device)
        self.position = word_len

    def forward(self, x):
        x_categorical = x[:, :self.position].long().to(device)  # 假设字符型数据位于前部，且需要转换为long类型以用于embedding
        x_numeric = (x[:, self.position:].unsqueeze(-1)).to(device)  # 假设浮点型数据位于后部
        x_numeric = x_numeric.to(device)
        categorical_out = self.lut(x_categorical)
        numeric_out = self.numeric_transform(x_numeric)
        out = torch.cat((categorical_out, numeric_out), dim=1).to(device)
        return (out * math.sqrt(self.d_model)).to(device)

# 定义词向量的位置编码
class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        '''d_model:词嵌入的维度，dropout:置0比率，max_len：每个句子最大长度'''
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout).to(device)

        # 初始化位置编码矩阵，矩阵大小 max_len × d_model
        pe = torch.zeros(max_len, d_model)
        # 初始化绝对位置矩阵，词汇的绝对位置是用索引去表示 矩阵大小max_len×1
        position = torch.arange(0, max_len).unsqueeze(1).to(device)

        #将绝对位置矩阵（position）融合到位置编码矩阵（pe）
        #将[max_len, 1]->[max_len, d_model]，因此需要变换矩阵div_term[1, d_model],同时，这个矩阵还需要将绝对位置编码缩放以便于梯度下降
        div_term = torch.exp(torch.arange(0, d_model, 2) * - (math.log(10000.0)/d_model)).to(device) #position是行，div_term是列
        pe[:, 0::2] = torch.sin(position * div_term)#偶数
        pe[:, 1::2] = torch.cos(position * div_term)#奇数

        # 将pe和embedding扩展到一个维度
        pe = pe.unsqueeze(0)

        #将pe注册成模型的buffer（认为对模型有帮助，但是不是模型的参数或超参数，不需要优化更新）
        self.register_buffer('pe', pe)

    def forward(self, x):
        #pe是设置成一个足够大的，但是实际使用时不会用这么多，截取需要的即可
        x = x + Variable(self.pe[:, x.size(1)], requires_grad=False)
        return self.dropout(x)


# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        '''d_model:第一个线性层维度，第二个线性层输入及第一个线性层输出'''
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff).to(device)
        self.w2 = nn.Linear(d_ff, d_model).to(device)
        self.dropout = nn.Dropout(p=dropout).to(device)

    def forward(self, x):
        return self.w2(self.dropout(self.w1(x))).to(device)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        '''features:输入的维度，eps:足够小的数（分母中出现）'''
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features)).to(device)
        self.b2 = nn.Parameter(torch.zeros(features)).to(device)
        self.eps = torch.tensor(eps).to(device)

    def forward(self, x):
        # print(x)
        print(type(x))
        mean = x.mean(-1, keepdim=True).to(device)
        std = x.std(-1, keepdim=True).to(device)
        return (self.a2 * (x - mean) / (std + self.eps) +self.b2).to(device)

'''
子层连接结构（残差连接）
'''
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0):
        '''size:词嵌入维度'''
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size).to(device)
        # self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout).to(device)

    def forward(self, x, sublayer):
        return x + self.dropout(self.norm(sublayer(x))).to(device)

def clones(module, N):
    return nn.ModuleList([(module) for _ in range(N)]).to(device)

def subsequent_mask(size):
    '''生成向后遮掩的掩码张量，size是掩码张量最后两个维度的大小，最后两维形成一个方阵'''
    attn_shape = (1, size, size) #定义掩码张量的形状

    # 形成上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape, k=1)).astype('uint8')

    # 形成下三角矩阵
    return torch.from_numpy(1 - subsequent_mask)

# 注意力机制
def attention(query, key, value, mask=None, dropout=None):
    # 取词嵌入维度（query最后一个维度大小）
    d_k = query.size(-1)
    scores = (torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)).to(device)

    # 判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的masked_fill方法，将掩码张量和scores张量每个位置一一比较，然后用极小值替换
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)

    # 判断是否使用dropout随机置0
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0):
        '''head:头数，embedding_dim:词嵌入维度
        多头注意力机制是将最后一维（特征）切割成多个
        '''
        super(MultiHeadAttention, self).__init__()
        # 给每个头分配等量的词特征，也就是embedding_dim/head个
        assert embedding_dim % head == 0

        self.d_k = embedding_dim // head
        self.head = head

        # 获取四个（Q，K，V以及最后拼接的）线性层
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4).to(device)

        self.attn = None # 保存注意力得分，当前没有所以是0
        self.dropout = nn.Dropout(p=dropout).to(device)

    def forward(self, query, key, value, mask=None, dropout=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 一个batch_size里有多少变量
        batch_size = query.size(0)

        # 对q,k,v进行linear变换
        # head*d_k是特征维度，transpose是未来让句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义和句子位置的关系
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears.to(device), (query, key, value))]

        x, self.attn = attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 用最后一个线性层contact
        return self.linears[-1](x)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0):
        '''size:词嵌入维度大小，self_attn：自注意力机制（对象），feed_forward:前馈神经网络（对象）'''
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn.to(device)
        self.feed_forward = feed_forward.to(device)

        self.sublayer = clones(SublayerConnection(size, dropout), 2).to(device)
        self.size = size

    def forward(self, x, mask=None):
        print('*'*64)
        print(type(x))
        x = self.sublayer[0](x, lambda x:self.self_attn(x, x, x)).to(device)
        
        return self.sublayer[1](x, self.feed_forward).to(device)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N).to(device)
        self.norm = LayerNorm(layer.size).to(device)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask).to(device)
        return self.norm(x).to(device)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0):
        '''size:词嵌入维度，self_attn:多头自注意力对象， src_attn:多头注意力对象（），feed_forward:前馈神经网络'''
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn.to(device)
        self.src_attn = src_attn.to(device)
        self.size = size
        self.feed_forward = feed_forward.to(device)
        self.sublayers = clones(SublayerConnection(size, dropout), 3).to(device)

    def forward(self, x, memory, source_mask=None, target_mask=None):
        '''memory:编码器的语义存储变量，source_mask:源数据掩码张量，target_maks:目标数据掩码张量'''
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x)).to(device)
        x = self.sublayers[1](x, lambda x: self.src_attn(x, memory, memory)).to(device)
        x = self.sublayers[2](x, self.feed_forward).to(device)
        return x

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N).to(device)
        self.norm = LayerNorm(layer.size).to(device)

    def forward(self, x, memory, source_mask=None, target_mask=None):
        '''x：目标数据的嵌入表示，memory：编码器的输出张量'''
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


# 输出
class Generator(nn.Module):
    def __init__(self, d_model, feature_size, output_size):
        '''d_model:词嵌入维度，feature_size:特征维度，outpout_size:输出维度'''
        super(Generator, self).__init__()
        self.flatten = nn.Flatten().to(device)
        self.project = nn.Linear(feature_size * d_model, output_size).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
    def forward(self, x):
        x = self.flatten(x)
        x = self.project(x)
        x = self.sigmoid(x)
        return x

# 模型构建
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        '''编码器对象、解码器对象、源数据嵌入函数、目标数据嵌入函数、输出部分对象'''
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.source_embed = source_embed.to(device)
        self.target_embed = target_embed.to(device)
        self.generator = generator.to(device)

    def forward(self, source, target, source_mask=None, target_mask=None):
        '''source:源数据，target：目标数据，source_mask和target_mask代表掩码张量'''
        return self.generator\
            (self.decode(self.encode(source, source_mask),
                           target, source_mask, target_mask)).to(device)

    def encode(self, source, source_mask=None):
        return self.encoder(self.source_embed(source), source_mask).to(device)

    def decode(self, memory, target, source_mask=None, target_mask=None):
        return self.decoder(self.target_embed(target), memory, source_mask, target_mask).to(device)

# Transformer实现
class Transformer(nn.Module):
    def __init__(self, source_vocab, target_vocab, output_size, position, N=6, d_model=512, d_ff=2048, head=8, dropout=0):
        '''source_vocab:源数据维度，target_vocab:目标数据特征维度，N:几层encoder/decoder
            d_model:词向量嵌入维度，d_ff:前馈全连接网络中变换矩阵的维度，head：注意力头的数量
            output_size: 输出维度
        '''
        super(Transformer, self).__init__()
        c = copy.deepcopy
        # attn = MultiHeadAttention(head, d_model, dropout).to(device)
        attn = MultiheadAttention(d_model, head).to(device)
        position = PositionEncoding(d_model, dropout).to(device)
        ff = FeedForward(d_model, d_ff, dropout).to(device)

        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, source_vocab, position), c(position)),
            nn.Sequential(Embeddings(d_model, target_vocab, position), c(position)),
            Generator(d_model, target_vocab, output_size)
        ).to(device)

    def forward(self, source, target, source_mask=None, target_mask=None):
        return self.model(source, target, source_mask, target_mask)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes,d_model=512, num_heads=4, num_encoder_layers=3, num_decoder_layers=3):
        super(TransformerModel, self).__init__()
        position = PositionEncoding(d_model, 0).to(device)
        self.embedding = nn.Sequential(Embeddings(d_model, 0, 10000), position)
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