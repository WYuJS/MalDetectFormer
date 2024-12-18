import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import dgl
import psutil
from model.EGCN import EGCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 区间
def intervalPadding(data, batch_size, interval_size, embed_size, num_intervals):
    total_size_needed = ((batch_size + num_intervals - 1) // num_intervals) * num_intervals
    padding_needed = total_size_needed - batch_size
    # 创建全0填充
    if padding_needed > 0:
        padding = torch.zeros(padding_needed, interval_size, embed_size).to(device)
        # 将填充添加到数据末尾
        padded_data = torch.cat((data, padding), dim=0)
    else:
        padded_data = data
    return padded_data


# 定义Embeddings类来实现文本嵌入层，这里s说明两个一模一样的嵌入层共享参数
class Embeddings(nn.Module):
    def __init__(self, d_model):
        '''d_model:词嵌入的维度（每个词转换成的大小）， vocab:词表的大小, word_len：字符型（进入nn.embedding）长度'''
        super(Embeddings, self).__init__()
        # self.lut = nn.Embedding(vocab, d_model).to(device)
        self.d_model = d_model
        # 浮点数数据的线性层，注意输入特征数在forward中动态确定
        self.numeric_transform = nn.Linear(in_features=1, out_features=d_model).float().to(device)

    def forward(self, x):
        # x_categorical = x[:, :self.position].long().to(device)  # 假设字符型数据位于前部，且需要转换为long类型以用于embedding
        x_numeric = (x.unsqueeze(-1)).to(device)  # 假设浮点型数据位于后部
        # categorical_out = self.lut(x_categorical)
        numeric_out = self.numeric_transform(x_numeric)
        # out = torch.cat((categorical_out, numeric_out), dim=1)
        return numeric_out * math.sqrt(self.d_model)


# 定义词向量的位置编码
class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500000):
        '''d_model:词嵌入的维度，dropout:置0比率，max_len：每个句子最大长度'''
        super(PositionEncoding, self).__init__()
        self.d_model = d_model


    def forward(self, x, interval):
        max_len = x.size(1)
        position = torch.arange(0, max_len).unsqueeze(1).to(device)
        pe = torch.zeros(max_len, self.d_model).to(device)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1).to(device)
        i = torch.arange(self.d_model, dtype=torch.float).to(device)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / self.d_model).to(device)
        angle_rads = ((pos / interval) + angle_rates).to(device)
        pe[:, 0::2] = torch.sin(position * angle_rads[:, 0::2])  # Apply sin to even indices (2i)
        pe[:, 1::2] = torch.cos(position * angle_rads[:, 1::2]) 
        pe = pe.unsqueeze(0).to(device)
        # pe是设置成一个足够大的，但是实际使用时不会用这么多，截取需要的即可
        x = x + Variable(pe, requires_grad=False)
        del pe
        return (x)


# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        '''d_model:第一个线性层维度，第二个线性层输入及第一个线性层输出'''
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff).to(device)
        self.w2 = nn.Linear(d_ff, d_model).to(device)
        self.dropout = nn.Dropout(p=dropout).to(device)

    def forward(self, x):
        return self.w2(self.dropout(self.w1(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        '''features:输入的维度，eps:足够小的数（分母中出现）'''
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features)).to(device)
        self.b2 = nn.Parameter(torch.zeros(features)).to(device)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


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
        return x + self.dropout(self.norm(sublayer(x)))


def clones(module, N):
    # return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    return nn.ModuleList([(module) for _ in range(N)])


# 注意力机制
def attention(query, key, value, dropout=None):
    # 取词嵌入维度（query最后一个维度大小）
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = torch.softmax(scores, dim=-1)

    # 判断是否使用dropout随机置0
    if dropout is not None:
        p_attn = F.dropout(p_attn, p=dropout, training=True)

    return torch.matmul(p_attn, value), p_attn


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, attn_dropout=0):
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

        self.attn = None  # 保存注意力得分，当前没有所以是0

    def forward(self, query, key, value, p):
        'p：丢失率'

        # 一个batch_size里有多少变量
        batch_size = query.size(0)

        # 对q,k,v进行linear变换
        # head*d_k是特征维度，transpose是未来让句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义和句子位置的关系
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, p)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 用最后一个线性层contact
        return self.linears[-1](x)


class IntervalAttention(nn.Module):
    def __init__(self, embed_size, interval_size, heads):
        super(IntervalAttention, self).__init__()
        self.interval_size = interval_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert embed_size % heads == 0, "Embedding size must be divisible by number of heads."

        self.query_linear = nn.Linear(embed_size, embed_size ).to(device)
        self.key_linear = nn.Linear(embed_size , embed_size ).to(device)
        self.value_linear = nn.Linear(embed_size, embed_size ).to(device)
        self.fc_out = nn.Linear(embed_size, embed_size).to(device)

    def forward(self, q, k, v, num_intervals):
        batch_size, interval_size, embed_size = q.shape

        # 区间注意力填充
        q = intervalPadding(q, batch_size, interval_size, embed_size, num_intervals)
        k = intervalPadding(k, batch_size, interval_size, embed_size, num_intervals)
        v = intervalPadding(v, batch_size, interval_size, embed_size, num_intervals)

        # 转换维度(区间注意力)
        q = q.reshape(-1, num_intervals, interval_size, embed_size)
        k = k.reshape(-1, num_intervals, interval_size, embed_size)
        v = v.reshape(-1, num_intervals, interval_size, embed_size)

        # 合并区间内的数据
        # q = q.view(q.shape[0], num_intervals, -1)
        # k = k.view(q.shape[0], num_intervals, -1)
        # v = v.view(q.shape[0], num_intervals, -1)
        q = self.query_linear(q)
        k = self.key_linear(k)
        v = self.value_linear(v)

        q = q.view(q.shape[0], -1)
        k = k.view(q.shape[0], -1)
        v = v.view(v.shape[0], -1)

        queries = q
        keys = k
        values = v

        # 计算注意力分数
        # attention_scores = torch.einsum("bqd,bkd->bqk", queries, keys)
        attention_scores = torch.matmul(queries, keys.t())
        attention = torch.softmax(attention_scores / (self.head_dim ** 0.5), dim=1)
        # 应用注意力分数
        # out = torch.einsum("bql,bld->bqd", attention, values)
        out = torch.matmul(attention, values)
        out = out.reshape(q.shape[0], num_intervals * interval_size, embed_size)
        out = self.fc_out(out)
        return (out.reshape(q.shape[0] * num_intervals, interval_size, embed_size))[:batch_size, :, :]


class NetSentinelAttention(nn.Module):
    def __init__(self, head, embedding_dim, interval_size):
        super(NetSentinelAttention, self).__init__()
        self.multiAttention = MultiHeadAttention(head, embedding_dim).to(device)
        self.IntervalAttention = IntervalAttention(embedding_dim, interval_size, head).to(device)

    def forward(self, q, k, v, num_intervals, p):
        # 分别注意力机制
        out1 = self.IntervalAttention(q, k, v, num_intervals)
        out2 = self.multiAttention(q, k, v, p)

        out = out1 + out2
        return out


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0):
        '''size:词嵌入维度大小，self_attn：自注意力机制（对象），feed_forward:前馈神经网络（对象）'''
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn.to(device)
        self.feed_forward = feed_forward.to(device)
        self.sublayer = clones(SublayerConnection(size, dropout), 2).to(device)
        self.size = size
        self.norm = LayerNorm(size).to(device)
        # self.norm = nn.LayerNorm(size).to(device)
        self.dropout = nn.Dropout(p=dropout).to(device)

    def forward(self, x, num_intervals, EGCNPredictLayer, p, raw_x):
        # 注意力部分
        attn_out = self.self_attn(x, x, x, num_intervals, p)
        # x = self.sublayer[0](x,  lambda x: self.self_attn(x, x, x, num_intervals))
        x = x + self.dropout(self.norm(attn_out))

        out1 = self.sublayer[1](x, self.feed_forward)

        # egcn部分
        out2 = EGCNPredictLayer(raw_x, attn_out)

        out = out1 + out2
        return out


class Encoder(nn.Module):
    def __init__(self, layer, N, EGCNPredictLayer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N).to(device)
        self.norm = LayerNorm(layer.size).to(device)
        self.EGCNPredictLayer = EGCNPredictLayer

        
    def forward(self, x, num_intervals, p, raw_x):
        for layer in self.layers:
            x = layer(x, num_intervals, self.EGCNPredictLayer, p, raw_x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0):
        '''size:词嵌入维度，self_attn:多头自注意力对象， src_attn:多头注意力对象（），feed_forward:前馈神经网络'''
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn.to(device)
        self.src_attn = src_attn.to(device)
        self.size = size
        self.feed_forward = feed_forward.to(device)
        self.sublayers = clones(SublayerConnection(size, dropout), 3).to(device)

    def forward(self, x, memory, num_intervals, p):
        '''memory:编码器的语义存储变量'''
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, num_intervals, p))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, memory, memory, num_intervals, p))
        x = self.sublayers[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N).to(device)
        self.norm = LayerNorm(layer.size).to(device)

    def forward(self, x, memory, num_intervals, p):
        '''x：目标数据的嵌入表示，memory：编码器的输出张量'''
        for layer in self.layers:
            x = layer(x, memory, num_intervals, p)
        return self.norm(x)


# 输出
class Generator(nn.Module):
    def __init__(self, d_model, feature_size, output_size):
        '''d_model:词嵌入维度，feature_size:特征维度，outpout_size:输出维度'''
        super(Generator, self).__init__()
        self.flatten = nn.Flatten().to(device)
        self.project = nn.Linear(feature_size * d_model, output_size).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, x, is_multi):
        x = self.flatten(x)
        x = self.project(x)
        # return (x) if is_multi else self.sigmoid(x)
        return x


# 模型构建
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, src_pos_emdedding, tgt_embed, tgt_pos_embedding, generator):
        '''编码器对象、解码器对象、源数据嵌入函数、目标数据嵌入函数、输出部分对象'''
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.src_embed = src_embed.to(device)
        self.src_pos_emdedding = src_pos_emdedding
        self.tgt_pos_embedding = tgt_pos_embedding
        self.tgt_embed = tgt_embed.to(device)
        self.generator = generator.to(device)

    def forward(self, source, target, num_intervals, p, is_multi=False):
        '''source:源数据，target：目标数据'''
        return self.generator \
            (self.decode(self.encode(source, num_intervals, p),
                         target, num_intervals, p), is_multi)

    def encode(self, source, num_intervals, p):
        source_embed = self.src_embed(source)
        source_pos_embed = self.src_pos_emdedding(source_embed, num_intervals)
        return self.encoder(source_pos_embed, num_intervals, p, source)

    def decode(self, memory, target, num_intervals, p):
        tgt_embed = self.tgt_embed(target)
        tgt_pos_embed = self.tgt_pos_embedding(tgt_embed, num_intervals)
        return self.decoder(tgt_pos_embed, memory, num_intervals, p)


class EGCNPredictLayer(nn.Module):
    def __init__(self, g, G, d_model, source_vocab, egcn_hidden_size, ip_node_name_to_index, net_node_name_to_index):
        super(EGCNPredictLayer, self).__init__()
        self.FeatureAlignment = nn.Linear(1, d_model)
        self.G = G
        self.g = g
        # self.model = model
        self.model = EGCN(source_vocab, egcn_hidden_size)
        self.ip_node_name_to_index = ip_node_name_to_index
        self.net_node_name_to_index = net_node_name_to_index

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_data):
        x = self.model(self.g, self.G, x[:, 0], x[:, 1], x[:, 2], x[:, 3], \
                          self.ip_node_name_to_index, self.net_node_name_to_index)
        x = x.unsqueeze(-1)
        x = self.FeatureAlignment(x)
        x = x + attn_data

        return self.norm(x)
        # return attn_data

# Transformer实现
class MalDetectFormer(nn.Module):
    def __init__(self, source_vocab,
                 target_vocab,
                 output_size,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_model=512,
                 d_ff=2048,
                 head=8,
                 dropout=0,
                 egcn_hidden_size=64,
                 g=None,
                 G=None,
                 g_node_name_to_index=None,
                 G_node_name_to_index=None):
        '''source_vocab:源数据维度，target_vocab:目标数据特征维度，N:几层encoder/decoder
            d_model:词向量嵌入维度，d_ff:前馈全连接网络中变换矩阵的维度，head：注意力头的数量
            output_size: 输出维度，max_vocab:词表最大长度, G是EGCN模型
        '''
        super(MalDetectFormer, self).__init__()
        # attn = MultiHeadAttention(head, d_model, dropout)
        attn_encode = NetSentinelAttention(head, d_model, source_vocab).to(device)
        attn_decode1 = NetSentinelAttention(head, d_model, source_vocab).to(device)
        attn_decode2 = NetSentinelAttention(head, d_model, source_vocab).to(device)
        position_encode = PositionEncoding(d_model, dropout).to(device)
        position_decode = PositionEncoding(d_model, dropout).to(device)
        ff_encode = FeedForward(d_model, d_ff, dropout).to(device)
        ff_decode = FeedForward(d_model, d_ff, dropout).to(device)
        eGCNPredictLayer = EGCNPredictLayer(g, G, d_model, source_vocab, egcn_hidden_size, g_node_name_to_index, G_node_name_to_index)

        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, attn_encode, ff_encode, dropout), num_encoder_layers, eGCNPredictLayer),
            Decoder(DecoderLayer(d_model, attn_decode1, attn_decode2, ff_decode, dropout), num_decoder_layers),
            Embeddings(d_model), position_encode,
            Embeddings(d_model), position_decode,
            Generator(d_model, target_vocab, output_size)
        )

    def forward(self, source, target, num_intervals, p, is_multi=False):
        return self.model(source, target, num_intervals, p, is_multi).float()
