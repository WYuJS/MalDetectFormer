import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from .layers.Embed import DataEmbedding
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 填充改变形状
def Reshape(data, seq_len, output_size):
    total_size_needed = ((data.shape[0] + 1) // seq_len + 1) * seq_len
    padding_needed = total_size_needed - data.shape[0]
    # 创建全0填充
    if data.shape[0] % seq_len != 0:
        padding = torch.zeros(padding_needed, output_size).to(device)
        # 将填充添加到数据末尾
        data = torch.cat((data, padding), dim=0)
    return data.reshape(-1, seq_len, output_size)

class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len=4, label_len=48,  out_len=1,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.2, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,):
        super(Informer, self).__init__()
        self.pred_len = seq_len
        self.feature_size = enc_in
        self.output_attention = output_attention
        self.output_size = c_out

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout).to(device)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout).to(device)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        ).to(device)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        ).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,is_multi,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        batch_size = x_enc.shape[0]
        x_enc = Reshape(x_enc, self.pred_len, self.feature_size)
        x_dec = Reshape(x_dec, self.pred_len, self.feature_size)
        x_mark_enc = Reshape(x_mark_enc, self.pred_len, x_mark_enc.shape[1])
        x_mark_dec = Reshape(x_mark_dec, self.pred_len, x_mark_dec.shape[1])

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            x =  (dec_out[:, -self.pred_len:, :], attns)
        else:
            x =  dec_out[:, -self.pred_len:, :]  # [B, L, D]
        x = x.reshape(-1, self.output_size)[:batch_size,:]
        # return (x) if is_multi else self.sigmoid(x)
        return x

