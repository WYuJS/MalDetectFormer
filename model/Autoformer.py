# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

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

class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, enc_in, dec_in, c_out, output_size, seq_len=4, label_len=2,  pred_len=4, L=3, base='legendre',
                d_model=512, n_heads=8, e_layers=3, d_layers=2,mode_select='random',factor=1,
                dropout=0.2, embed='timeF', freq='h', activation='gelu',version='Fourier',modes=64, moving_avg=12,
                output_attention=True,cross_activation='tanh', d_ff=2048):
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.output_size = output_size
        self.feature_size = enc_in
        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp(kernel_size[0])
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout).to(device)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout).to(device)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        ).to(device)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        ).to(device)
        self.generator = nn.Linear(c_out, output_size).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,is_multi,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        batch_size = x_enc.shape[0]
        x_enc = Reshape(x_enc, self.pred_len, self.feature_size)
        x_dec = Reshape(x_dec, self.pred_len, self.feature_size)
        x_mark_enc = Reshape(x_mark_enc, self.pred_len, x_mark_enc.shape[1])
        x_mark_dec = Reshape(x_mark_dec, self.pred_len, x_mark_dec.shape[1])
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        dec_out = self.generator(dec_out)
        if self.output_attention:
            x =  (dec_out[:, -self.pred_len:, :], attns)
        else:
            x =  dec_out[:, -self.pred_len:, :]  # [B, L, D]
        x = x[0].reshape(-1, self.output_size)[:batch_size,:]
        torch.save(attns, 'autoformer_attn.pt')
        # return (x) if is_multi else self.sigmoid(x)
        return x
