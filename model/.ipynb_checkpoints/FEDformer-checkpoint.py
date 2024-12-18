import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from .layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from .layers.SelfAttention_Family import FullAttention, ProbAttention
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
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

class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len=4, label_len=48,  pred_len=1, L=3, base='legendre',
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,mode_select='random',
                dropout=0.2, embed='timeF', freq='h', activation='gelu',version='Fourier',modes=64, moving_avg=[12, 24],
                output_attention=False,cross_activation='tanh', d_ff=2048):
        super(FEDformer, self).__init__()
        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.output_size = c_out
        self.feature_size = enc_in

        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout).to(device)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout).to(device)

        if version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base).to(device)
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base).to(device)

            decoder_cross_att = MultiWaveletCross(in_channels=d_model,
                                                  out_channels=d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=modes,
                                                  ich=d_model,
                                                  base=base,
                                                  activation=cross_activation).to(device)

        else:
            encoder_self_att = FourierBlock(in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.seq_len,
                                            modes=modes,
                                            mode_select_method=mode_select).to(device)

            decoder_self_att = FourierBlock(in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=modes,
                                            mode_select_method=mode_select).to(device)

            decoder_cross_att = FourierCrossAttention(in_channels=d_model,
                                                      out_channels=d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=modes,
                                                      mode_select_method=mode_select).to(device)

        # Encoder
        enc_modes = int(min(modes, seq_len//2))
        dec_modes = int(min(modes, (seq_len//2+pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))
        self.sigmoid = nn.Sigmoid().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)


        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        d_model, n_heads),

                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
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
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,is_multi,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = Reshape(x_enc, self.pred_len, self.feature_size)
        x_dec = Reshape(x_dec, self.pred_len, self.feature_size)
        x_mark_enc = Reshape(x_mark_enc, self.pred_len, x_mark_enc.shape[1])
        x_mark_dec = Reshape(x_mark_dec, self.pred_len, x_mark_dec.shape[1])
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            x =  (dec_out[:, -self.pred_len:, :], attns)
        else:
            x =  dec_out[:, -self.pred_len:, :]  # [B, L, D]
        x = x.reshape(-1, self.output_size)[:batch_size,:]
        # return (x) if is_multi else self.sigmoid(x)
        return x

