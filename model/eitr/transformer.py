# from typing import Optional
import copy

import torch.nn.functional as F
from torch import nn


class transformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, activation='relu',
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, d_model)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos, task_embed=None):
        tgt = memory = self.encoder(src, pos)
        output = self.decoder(tgt, memory, task_embed)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, d_model):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src, pos):
        output = src

        for layer in self.layers:
            output = layer(output, pos)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, d_model):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, tgt, memory, task_embed):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, task_embed)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.ffn_dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def with_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos):
        # self attention
        q = k = self.with_embed(src, pos)
        v = src
        src2 = self.self_attn(q, k, v)[0]
        src3 = src + self.attn_dropout(src2)
        src4 = self.norm1(src3)

        # FFN
        src5 = self.linear2(self.ffn_dropout1(self.activation(self.linear1(src4))))
        src6 = src4 + self.ffn_dropout2(src5)
        src7 = self.norm2(src6)

        return src7


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.sattn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cattn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.ffn_dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, task_embed):
        # self attention
        v = tgt
        q = k = self.with_embed(tgt, task_embed)
        tgt2 = self.self_attn(q, k, v)[0]
        tgt3 = tgt + self.sattn_dropout(tgt2)
        tgt4 = self.norm1(tgt3)

        # cross attention
        q = self.with_embed(tgt4, task_embed)
        k = v = memory
        tgt5 = self.cross_attn(q, k, v)[0]
        tgt6 = tgt4 + self.cattn_dropout(tgt5)
        tgt7 = self.norm2(tgt6)

        # FFN
        tgt8 = self.linear2(self.ffn_dropout1(self.activation(self.linear1(tgt7))))
        tgt9 = tgt7 + self.ffn_dropout2(tgt8)
        tgt10 = self.norm3(tgt9)

        return tgt10


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_transformer(args):
    return transformer(**args)
