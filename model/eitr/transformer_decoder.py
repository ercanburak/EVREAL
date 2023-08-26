import copy

import torch.nn.functional as F
from torch import nn


class transformer_decoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6, dim_feedforward=2048, activation='relu',
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory):
        output = self.decoder(tgt, memory)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, tgt, memory):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.sattn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cattn_dropout = nn.Dropout(dropout)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.ffn_dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory):
        # self attention
        q = k = v = self.norm1(tgt)
        tgt1 = self.self_attn(q, k, v)[0]
        tgt2 = tgt + self.sattn_dropout(tgt1)

        # cross attention
        q = self.norm21(tgt2)
        k = v = self.norm22(memory)
        tgt3 = self.cross_attn(q, k, v)[0]
        tgt4 = tgt2 + self.cattn_dropout(tgt3)

        # FFN
        tgt5 = self.norm3(tgt4)
        tgt6 = self.linear2(self.ffn_dropout1(self.activation(self.linear1(tgt5))))
        tgt7 = tgt4 + self.ffn_dropout2(tgt6)

        return tgt7


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
