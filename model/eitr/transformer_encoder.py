import copy

import torch.nn.functional as F
from torch import nn


class transformer_encoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, activation='relu',
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos):
        output = self.encoder(src, pos)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def with_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos):
        output = self.with_embed(src, pos)

        for layer in self.layers:
            output = layer(output)

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

    def forward(self, src):
        # self attention
        q = k = v = self.norm1(src)
        src1 = self.self_attn(q, k, v)[0]
        src2 = src + self.attn_dropout(src1)

        # FFN
        src3 = self.norm2(src2)
        src4 = self.linear2(self.ffn_dropout1(self.activation(self.linear1(src3))))
        src5 = src2 + self.ffn_dropout2(src4)

        return src5


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
