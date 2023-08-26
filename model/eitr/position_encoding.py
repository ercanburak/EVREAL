import numpy as np
import torch
from torch import nn


class PositionalEncodingSine(nn.Module):

    def __init__(self, d_hid, n_position=10000):
        super().__init__()

        # Not a parameter
        self.pos_table = self._get_sinusoid_encoding_table(n_position, d_hid)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach().to(x.device)


def build_position_encoding(pos_type, d_model):
    if pos_type == 'sine':
        position_embedding = PositionalEncodingSine(d_model)
    else:
        raise ValueError(f'not support {pos_type}')

    return position_embedding
