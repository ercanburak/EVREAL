import torch
from einops import rearrange
from torch import nn

from model.model_util import skip_sum
from model.submodules import ConvLayer, RecurrentConvLayer, UpsampleConvLayer
from .position_encoding import build_position_encoding
from .transformer_decoder import transformer_decoder
from .transformer_encoder import transformer_encoder


class mls_tpa(nn.Module):
    """
    """

    def __init__(self, num_bins, norm) -> None:
        super().__init__()
        self.head = ConvLayer(in_channels=num_bins, out_channels=32, kernel_size=5, stride=1, padding=2, norm=norm)
        self.DownsampleConv = nn.ModuleList([
            RecurrentConvLayer(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, norm=norm),
            RecurrentConvLayer(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, norm=norm),
            RecurrentConvLayer(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, norm=norm)
        ])

        self.position_embedding = build_position_encoding('sine', 256)
        self.split0 = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.trans_encoder0 = transformer_encoder(d_model=256, nhead=8, num_encoder_layers=3,
                                                  dim_feedforward=1024, activation='relu', dropout=0.1)
        self.trans_decoder0 = transformer_decoder(d_model=256, nhead=8, num_decoder_layers=2,
                                                  dim_feedforward=1024, activation='relu', dropout=0.1)
        self.split1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.trans_encoder1 = transformer_encoder(d_model=256, nhead=8, num_encoder_layers=3,
                                                  dim_feedforward=1024, activation='relu', dropout=0.1)
        self.trans_decoder1 = transformer_decoder(d_model=256, nhead=8, num_decoder_layers=2,
                                                  dim_feedforward=1024, activation='relu', dropout=0.1)
        self.split2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4, stride=4, padding=0)
        self.trans_encoder2 = transformer_encoder(d_model=256, nhead=8, num_encoder_layers=3,
                                                  dim_feedforward=1024, activation='relu', dropout=0.1)
        self.trans_decoder2 = transformer_decoder(d_model=256, nhead=8, num_decoder_layers=2,
                                                  dim_feedforward=1024, activation='relu', dropout=0.1)
        # self.split3 = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=8, stride=8, padding=0)
        # self.trans_encoder3 = transformer_encoder(d_model=256, nhead=8, num_encoder_layers=3, 
        #                                         dim_feedforward=1024, activation='relu', dropout=0.1)
        # self.trans_decoder3 = transformer_decoder(d_model=256, nhead=8, num_decoder_layers=2,
        #                                         dim_feedforward=1024, activation='relu', dropout=0.1)

        self.UpsampleConv = nn.ModuleList([
            UpsampleConvLayer(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, norm=norm)
        ])
        self.pred = ConvLayer(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, norm=norm,
                              activation=None)
        self.final_activation = torch.sigmoid
        self.skip_ftn = skip_sum
        self.states = [None] * 3

    def reset_states(self):
        self.states = [None] * 3

    def func(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # downsample conv
        blocks = []
        for i, body in enumerate(self.DownsampleConv):
            x, state = body(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # ************* path to transformer
        n, c, H, W = head.size()
        # ******** scale 0
        words0 = self.split0(blocks[-1]).transpose(1, 2)
        pos0 = self.position_embedding(words0)
        hs0 = self.trans_encoder0(src=words0.transpose(0, 1), pos=pos0.transpose(0, 1))
        # ******** scale 1
        words1 = self.split1(blocks[-2]).flatten(2).transpose(1, 2)
        pos1 = self.position_embedding(words1)
        hs1 = self.trans_encoder1(src=words1.transpose(0, 1), pos=pos1.transpose(0, 1))
        # ******** scale 2
        words2 = self.split2(blocks[-3]).flatten(2).transpose(1, 2)
        pos2 = self.position_embedding(words2)
        hs2 = self.trans_encoder2(src=words2.transpose(0, 1), pos=pos2.transpose(0, 1))
        # ******** scale 3
        # words3 = self.split3(head).flatten(2).transpose(1, 2)
        # pos3 = self.position_embedding(words3)
        # hs3 = self.trans_encoder3(src=words3.transpose(0, 1), pos=pos3.transpose(0, 1))

        hc0 = self.trans_decoder0(tgt=hs0, memory=hs0)
        hc1 = self.trans_decoder1(tgt=hs1, memory=hs0)
        hc2 = self.trans_decoder2(tgt=hs2, memory=hs1)
        # hc3 = self.trans_decoder3(tgt=hs3, memory=hs2)

        # hs_trans = (hs0 + hs1 + hs2 + hs3 + hc0 + hc1 + hc2 + hc3) / 8
        hs_trans = (hs0 + hs1 + hs2 + hc0 + hc1 + hc2) / 6
        hs = rearrange(hs_trans, '(h w) n c -> n c h w', h=H // 8)

        # decoder
        for i, body in enumerate(self.UpsampleConv):
            hs = body(self.skip_ftn(hs, blocks[3 - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(hs, head))
        if self.final_activation is not None:
            img = self.final_activation(img)  # sigmoid

        return img
