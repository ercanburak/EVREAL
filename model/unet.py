import torch
import torch.nn as nn

from .model_util import skip_sum
from .submodules import \
    ConvLayer, UpsampleConvLayer, TransposedConvLayer, RecurrentConvLayer, ResidualBlock, DynamicUpsampleLayer


class BaseUNet(nn.Module):
    def __init__(self, base_num_channels, num_encoders, num_residual_blocks, num_output_channels, skip_type, norm,
                 use_upsample_conv, num_bins, recurrent_block_type=None, kernel_size=5, channel_multiplier=2,
                 use_dynamic_decoder=False):
        super().__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        self.use_dynamic_decoder = use_dynamic_decoder

        self.encoder_input_sizes = [int(self.base_num_channels * pow(channel_multiplier, i)) for i in
                                    range(self.num_encoders)]
        self.encoder_output_sizes = [int(self.base_num_channels * pow(channel_multiplier, i + 1)) for i in
                                     range(self.num_encoders)]
        self.max_num_channels = self.encoder_output_sizes[-1]

        self.skip_ftn = eval('skip_' + skip_type)

        if use_upsample_conv:
            self.UpsampleLayer = UpsampleConvLayer
        else:
            self.UpsampleLayer = TransposedConvLayer

        assert self.num_output_channels > 0

    def build_encoders(self):
        encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))
        return encoders

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):

            if idx == 0 and self.use_dynamic_decoder:
                decoders.append(DynamicUpsampleLayer(
                    2 * input_size if self.skip_type == 'concat' else input_size,
                    output_size, kernel_size=self.kernel_size, padding=self.kernel_size // 2,
                    in_fuse_channels=1 + self.num_bins))
            else:
                decoders.append(self.UpsampleLayer(
                    2 * input_size if self.skip_type == 'concat' else input_size,
                    output_size, kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2, norm=self.norm))

        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                         num_output_channels, 1, padding="same", activation=None, norm=norm)

    def build_head_layer(self):
        head = ConvLayer(self.num_bins, self.base_num_channels,
                         kernel_size=self.kernel_size, stride=1,
                         padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        return head


class UNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        if 'num_output_channels' not in unet_kwargs:
            unet_kwargs['num_output_channels'] = 1
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        super().__init__(**unet_kwargs)

        self.head = self.build_head_layer()
        self.encoders = self.build_encoders()
        self.build_resblocks()
        self.decoders = self.build_decoders()

        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = [None] * self.num_encoders

    def forward(self, x, prev_recs=None):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        # head
        ev_tensor = x
        x = self.head(x)

        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        for i, decoder in enumerate(self.decoders):
            skip_from_encoder = blocks[self.num_encoders - i - 1]
            if isinstance(decoder, DynamicUpsampleLayer):
                x = decoder(self.skip_ftn(x, skip_from_encoder), ev_tensor, prev_recs)
            else:
                x = decoder(self.skip_ftn(x, skip_from_encoder))

        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)

        if self.num_output_channels == 3:
            return {'image': img[:, 0:1, :, :], 'flow': img[:, 1:3, :, :]}
        elif self.num_output_channels == 1:
            return {'image': img}
