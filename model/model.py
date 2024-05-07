import numpy as np
import torch
from torch import nn
from torchvision import transforms

from model.model_util import copy_states
from .submodules import ResidualBlock, ConvGRU, ConvLayer
# local modules
from .unet import UNetRecurrent
from utils.color_utils import merge_channels_into_color_image
from utils.util import CropParameters


class FlowNet(nn.Module):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetflow = UNetRecurrent(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetflow.states)

    @states.setter
    def states(self, states):
        self.unetflow.states = states

    def reset_states(self):
        self.unetflow.states = [None] * self.unetflow.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetflow.forward(event_tensor)
        return output_dict


class ColorNet(nn.Module):
    """
    Split the input events into RGBW channels and feed them to an existing
    recurrent model with states.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.channels = {'R': [slice(0, None, 2), slice(0, None, 2)],
                         'G': [slice(0, None, 2), slice(1, None, 2)],
                         'B': [slice(1, None, 2), slice(1, None, 2)],
                         'W': [slice(1, None, 2), slice(0, None, 2)],
                         'grayscale': [slice(None), slice(None)]}
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.prev_states = {k: self.model.states for k in self.channels}
        if hasattr(self.model, 'prev_recs'):
            self.prev_recs = {k: self.model.prev_recs for k in self.channels}
        else:
            self.prev_recs = {k: None for k in self.channels}


    @property
    def num_encoders(self):
        return self.model.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with RGB image taking values in [0, 1], and
                 displacement within event_tensor.
        """
        height, width = event_tensor.shape[-2:]
        crop_halfres = CropParameters(int(width / 2), int(height / 2), self.model.num_encoders)
        crop_fullres = CropParameters(width, height, self.model.num_encoders)
        color_events = {}
        reconstructions_for_each_channel = {}
        for channel, s in self.channels.items():
            color_events = event_tensor[:, :, s[0], s[1]]
            if channel == 'grayscale':
                color_events = crop_fullres.pad(color_events)
            else:
                color_events = crop_halfres.pad(color_events)
            self.model.states = self.prev_states[channel]
            self.model.prev_recs = self.prev_recs[channel]
            img = self.model(color_events)['image']
            self.prev_states[channel] = self.model.states
            self.prev_recs[channel] = self.model.prev_recs
            if channel == 'grayscale':
                img = crop_fullres.crop(img)
            else:
                img = crop_halfres.crop(img)
            img = img[0, 0, ...].cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            reconstructions_for_each_channel[channel] = img
        image_bgr = merge_channels_into_color_image(reconstructions_for_each_channel)  # H x W x 3
        image_bgr = transforms.functional.to_tensor(image_bgr)
        return {'image': image_bgr}


class E2VIDRecurrent(nn.Module):
    """
    Compatible with E2VID_lightweight
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrent(unet_kwargs)
        self.prev_recs = None

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders
        self.prev_recs = None

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        if self.prev_recs is None:
            self.prev_recs = torch.zeros(event_tensor.shape[0], 1, event_tensor.shape[2], event_tensor.shape[3],
                                         device=event_tensor.device)
        output_dict = self.unetrecurrent.forward(event_tensor, self.prev_recs)
        self.prev_recs = output_dict['image'].detach()
        return output_dict


class FireNet(nn.Module):
    """
    Refactored version of model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
    The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
    However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
    """

    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3):
        super().__init__()
        self.num_bins = num_bins
        padding = kernel_size // 2
        self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        self.num_recurrent_units = 2
        self.reset_states()

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H x W image
        """
        x = self.head(x)
        x = self.G1(x, self._states[0])
        self._states[0] = x
        x = self.R1(x)
        x = self.G2(x, self._states[1])
        self._states[1] = x
        x = self.R2(x)
        return {'image': self.pred(x)}



