import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from .utils import coords_grid, upflow8
from argparse import Namespace
# from utils.image_utils import ImagePadder
from ..utils.image_process import ImagePadder
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def get_args():
    # This is an adapter function that converts the arguments given in out config file to the format, which the ERAFT
    # expects.
    args = Namespace(small=False,
                     dropout=False,
                     mixed_precision=False,
                     clip=1.0)
    return args



class ERAFT(nn.Module):
    def __init__(self, num_bins):
        # args:
        super(ERAFT, self).__init__()
        args = get_args()
        self.args = args
        self.image_padder = ImagePadder(image_dim=None, min_size=32) #cfgs.image_dim
        self.subtype = 'standard' 

        assert (self.subtype == 'standard' or self.subtype == 'warm_start')

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        
        self.event_bins = num_bins

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                    n_first_channels=self.event_bins)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0,
                                    n_first_channels=self.event_bins)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def _load_net_from_checkpoint(self, ckpt_file):
        
        checkpoint = torch.load(ckpt_file, map_location=torch.device("cpu"))

        try:
            if "model" in checkpoint.keys():
                checkpoint = checkpoint.pop("model")
            elif 'model_state_dict' in checkpoint.keys():
                checkpoint = checkpoint.pop("model_state_dict")
            if "module." in list(checkpoint.keys())[0]:
                for key in list(checkpoint.keys()):
                    checkpoint.update({key[7:]:checkpoint.pop(key)})
        except:
            raise KeyError("'model' not in or mismatch state_dict.keys(), please check checkpoint path {}".format(checkpoint))
        
        
        pretrained_dict = {}
        for key, value in checkpoint.items():
            new_key = key
            if new_key in self.state_dict() and value.shape == self.state_dict()[new_key].shape:
                pretrained_dict[new_key] = value
                
        self.load_state_dict(pretrained_dict, strict=False)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """
        # Pad Image (for flawless up&downsampling)
        image1 = self.image_padder.pad(image1)
        image2 = self.image_padder.pad(image2)

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if (self.subtype == 'standard' or self.subtype == 'warm_start'):
                cnet = self.cnet(image2)
            else:
                raise Exception
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize Grids. First channel: x, 2nd channel: y. Image is just used to get the shape
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)
            flow_up = self.image_padder.unpad(flow_up)

        return dict(
                    flow_preds=flow_predictions,
                    flow_init=coords1 - coords0,
                    flow_final=flow_up,
                )
