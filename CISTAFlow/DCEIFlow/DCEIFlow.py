import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append('..')
# sys.path.append('core')
# sys.path.append('utils')

from .core.decoder.with_event_updater import BasicUpdateBlockNoMask, SmallUpdateBlock
from .core.backbone.raft_encoder import BasicEncoder, SmallEncoder
from .core.corr.raft_corr import CorrBlock, AlternateCorrBlock
from .utils.sample_utils import coords_grid, upflow8, upflow4

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

# from ..utils.image_process import ImagePadder
from argparse import Namespace


class ImagePadder(object):
    # =================================================================== #
    # In some networks, the image gets downsized. This is a problem, if   #
    # the to-be-downsized image has odd dimensions ([15x20]->[7.5x10]).   #
    # To prevent this, the input image of the network needs to be a       #
    # multiple of a minimum size (min_size)                               #
    # The ImagePadder makes sure, that the input image is of such a size, #
    # and if not, it pads the image accordingly.                          #
    # =================================================================== #

    def __init__(self, image_dim=None, min_size=64):
        # --------------------------------------------------------------- #
        # The min_size additionally ensures, that the smallest image      #
        # does not get too small                                          #
        # --------------------------------------------------------------- #
        self.min_size = min_size
        if image_dim is None:
            self.pad_height = None
            self.pad_width = None
        else:
            self.height, self.width = image_dim
            if isinstance(min_size, (tuple, list)):
                self.pad_height = (min_size[0] - self.height % min_size[0])%min_size[0]
                self.pad_width = (min_size[1] - self.width % min_size[1])%min_size[1]
            else:
                self.pad_height = (min_size - self.height % min_size)%min_size
                self.pad_width = (min_size - self.width % min_size)%min_size
        

    def pad(self, image):
        # --------------------------------------------------------------- #
        # If necessary, this function pads the image on the left & top    #
        # --------------------------------------------------------------- #
        # height, width = image.shape[-2:]
        if self.pad_width is None:
            height, width = image.shape[-2:]
            self.pad_height = (self.min_size - height % self.min_size)%self.min_size
            self.pad_width = (self.min_size - width % self.min_size)%self.min_size
        # else:
        #     pad_height = (self.min_size - height % self.min_size)%self.min_size
        #     pad_width = (self.min_size - width % self.min_size)%self.min_size
        #     if pad_height != self.pad_height or pad_width != self.pad_width:
        #         raise
            
        return torch.nn.ZeroPad2d((self.pad_width, 0, self.pad_height, 0))(image)

    def unpad(self, image):
        # --------------------------------------------------------------- #
        # Removes the padded rows & columns                               #
        # --------------------------------------------------------------- #
        return image[..., self.pad_height:, self.pad_width:]



class EIFusion(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 192, 1, padding=0)
        self.conv2 = nn.Conv2d(input_dim, 192, 1, padding=0)
        self.convo = nn.Conv2d(192*2, input_dim, 3, padding=1)

    def forward(self, x1, x2):
        c1 = F.relu(self.conv1(x1))
        c2 = F.relu(self.conv2(x2))
        out = torch.cat([c1, c2], dim=1)
        out = F.relu(self.convo(out))
        return out + x1


# class Args:
#     def __init__(self):
#         pass

def get_args():
    # This is an adapter function that converts the arguments given in out config file to the format, which the ERAFT
    # expects.
    args = Namespace(corr_levels = 4,
                     corr_radius = 3,
                     mixed_precision=False,
                     )
    return args

class DCEIFlow(nn.Module):
    def __init__(self, num_bins): #args
        super().__init__()
        
        self.image_padder = ImagePadder(image_dim=None, min_size=32) #args.image_dim
        self.ds = 8 #args.ds
        self.is_bi = False 
        self.args = get_args()
        self.small = False
        self.dropout = 0
        self.alternate_corr = False
        

        self.selected_upflow_function = globals().get(f"upflow{self.ds}")

        
        self.event_bins = num_bins #args.event_bins if args.no_event_polarity is True else 2 * args.event_bins

        if self.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.args.corr_levels = 4
            self.args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.args.corr_levels = 4
            self.args.corr_radius = 4
        self.args.mixed_precision = False
        # feature network, context network, and update block
        if self.small:
            self.fnet = SmallEncoder(input_dim=3, output_dim=128, norm_fn='instance', dropout=self.dropout)
            self.cnet = SmallEncoder(input_dim=3, output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
            self.fusion = EIFusion(input_dim=128)
            self.enet = SmallEncoder(input_dim=self.event_bins, output_dim=128, norm_fn='instance', dropout=self.dropout)
        else:
            self.fnet = BasicEncoder(ds=self.ds, input_dim=1, output_dim=256, norm_fn='instance', dropout=self.dropout) 
            self.cnet = BasicEncoder(ds=self.ds, input_dim=1, output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout)
            self.update_block = BasicUpdateBlockNoMask(self.args, hidden_dim=hdim)
            self.fusion = EIFusion(input_dim=256)
            self.enet = BasicEncoder(ds=self.ds, input_dim=self.event_bins, output_dim=256, norm_fn='instance', dropout=self.dropout)
    
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        # print(H, W, H//8, W//8)
        coords0 = coords_grid(N, H//self.ds, W//self.ds).to(img.device)
        coords1 = coords_grid(N, H//self.ds, W//self.ds).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.ds, self.ds, H, W) # self.ds?
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.ds * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, self.ds*H, self.ds*W)

     
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

        
    def forward(self, event_voxel, image1, image2=None, reversed_event_voxel=None, iters=6, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """

        
        image1 = 2 * image1 - 1.0
        image1 = self.image_padder.pad(image1)
        image1 = image1.contiguous()
        

        # image2 = None
        if image2 is not None: #self.training or self.isbi:
            image2 = 2 * image2 - 1.0
            image2 = self.image_padder.pad(image2)
            image2 = image2.contiguous()

        
        event_voxel = self.image_padder.pad(event_voxel)
        event_voxel = event_voxel.contiguous()
        
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        reversed_emap = None
        with autocast(enabled=self.args.mixed_precision):
            emap = self.enet(event_voxel)
            if image2 is not None: #self.isbi and 'reversed_event_voxel' in batch.keys():
                assert image2 is not None
                fmap1, fmap2 = self.fnet([image1, image2])
                if reversed_event_voxel is not None:
                    reversed_event_voxel = self.image_padder.pad(reversed_event_voxel).contiguous()
                    reversed_emap = self.enet(reversed_event_voxel)
            else:
                reversed_emap = None
                if image2 is None:
                    fmap1 = self.fnet(image1)
                    fmap2 = None
                else:
                    fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        emap = emap.float()
        if fmap2 is not None:
            fmap2 = fmap2.float()

        with autocast(enabled=self.args.mixed_precision):
            pseudo_fmap2 = self.fusion(fmap1, emap)
        
        corr_fn = CorrBlock(fmap1, pseudo_fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        flow_predictions_bw = []
        flow_up = None
        flow_up_bw = None
        pseudo_fmap1 = None

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, emap, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = self.selected_upflow_function(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)
            flow_up = self.image_padder.unpad(flow_up)
            

        if fmap2 is not None and reversed_emap is not None:

            with autocast(enabled=self.args.mixed_precision):
                # pseudo_fmap1 = fmap2 + r_emap
                pseudo_fmap1 = self.fusion(fmap2, reversed_emap)

            if self.alternate_corr:
                corr_fn = AlternateCorrBlock(fmap2, pseudo_fmap1, radius=self.args.corr_radius)
            else:
                corr_fn = CorrBlock(fmap2, pseudo_fmap1, radius=self.args.corr_radius)

            # run the context network
            with autocast(enabled=self.args.mixed_precision):
                cnet = self.cnet(image2)
                net, inp = torch.split(cnet, [hdim, cdim], dim=1)
                net = torch.tanh(net)
                inp = torch.relu(inp)

            coords0, coords1 = self.initialize_flow(image2)

            if flow_init is not None:
                coords1 = coords1 + flow_init

            for itr in range(iters):
                coords1 = coords1.detach()
                corr = corr_fn(coords1) # index correlation volume

                flow = coords1 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    net, up_mask, delta_flow = self.update_block(net, inp, corr, reversed_emap, flow)

                # F(t+1) = F(t) + \Delta(t)
                coords1 = coords1 + delta_flow

                # upsample predictions
                if up_mask is None:
                    flow_up_bw = self.selected_upflow_function(coords1 - coords0)
                else:
                    flow_up_bw = self.upsample_flow(coords1 - coords0, up_mask)
                
                flow_predictions_bw.append(flow_up_bw)

        # return coords1 - coords0, flow_predictions, flow_up, flow_predictions_bw, flow_up_bw
        if self.is_bi:
            batch = dict(
                flow_preds=flow_predictions,
                flow_preds_bw=flow_predictions_bw,
                flow_init=coords1 - coords0,
                flow_final=flow_up,
                flow_final_bw=flow_up_bw,
                fmap2_gt=fmap2,
                fmap2_pseudo=pseudo_fmap2,
                fmap1_gt=fmap1,
                fmap1_pseudo=pseudo_fmap1,
            )
        else:
            if image2 is not None:
                batch = dict(
                    flow_preds=flow_predictions,
                    flow_init=coords1 - coords0,
                    flow_final=flow_up,
                    fmap2_gt=fmap2,
                    fmap2_pseudo=pseudo_fmap2,
                )
            else:
                batch = dict(
                    flow_preds=flow_predictions,
                    flow_init=coords1 - coords0,
                    flow_final=flow_up,
                )
        return batch
