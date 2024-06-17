import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def forward_interpolate_pytorch(flow_in):
    # Same as the numpy implementation, but differentiable :)
    # Flow: [B,2,H,W]
    flow = flow_in.clone()
    if len(flow.shape) < 4:
        flow = flow.unsqueeze(0)

    b, _, h, w = flow.shape
    device = flow.device

    dx ,dy = flow[:,0], flow[:,1]
    y0, x0 = torch.meshgrid(torch.arange(0, h, 1), torch.arange(0, w, 1))
    x0 = torch.stack([x0]*b).to(device)
    y0 = torch.stack([y0]*b).to(device)

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.flatten(start_dim=1)
    y1 = y1.flatten(start_dim=1)
    dx = dx.flatten(start_dim=1)
    dy = dy.flatten(start_dim=1)

    # Interpolate Griddata...
    # Note that a Nearest Neighbor Interpolation would be better. But there does not exist a pytorch fcn yet.
    # See issue: https://github.com/pytorch/pytorch/issues/50339
    flow_new = torch.zeros(flow.shape, device=device)
    for i in range(b):
        flow_new[i,0] = grid_sample_values(torch.stack([x1[i],y1[i],dx[i]]), h, w)[0]
        flow_new[i,1] = grid_sample_values(torch.stack([x1[i],y1[i],dy[i]]), h, w)[0]

    return flow_new


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """

    def __init__(self, W, H):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        self.gridX, self.gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        # if is_train:
        # self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        # self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        # else:
        #     self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        #     self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        # self.gridX = self.gridX.to(flow.device)
        # self.gridY = self.gridY.to(flow.device)
        gridX = torch.tensor(self.gridX, requires_grad=False, device=flow.device)
        gridY = torch.tensor(self.gridY, requires_grad=False, device=flow.device)
        
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = gridX.unsqueeze(0).expand_as(u).float() + u
        y = gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=True, padding_mode='reflection')
        return imgOut


class forwardWarp(nn.Module):
    """
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the forward warping
        block.
    """

    def __init__(self, W, H):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(forwardWarp, self).__init__()
        # create a grid
        self.gridX, self.gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H

    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the forward warping
        block.
        I1  = forwardwarp(I0, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I0.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        # self.gridX = self.gridX.to(flow.device)
        # self.gridY = self.gridY.to(flow.device)
        gridX = torch.tensor(self.gridX, requires_grad=False, device=flow.device)
        gridY = torch.tensor(self.gridY, requires_grad=False, device=flow.device)
        
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = gridX.unsqueeze(0).expand_as(u).float() - u
        y = gridY.unsqueeze(0).expand_as(v).float() - v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=True, padding_mode='reflection')
        return imgOut

class FrameWarp(object):
    def __init__(self, mode):
        '''
        Mode: forward or backward
        '''
        self.mode = mode
        self.flowWarp_dict = dict()
        
    def get_flowWarp_module(self, width: int, height: int):
        module = self.flowWarp_dict.get((width, height))
        if module is None:
            if self.mode == 'forward':
                module  = forwardWarp(width, height)
            else:
                module  = backWarp(width, height)
            self.flowWarp_dict[(width, height)] = module
        assert module is not None
        return module
    
    def warp_frame(self, I, flow):
        '''
        Forward warp: I0, F_0_1
        Backward warp: I1, F_1_0
        '''
        height, width  = I.shape[-2:]
        flow_warp = self.get_flowWarp_module(width, height)

        warped_I = flow_warp(I, flow)
        return warped_I