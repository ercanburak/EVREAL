'''Some '''

import numpy as np
import torch
from math import sqrt, ceil, floor
from torch.nn import ReflectionPad2d, ZeroPad2d




def normalize_image(image, low=1, high=99):
    ''' For torch.tensor'''
    # images = torch.stack([item for item in sequence], dim=0)
    mini = np.percentile(torch.flatten(image.cpu()), low)
    maxi= np.percentile(torch.flatten(image.cpu()), high)
    image = (image - mini) / (maxi - mini + 1e-5)
    image = torch.clamp(image, 0, 1)
    return image

def optimal_crop_size(max_size, max_subsample_factor):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ReflectionPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)



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
            
        return ZeroPad2d((self.pad_width, 0, self.pad_height, 0))(image)

    def unpad(self, image):
        # --------------------------------------------------------------- #
        # Removes the padded rows & columns                               #
        # --------------------------------------------------------------- #
        return image[..., self.pad_height:, self.pad_width:]
