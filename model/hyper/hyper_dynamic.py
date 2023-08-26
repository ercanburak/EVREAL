import torch
from torch import nn

from .fourier_bessel import bases_list


class ConvolutionalContextFusion(nn.Module):
    """ This module takes the event tensor and the previous reconstructions and fuses them together.
    First, these tensors are concatenated in the channel dimension. Then, the tensor is downsampled.
    Finally, a convolution is applied to the downsampled tensor.
    """

    def __init__(self, in_channels, out_channels, downsample_factor=4, kernel_size=3, padding="same"):
        super().__init__()
        self.scale = 1.0 / downsample_factor
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, ev_tensor, prev_recs):
        context = torch.cat((ev_tensor, prev_recs), dim=1)
        context = nn.functional.interpolate(context, scale_factor=self.scale, mode='bilinear', align_corners=False)
        context = self.conv(context)
        return context


class DynamicAtomGeneration(nn.Module):
    """ This module takes the context tensor and generates a set of dynamic atoms for each pixel.
    The context tensor is first passed through a convolutional network. The output of this network is a set of
    coefficients for the set of multiscale Fourier Bessel basis elements. These coefficients are then used to generate
    the dynamic atoms.
    """
    def __init__(self, kernel_size=3, num_atoms=6, num_bases=6, in_context_channels=32, hid_channels=64, stride=1):
        super().__init__()
        self.stride = stride
        self.num_atoms = num_atoms
        bases = bases_list(kernel_size, num_bases)  # This is the list of multiscale Fourier Bessel basis elements
        self.register_buffer('bases', torch.Tensor(bases).float())  # Tensor for the multiscale Fourier Bessel bases
        self.num_multiscale_bases = len(bases)  # This is the total number of multiscale Fourier Bessel basis elements
        num_basis_coeff = num_atoms * self.num_multiscale_bases  # This is the number of basis coefficients per pixel

        self.bases_net = nn.Sequential(
            nn.Conv2d(in_context_channels, hid_channels, kernel_size=3, padding="same", stride=stride),
            nn.BatchNorm2d(hid_channels),
            nn.Tanh(),
            nn.Conv2d(hid_channels, num_basis_coeff, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_basis_coeff),
            nn.Tanh()
        )

    def forward(self, context):
        N, _, H, W = context.shape
        H = H // self.stride
        W = W // self.stride
        basis_coefficients = self.bases_net(context)
        basis_coefficients = basis_coefficients.view(N, self.num_atoms, self.num_multiscale_bases, H, W)
        per_pixel_dynamic_atoms = torch.einsum('bmkhw,kl->bmlhw', basis_coefficients, self.bases)
        return per_pixel_dynamic_atoms


class DynamicConv(nn.Module):
    """
    This module takes an input tensor and convolves it with dynamic per-pixel kernels. The dynamic kernels are generated
    by multiplying the dynamic atoms with the learned compositional coefficients.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_atoms=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_atoms = num_atoms
        self.compositional_coefficients = nn.Parameter(torch.Tensor(out_channels, in_channels * num_atoms, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.compositional_coefficients, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input_tensor, per_pixel_dynamic_atoms):
        N, C, H, W = input_tensor.shape
        H = H // self.stride
        W = W // self.stride
        x = nn.functional.unfold(input_tensor, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x = x.view(N, self.in_channels, self.kernel_size * self.kernel_size, H, W)
        intermediate_features = torch.einsum('bmlhw,bclhw->bcmhw', per_pixel_dynamic_atoms, x)
        intermediate_features = intermediate_features.reshape(N, self.in_channels * self.num_atoms, H, W)
        out = nn.functional.conv2d(intermediate_features, self.compositional_coefficients, self.bias)
        return out
