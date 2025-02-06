# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

import torch.nn as nn
from .modules import Flatten, Activation


class SegmentationHead(nn.Sequential):
    """
    A simple segmentation head: Conv2D -> (optional) Upsampling -> Activation (if any).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = None,
        upsampling: int = 1,
    ):
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        upsample = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        act = Activation(activation)

        super().__init__(conv2d, upsample, act)


class CountHead(nn.Sequential):
    """
    A counting/density head: Conv2D -> (optional) Upsampling -> Activation (if any).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = None,
        upsampling: int = 1,
    ):
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        upsample = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        act = Activation(activation)

        super().__init__(conv2d, upsample, act)
