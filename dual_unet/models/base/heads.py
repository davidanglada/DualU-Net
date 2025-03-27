from typing import Optional, Union
import torch
import torch.nn as nn
from .modules import Flatten, Activation


class SegmentationHead(nn.Sequential):
    """
    A standard segmentation head that consists of a single Conv2D layer with optional upsampling
    and an activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (e.g., number of segmentation classes).
        kernel_size (int): Size of the convolutional kernel. Defaults to 3.
        activation (Optional[Union[str, callable]]): Activation function name or callable.
            Can be None for no activation, or a string like 'relu', 'sigmoid', etc.
        upsampling (int): Upsampling factor. If > 1, uses nn.UpsamplingBilinear2d with the given
            scale factor. Defaults to 1 (no upsampling).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 1
    ) -> None:
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        act = Activation(activation)

        super().__init__(conv2d, up, act)


class CountHead(nn.Sequential):
    """
    A head for predicting a per-pixel count or density map, similar in structure to a segmentation head.
    It consists of a single Conv2D layer with optional upsampling and an activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (e.g., for a density or count map).
        kernel_size (int): Size of the convolutional kernel. Defaults to 3.
        activation (Optional[Union[str, callable]]): Activation function name or callable.
            Can be None for no activation, or a string like 'relu', 'sigmoid', etc.
        upsampling (int): Upsampling factor. If > 1, uses nn.UpsamplingBilinear2d with the given
            scale factor. Defaults to 1 (no upsampling).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 1
    ) -> None:
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        act = Activation(activation)

        super().__init__(conv2d, up, act)


class ClassificationHead(nn.Sequential):
    """
    A classification head typically used at the end of an encoder or a feature extractor.
    It pools the spatial dimensions, optionally applies dropout, and then uses a Linear layer.

    Args:
        in_channels (int): Number of input channels (the dimension of the features).
        classes (int): Number of output classes.
        pooling (str): Either 'avg' or 'max' for Adaptive pooling across spatial dimensions.
        dropout (float): Probability of dropout. If 0, no dropout is applied.
        activation (Optional[Union[str, callable]]): Activation after the linear layer,
            e.g., 'softmax', 'sigmoid', etc. If None, no activation is applied.
    """

    def __init__(
        self,
        in_channels: int,
        classes: int,
        pooling: str = "avg",
        dropout: float = 0.2,
        activation: Optional[Union[str, callable]] = None
    ) -> None:
        if pooling not in ("max", "avg"):
            raise ValueError(f"Pooling should be one of ('max', 'avg'), got {pooling}.")
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)

        flatten = Flatten()
        drop = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        act = Activation(activation)

        super().__init__(pool, flatten, drop, linear, act)
