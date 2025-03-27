import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union


try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    """
    A block consisting of a Conv2D layer, optional BatchNorm, and a ReLU activation.
    Supports both standard BatchNorm2d and InPlaceABN (inplace activated batch norm),
    if installed.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the convolution.
        padding (int): Padding for the convolution. Defaults to 0.
        stride (int): Stride for the convolution. Defaults to 1.
        use_batchnorm (bool or str): If True, use BatchNorm2d. If "inplace", use InPlaceABN.
                                     If False, do not use batch normalization.
                                     Requires inplace_abn to be installed for "inplace".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_batchnorm: bool = True,
    ) -> None:

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'`, the `inplace_abn` package must be installed. "
                "For installation instructions, see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            # Use InPlaceABN with a ReLU (or LeakyReLU) activation internally
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            # We'll skip the explicit ReLU layer since InPlaceABN includes activation
            relu = nn.Identity()
        elif use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super().__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze and Excitation (SCSE) block.

    This module applies both channel squeeze and excitation (cSE) and
    spatial squeeze and excitation (sSE) on an input feature map. The output
    is a sum of the cSE and sSE weighted feature maps.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for the channel squeeze. Defaults to 16.
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying both channel and spatial weighting.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Weighted sum of the original features.
        """
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):
    """
    A thin wrapper around torch.argmax to be used as a module, typically
    for activation in segmentation tasks.

    Args:
        dim (int, optional): The dimension along which to perform the argmax.
                             Defaults to None, which flattens the input.
    """

    def __init__(self, dim: int = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform argmax on the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Argmax indices of `x` along `self.dim`.
        """
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):
    """
    A flexible activation module that can be used to apply a variety of activations
    by name or a custom callable.

    Supported names include:
        - None or 'identity'
        - 'sigmoid'
        - 'softmax2d' or 'softmax'
        - 'logsoftmax'
        - 'tanh'
        - 'argmax' or 'argmax2d'
        - a user-specified callable

    Args:
        name (str or callable): The name of the activation or a callable.
        **params: Additional keyword arguments passed to the activation if it is callable.
    """

    def __init__(self, name: Any, **params: Any) -> None:
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            # dim=1 is typical for segmentation (C x H x W)
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            # For semantic segmentation typically along channel dim=1
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable or one of "
                f"['sigmoid', 'softmax2d', 'softmax', 'logsoftmax', 'tanh', 'argmax', "
                f"'argmax2d', None, 'identity'], got {name}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the chosen activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated output.
        """
        return self.activation(x)


class Attention(nn.Module):
    """
    A module for applying attention mechanisms to a feature map.

    Currently supports:
        - None: Identity
        - 'scse': SCSEModule

    Args:
        name (str): Name of the attention mechanism.
        **params: Additional parameters for the attention module if required.
    """

    def __init__(self, name: str, **params: Any) -> None:
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError(f"Attention {name} is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the attention operation.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map after attention.
        """
        return self.attention(x)


class Flatten(nn.Module):
    """
    A utility module to flatten a 4D tensor into 2D by keeping batch dimension and
    flattening across all other dimensions.

    Typically used before a fully connected layer.

    Example:
        x of shape (B, C, H, W) -> Flatten -> shape (B, C*H*W)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flatten the input while preserving the batch dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Flattened tensor of shape (B, C*H*W).
        """
        return x.view(x.size(0), -1)
