# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    """
    A Conv2d -> BatchNorm (optional) -> ReLU block with flexible options:
      - InplaceABN support if `use_batchnorm='inplace'`
      - Standard BatchNorm2d if `use_batchnorm=True`
      - No BatchNorm if `use_batchnorm=False`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_batchnorm: bool = True
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for the Conv2d layer.
            padding (int): Zero-padding for the Conv2d layer, default 0.
            stride (int): Stride for the Conv2d layer, default 1.
            use_batchnorm (bool or str): 
                - True: Use standard BatchNorm2d
                - False: No BatchNorm
                - "inplace": Use InPlaceABN (requires additional package)
        """
        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "To use `use_batchnorm='inplace'`, the `inplace_abn` package must be installed. "
                "See https://github.com/mapillary/inplace_abn for installation instructions."
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            # Use InPlaceABN
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()
        elif use_batchnorm and use_batchnorm != "inplace":
            # Use standard BatchNorm2d
            bn = nn.BatchNorm2d(out_channels)
        else:
            # No BatchNorm
            bn = nn.Identity()

        super().__init__(conv, bn, relu)


class ArgMax(nn.Module):
    """
    A wrapper module that performs argmax over a given dimension.
    """

    def __init__(self, dim=None):
        """
        Args:
            dim (int, optional): Dimension along which to compute argmax. 
                                 If None, flatten input before argmax.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Argmax indices of shape (N, *).
        """
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):
    """
    A flexible activation module that can apply a variety of functions,
    including sigmoid, softmax, logsoftmax, tanh, argmax, or a custom callable.
    """

    def __init__(self, name, **params):
        """
        Args:
            name: 
                None or "identity" -> nn.Identity
                "sigmoid" -> nn.Sigmoid
                "softmax2d" -> nn.Softmax(dim=1)
                "softmax" -> nn.Softmax
                "logsoftmax" -> nn.LogSoftmax
                "tanh" -> nn.Tanh
                "argmax" -> ArgMax
                "argmax2d" -> ArgMax(dim=1)
                or a callable for custom activation
            **params: Additional keyword arguments for the chosen activation.
        """
        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                "Activation must be one of: [None, 'identity', 'sigmoid', 'softmax2d', "
                "'softmax', 'logsoftmax', 'tanh', 'argmax', 'argmax2d'] or a callable. "
                f"Got: {name}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the chosen activation function.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output after activation.
        """
        return self.activation(x)


class Flatten(nn.Module):
    """
    Flattens a tensor from shape (N, C, H, W) to (N, C*H*W).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (N, C, H, W).

        Returns:
            torch.Tensor: Flattened output of shape (N, C*H*W).
        """
        return x.view(x.shape[0], -1)
