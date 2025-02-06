# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn


def patch_first_conv(model: nn.Module, in_channels: int) -> None:
    """
    Adjust the first convolution layer in a model to handle a custom number of input channels.

    Behavior:
      - If in_channels == 1, sum the original weights across the channel dimension.
      - If in_channels == 2, slice out the first two channels of weights and scale by (3/2).
      - If in_channels > 3, create new weights from scratch (reset to kaiming init).
    """
    # Find the first Conv2d module
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # Adjust the input channels
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        # Sum across the channel dimension to create a single-channel kernel
        weight = weight.sum(dim=1, keepdim=True)
    elif in_channels == 2:
        # Use first 2 channels and scale
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        # Create new random weights for >3 channels
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()


def replace_strides_with_dilation(module: nn.Module, dilation_rate: int) -> None:
    """
    Replace strides with dilation in every Conv2d layer of a given module, effectively increasing
    the receptive field without increasing the number of parameters.

    Args:
        module (nn.Module): The module (or sub-module) to modify.
        dilation_rate (int): The dilation factor to apply.
    """
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kw // 2) * dilation_rate)

            # Some EfficientNet variants have a "static_padding" attribute
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()
