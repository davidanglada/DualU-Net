import torch
import torch.nn as nn


def patch_first_conv(model: nn.Module, in_channels: int) -> None:
    """
    Adjust the input channels of the first convolution layer in a model to `in_channels`.

    If `in_channels` is 1 or 2, the existing weights are partially reused:
      - For `in_channels == 1`, the weights are summed across the original input channels
        to create a single-channel kernel.
      - For `in_channels == 2`, the first two channels are used and scaled by (3/2)
        to preserve total parameter norm.

    If `in_channels` is greater than 3, a new weight tensor is created and randomly
    re-initialized using the module's default parameter initialization.

    Args:
        model (nn.Module): The model containing the convolutional layer to be patched.
        in_channels (int): The new number of input channels for the first convolution.
    """
    # Locate the first nn.Conv2d module in the model
    conv_module: nn.Conv2d = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_module = module
            break
    if conv_module is None:
        return  # No Conv2d found, nothing to patch.

    # Change input channel metadata
    conv_module.in_channels = in_channels
    original_weight = conv_module.weight.detach()
    need_reset = False

    # Handle special cases for 1 or 2 input channels
    if in_channels == 1:
        # Sum weights across the original input channels to produce a single channel
        new_weight = original_weight.sum(dim=1, keepdim=True)
    elif in_channels == 2:
        # Use the first two channels, scale by (3/2) to somewhat preserve total magnitude
        new_weight = original_weight[:, :2] * (3.0 / 2.0)
    else:
        # For in_channels > 3, create a new weight parameter and flag for re-init
        need_reset = True
        new_weight = torch.empty(
            conv_module.out_channels,
            in_channels // conv_module.groups,
            *conv_module.kernel_size,
            dtype=original_weight.dtype,
            device=original_weight.device,
        )

    # Replace the module's weight
    conv_module.weight = nn.parameter.Parameter(new_weight)

    # If we made a new weight tensor, reset its parameters to default initialization
    if need_reset:
        conv_module.reset_parameters()


def replace_strides_with_dilation(module: nn.Module, dilation_rate: int) -> None:
    """
    Replace strides with dilated convolutions in all nn.Conv2d modules within a given module.

    Specifically:
      - Sets the stride of each Conv2d to (1, 1).
      - Sets the dilation to (dilation_rate, dilation_rate).
      - Adjusts the padding accordingly based on the kernel size and the new dilation.

    This is often used to maintain higher resolution feature maps in deeper layers
    (e.g., for segmentation tasks).

    Args:
        module (nn.Module): The module (or layer) whose Conv2d layers will have their strides replaced.
        dilation_rate (int): The dilation factor to apply.
    """
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            # Replace stride with 1
            mod.stride = (1, 1)

            # Set dilation
            mod.dilation = (dilation_rate, dilation_rate)

            # Adjust padding based on kernel size and dilation
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kw // 2) * dilation_rate)

            # Some models (e.g., EfficientNet) may have a "static_padding" attribute
            # that needs to be disabled when dilation is used
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()
