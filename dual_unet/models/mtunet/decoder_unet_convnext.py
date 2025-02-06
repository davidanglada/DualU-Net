# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------



import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md


class DecoderBlock(nn.Module):
    """
    A decoder block for U-Net-like architectures. It optionally upsamples the input,
    concatenates it with a skip connection (if provided), and applies two consecutive
    convolution + ReLU layers (with optional BatchNorm).

    Special case: If no skip is provided, the code applies extra upsampling steps.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ):
        """
        Args:
            in_channels (int): Number of channels in the incoming feature map.
            skip_channels (int): Number of channels in the skip connection feature map.
            out_channels (int): Number of output channels after the two convolutions.
            use_batchnorm (bool): If True, include BatchNorm2d in each conv layer.
        """
        super().__init__()

        # First Conv2dReLU: (in_channels + skip_channels) -> out_channels
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        # Second Conv2dReLU: out_channels -> out_channels
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor = None,
        interpolate: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of the decoder block.

        Args:
            x (torch.Tensor): The feature map from the previous decoder step.
            skip (torch.Tensor, optional): The skip connection feature map. If None, 
                the block applies extra upsampling.
            interpolate (bool): Whether to upsample by a factor of 2 at the start.
                If False, no initial upsampling is done here (useful for special cases).

        Returns:
            torch.Tensor: Output feature map of shape (N, out_channels, H_out, W_out).
        """
        if interpolate:
            # Upsample the input by factor 2
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            # Merge skip connection
            x = torch.cat([x, skip], dim=1)
        else:
            # If there's no skip, do additional upsampling
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        # Convolution layers
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class CenterBlock(nn.Sequential):
    """
    An optional center block typically placed at the deepest part of the U-Net, 
    consisting of two convolution + ReLU layers (with optional BatchNorm).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after each of the two conv layers.
            use_batchnorm (bool): If True, include BatchNorm2d in each conv layer.
        """
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    """
    A generic U-Net decoder that:
      - Optionally applies a CenterBlock on the deepest feature map.
      - Iteratively builds decoder blocks, each upsampling and merging skip connections.
      - Produces a final high-resolution feature map.

    This decoder can be reused for various tasks (e.g., segmentation, counting).
    """

    def __init__(
        self,
        encoder_channels: list,
        decoder_channels: list,
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        center: bool = False,
    ):
        """
        Args:
            encoder_channels (list): List of encoder output channel sizes at different stages.
                For example, if the encoder outputs [64, 128, 256, 512, 1024] features, 
                remove the first one for the skip logic -> [128, 256, 512, 1024].
            decoder_channels (list): List of channel sizes for each decoder stage, e.g. [256,128,64,32,16].
                Must match n_blocks in length.
            n_blocks (int): Depth of the decoder (number of blocks).
            use_batchnorm (bool): Whether to use BatchNorm2d in each conv layer.
            center (bool): If True, insert a CenterBlock before decoder blocks.
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but `decoder_channels` has length {len(decoder_channels)}."
            )

        # Remove the first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # Reverse for top-down approach
        encoder_channels = encoder_channels[::-1]

        # Example:
        #   head_channels = encoder_channels[0]
        #   in_channels = [head_channels] + decoder_channels[:-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # Optional center block
        if center:
            self.center = CenterBlock(
                in_channels=head_channels,
                out_channels=head_channels,
                use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # Build decoder blocks
        blocks = []
        for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels):
            block = DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                use_batchnorm=use_batchnorm,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: A list/tuple of Tensors from the encoder in order of ascending spatial size
                      (the first is the deepest, the last is the shallowest).
                      Typically something like: (feat5, feat4, feat3, feat2, feat1).

        Returns:
            torch.Tensor: The decoded high-resolution feature map.
        """
        # Remove the first skip with same spatial resolution
        features = features[1:]
        # Reverse to start from the deepest feature
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        # Apply center block (if any)
        x = self.center(head)

        # Apply each decoder block
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None

            # Example logic: For the final block(s), we might not interpolate by default
            # Adjust or remove as needed based on your design
            if i >= len(skips) - 1:
                x = decoder_block(x, skip, interpolate=False)
            else:
                x = decoder_block(x, skip, interpolate=True)

        return x
