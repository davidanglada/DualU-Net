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
    A basic U-Net decoder block that:
      1) Upsamples the incoming feature map by a factor of 2 (nearest-neighbor).
      2) Concatenates it with a corresponding skip connection (if not None).
      3) Applies two consecutive convolution + ReLU layers (with optional BatchNorm).

    No attention modules are used in this block.
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
            in_channels (int): Number of input channels (before concatenation).
            skip_channels (int): Number of channels in the skip connection.
            out_channels (int): Number of output channels after the convolutions.
            use_batchnorm (bool): If True, use BatchNorm2d between Conv2D and ReLU.
        """
        super().__init__()

        # First convolution: in_channels + skip_channels -> out_channels
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        # Second convolution: out_channels -> out_channels
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder block.

        Args:
            x (torch.Tensor): Feature map from the previous decoder step.
            skip (torch.Tensor): Corresponding skip connection feature map (can be None).

        Returns:
            torch.Tensor: Output feature map of shape (N, out_channels, H_out, W_out).
        """
        # 1) Upsample
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        # 2) Concatenate skip connection if available
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        # 3) Two consecutive convs
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class CenterBlock(nn.Sequential):
    """
    Optional center block used at the transition between encoder and decoder stages
    (common in some U-Net variants). Consists of two consecutive Conv2d+ReLU layers.
    """

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_batchnorm (bool): If True, use BatchNorm2d in each conv layer.
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
    A generic U-Net decoder, which:
      1) Optionally applies a CenterBlock if `center=True`.
      2) Iteratively builds decoder blocks to combine upsampled features
         with skip connections from the encoder.
      3) Produces the final high-resolution feature map.
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
            encoder_channels (list): List of channel sizes from the encoder, e.g. [512, 256, 128, 64, 32].
            decoder_channels (list): List of channel sizes in the decoder, e.g. [256,128,64,32,16].
                                     Length must match `n_blocks`.
            n_blocks (int): Number of decoder blocks. Must match the length of `decoder_channels`.
            use_batchnorm (bool): If True, use BatchNorm2d in all decoder conv layers.
            center (bool): Whether to add a CenterBlock at the beginning of the decoder.
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but `decoder_channels` has length {len(decoder_channels)}."
            )

        # The first encoder skip connection has the same spatial resolution as the input;
        # typically we don't use it in classic U-Net, so we remove it:
        # e.g., if encoder_channels = [3, 64, 128, 256, 512], remove the first one -> [64,128,256,512].
        encoder_channels = encoder_channels[1:]
        # Reverse so that we start from the deepest features
        encoder_channels = encoder_channels[::-1]

        # e.g. head_channels=512, so in_channels might be [512,256,128,64], skip_channels might be [128,64,32,0], etc.
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # Center block (optional)
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
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
            features: A list of Tensors from the encoder (skips), typically in ascending
                      spatial resolution order (from deepest to shallowest).
                      The very first element is often the output of the deepest encoder layer.

        Returns:
            torch.Tensor: The final decoder output after upsampling all stages.
        """
        # Same logic: remove the first skip with same spatial resolution
        features = features[1:]
        # Reverse to start from the deepest level
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        # Center
        x = self.center(head)

        # Apply each decoder block
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
