import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md


class DecoderBlock(nn.Module):
    """
    A single decoding block used in the UnetDecoder_Count architecture. This block:
      - (optionally) upscales the input by a factor of 2 in spatial resolution,
      - concatenates skip connections if provided,
      - applies attention if enabled,
      - applies two sequential Conv2D + ReLU operations (with optional batchnorm).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        attention_type: str = None
    ) -> None:
        """
        Args:
            in_channels (int): Number of channels in the decoder input.
            skip_channels (int): Number of channels in the skip connection features.
            out_channels (int): Number of output channels after this block.
            use_batchnorm (bool): Whether to use batch normalization in Conv2DReLU layers.
            attention_type (str, optional): Type of attention mechanism. If None, no attention is used.
        """
        super().__init__()
        # First conv layer
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # Optional attention after skipping
        self.attention1 = md.Attention(
            attention_type, 
            in_channels=in_channels + skip_channels
        )

        # Second conv layer
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # Optional attention after second conv
        self.attention2 = md.Attention(
            attention_type,
            in_channels=out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor = None,
        interpolate: bool = True
    ) -> torch.Tensor:
        """
        Forward pass for a single decoder block. Optionally upsamples `x`,
        concatenates skip features, applies attention, and does two Convs.

        Args:
            x (torch.Tensor): Decoder input feature map, shape (N, in_channels, H, W).
            skip (torch.Tensor, optional): Skip connection feature map, shape (N, skip_channels, H', W').
                The shape must match `x` after interpolation. If None, skip is not used.
            interpolate (bool): If True, upsample by factor of 2 using nearest neighbor.

        Returns:
            torch.Tensor: The processed feature map of shape (N, out_channels, H', W').
        """
        # Optionally upscale by factor of 2
        if interpolate:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        # If skip connection is provided
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # Concatenate skip
            x = self.attention1(x)
        else:
            # NOTE: The original code suggests there's possibly extra interpolation,
            # but this branch or approach might be incomplete. 
            pass

        # Two convolution + ReLU blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)

        return x


class CenterBlock(nn.Sequential):
    """
    A center block used in the decoder (e.g., bridging the bottom of the U-Net).
    It applies two sequential Conv2D + ReLU operations to refine features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_batchnorm (bool): Whether to use batch normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
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


class UnetDecoder_Count(nn.Module):
    """
    A specialized U-Net-like decoder designed for a "count" branch (i.e., it can 
    produce features for a density or count map). It expects encoded features from
    an encoder, optionally a center block, then successively upscales and merges 
    with skip connections.

    Attributes:
        blocks (nn.ModuleList): A list of `DecoderBlock` modules.
        center (nn.Module): Either a `CenterBlock` or an `Identity`, depending on the `center` flag.
    """

    def __init__(
        self,
        encoder_channels: list,
        decoder_channels: list,
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: str = None,
        center: bool = False,
    ) -> None:
        """
        Args:
            encoder_channels (list of int): Channels in the feature maps from the encoder.
            decoder_channels (list of int): Desired output channels for each decoder block.
            n_blocks (int): Number of decoder blocks. Typically 5 for U-Net.
            use_batchnorm (bool): Whether to use batch normalization in the conv layers.
            attention_type (str, optional): If not None, type of attention to use in the decoder blocks.
            center (bool): If True, use a `CenterBlock` in the middle of the U-Net.
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but `decoder_channels` length is {len(decoder_channels)}."
            )

        # Remove first skip with same spatial resolution & reverse
        encoder_channels = encoder_channels[1:][::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]  # last block has no skip
        out_channels = decoder_channels

        # Optional center block
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # Each DecoderBlock: in_ch + skip_ch -> out_ch
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder. The input `features` are typically the outputs 
        of an encoder, in increasing spatial resolution order.

        We remove the first skip (same spatial resolution as input), reverse the rest, 
        use the first as the "head", pass it through the center, then successively decode.

        Args:
            *features (torch.Tensor): The multi-scale feature maps from the encoder.

        Returns:
            torch.Tensor: The decoded feature map after all blocks.
        """
        # Remove first skip, reverse order
        features = features[1:][::-1]

        # The "head" is the deepest feature map
        head = features[0]
        # The rest are skip connections
        skips = features[1:]

        # Process head via center block
        x = self.center(head)

        # Decoding
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            # The original code sets interpolate=False for the final block 
            # or near-final block. Adjust if needed. 
            if i >= len(skips) - 1:
                x = decoder_block(x, skip, interpolate=False)
            else:
                x = decoder_block(x, skip, interpolate=True)

        return x
