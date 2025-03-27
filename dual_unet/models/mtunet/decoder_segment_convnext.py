import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md


class DecoderBlock(nn.Module):
    """
    A single decoding block for the UnetDecoder_Segment, used for segmentation.

    It performs the following steps in forward pass:
      1. Optional upsampling by factor of 2.
      2. Optional concatenation with a skip connection.
      3. Two sequential Conv2D + ReLU blocks (optionally with BatchNorm).
      4. Optional attention after each convolution block.

    Note:
        If skip is None, the code includes repeated interpolation lines (see code),
        but practically skip connections are typically always provided except 
        for the last block or special cases.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        attention_type: str = None
    ):
        """
        Args:
            in_channels (int): Number of channels in the input feature map of this block.
            skip_channels (int): Number of channels in the skip connection feature map.
            out_channels (int): Number of channels after the final conv in this block.
            use_batchnorm (bool): Whether to use batch normalization in Conv2dReLU layers.
            attention_type (str, optional): If not None, specifies the attention mechanism to apply.
        """
        super().__init__()
        # First conv + attention
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )

        # Second conv + attention
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(
            attention_type, in_channels=out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor = None,
        interpolate: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of a single decoder block.

        Args:
            x (torch.Tensor): The input feature map of shape (N, in_channels, H, W).
            skip (torch.Tensor, optional): The skip connection feature map of shape 
                (N, skip_channels, H', W'). If provided, it is concatenated after upsampling x.
            interpolate (bool): If True, upsample x by a factor of 2.

        Returns:
            torch.Tensor: The output feature map of shape (N, out_channels, H_out, W_out).
        """
        # Upsample the input by factor of 2 if requested
        if interpolate:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            # If skip is missing, repeatedly upsample x. Possibly a fallback or error condition.
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        # First conv + attention
        x = self.conv1(x)
        # Second conv + attention
        x = self.conv2(x)
        x = self.attention2(x)

        return x


class CenterBlock(nn.Sequential):
    """
    A center block that can optionally be placed between the encoder and decoder in a U-Net.
    It consists of two consecutive Conv2dReLU layers to further process 
    the deepest feature map before decoding starts.
    """

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        """
        Args:
            in_channels (int): Number of input channels from the encoder's deepest layer.
            out_channels (int): Number of output channels from the center block.
            use_batchnorm (bool): Whether to use batch normalization in the Conv2dReLU layers.
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


class UnetDecoder_Segment(nn.Module):
    """
    A U-Net style decoder specifically for segmentation tasks.

    The decoder is constructed in a top-down fashion, where:
      - Features from the encoder are provided in increasing resolution order.
      - The first skip (same resolution as the input) is removed.
      - The remaining features are reversed to start from the deepest layer.
      - Optionally, a CenterBlock is applied at the deepest layer.
      - A series of DecoderBlocks is then used to progressively upsample and merge with skip connections.
    """

    def __init__(
        self,
        encoder_channels: list,
        decoder_channels: list,
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: str = None,
        center: bool = False
    ):
        """
        Args:
            encoder_channels (list of int): The number of channels for each feature map from the encoder.
            decoder_channels (list of int): The desired number of output channels at each decoder stage.
            n_blocks (int): Number of decoding stages (commonly 5 in U-Net).
            use_batchnorm (bool): If True, apply batch normalization in each Conv2dReLU.
            attention_type (str, optional): Type of attention to use in the decoder blocks (or None).
            center (bool): If True, use a CenterBlock in the deepest part of the decoder.
        """
        super().__init__()

        # Check consistency of the provided number of blocks vs. channels
        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but `decoder_channels` has {len(decoder_channels)} entries."
            )

        # Remove the first skip (same resolution as input) and reverse
        encoder_channels = encoder_channels[1:][::-1]

        # Prepare the in/out channels for each decoder block
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # Optional center block at the deepest level
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # Build decoder blocks
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net segmentation decoder.

        Steps:
          1. Remove the first skip (same resolution as input).
          2. Reverse the feature list so the last feature is the deepest (head).
          3. Pass the head through the (optional) center block.
          4. For each decoder block, optionally upsample, merge skip, and decode.

        Args:
            *features (torch.Tensor): A list of encoder feature maps at different resolutions.

        Returns:
            torch.Tensor: The final upsampled segmentation feature map.
        """
        # Remove the first skip and reverse
        features = features[1:][::-1]

        # The head is the deepest feature map
        head = features[0]
        # The remaining are skip connections
        skips = features[1:]

        # Pass head through center block
        x = self.center(head)

        # Decode with skip connections
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            # The original code forcibly disables upsampling on the final block
            if i >= len(skips) - 1:
                x = decoder_block(x, skip, interpolate=False)
            else:
                x = decoder_block(x, skip, interpolate=True)

        return x
