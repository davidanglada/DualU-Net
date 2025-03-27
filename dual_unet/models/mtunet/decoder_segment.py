import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md


class DecoderBlock(nn.Module):
    """
    A single decoding block for the U-Net segmentation decoder. It:
      1. Upsamples the input feature map by a factor of 2.
      2. Optionally concatenates a skip connection.
      3. Applies two Conv2d + ReLU operations (optionally with BatchNorm).
      4. Optionally applies attention after each convolution layer.
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
            in_channels (int): Number of channels in the decoder input.
            skip_channels (int): Number of channels in the skip connection.
            out_channels (int): Number of output channels after the block.
            use_batchnorm (bool): Whether to use batch normalization in Conv2dReLU.
            attention_type (str, optional): If provided, the type of attention mechanism used.
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
        # Optional attention after concatenation
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
        # Optional attention after the second conv
        self.attention2 = md.Attention(
            attention_type,
            in_channels=out_channels
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for a single decoder block.

        1. Upsample x by factor of 2 (nearest neighbor).
        2. If skip is provided, concatenate it with x and apply the first attention.
        3. Apply two Conv2dReLU layers, each potentially followed by attention.

        Args:
            x (torch.Tensor): Input feature map of shape (N, in_channels, H, W).
            skip (torch.Tensor, optional): Skip feature map of shape (N, skip_channels, H', W').

        Returns:
            torch.Tensor: Output feature map of shape (N, out_channels, 2H, 2W).
        """
        # Upsample by factor of 2
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        # Concatenate skip if present
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        # Apply first conv
        x = self.conv1(x)
        # Apply second conv
        x = self.conv2(x)
        x = self.attention2(x)

        return x


class CenterBlock(nn.Sequential):
    """
    A center block optionally placed at the bottom of the U-Net. 
    It applies two consecutive Conv2d + ReLU layers to refine the deepest feature map.
    """

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        """
        Args:
            in_channels (int): Number of channels in the center block input.
            out_channels (int): Number of channels after the block.
            use_batchnorm (bool): Whether to use batch normalization in the Conv2dReLU.
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
    A U-Net style decoder tailored for segmentation tasks. It reverses encoder features, optionally 
    applies a center block, and then uses multiple DecoderBlocks to produce the final segmentation 
    feature map.
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
            encoder_channels (list): Channel counts for features returned by the encoder.
                Usually in order of increasing spatial resolution.
            decoder_channels (list): Desired channel counts for each decoder stage.
            n_blocks (int): Number of decoding stages (commonly 5 in U-Net).
            use_batchnorm (bool): Whether to use batch normalization in the Conv2dReLU layers.
            attention_type (str, optional): If set, the type of attention used in the decoder.
            center (bool): Whether to use a CenterBlock at the bottom of the U-Net.
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but `decoder_channels` provides {len(decoder_channels)} stages."
            )

        # Remove the first skip (same resolution as input) and reverse the list
        encoder_channels = encoder_channels[1:][::-1]

        # Build the in/out channel tuples for each stage
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + decoder_channels[:-1]
        skip_channels = encoder_channels[1:] + [0]
        out_channels = decoder_channels

        # Optional center block
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # Create decoding blocks
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net decoder for segmentation.

        Args:
            *features (torch.Tensor): The list of encoder feature maps. The first is 
                the same resolution as the input and is removed, the rest are reversed 
                so that the last is the deepest (head).

        Returns:
            torch.Tensor: The final decoded feature map with the shape determined by 
            the last DecoderBlock's output channels and the largest spatial resolution.
        """
        # Discard first skip, reverse the rest
        features = features[1:][::-1]

        # The first one in reversed order is the "head" (deepest layer)
        head = features[0]
        skips = features[1:]

        # Pass through the center block
        x = self.center(head)

        # Decode, using skip connections
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
