import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md


class DecoderBlock(nn.Module):
    """
    A single decoding block used in the UnetDecoder_Count architecture. This block:
      - Upsamples the input by a factor of 2,
      - Optionally concatenates a skip connection,
      - Applies optional attention layers, and
      - Uses two sequential Conv2D + ReLU operations (with optional batch norm).
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
            in_channels (int): Number of channels in the input feature map.
            skip_channels (int): Number of channels in the skip connection feature map.
            out_channels (int): Number of output channels after this block.
            use_batchnorm (bool): If True, apply batch normalization in each Conv2dReLU.
            attention_type (str, optional): Type of attention mechanism. If None, no attention is used.
        """
        super().__init__()

        # First Conv2D + ReLU block
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)

        # Second Conv2D + ReLU block
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for a single decoder block.

        Steps:
          1. Upsample the input tensor by factor of 2 (nearest neighbor).
          2. Concatenate the skip connection (if provided).
          3. Apply the first Conv2d + attention sequence.
          4. Apply the second Conv2d + attention sequence.

        Args:
            x (torch.Tensor): Input feature map of shape (N, in_channels, H, W).
            skip (torch.Tensor, optional): Skip feature map of shape (N, skip_channels, H', W').
                                           If provided, it is concatenated to the upsampled x.

        Returns:
            torch.Tensor: The output feature map of shape (N, out_channels, 2H, 2W) (assuming skip not None).
        """
        # Upsample by factor of 2
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        # Two convolution + ReLU + attention blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    """
    A center block used in the decoder, optionally placed at the deepest part of the network.
    It applies two sequential Conv2D + ReLU operations to refine the features.

    Args:
        in_channels (int): Number of channels entering the center block.
        out_channels (int): Number of output channels from the center block.
        use_batchnorm (bool): Whether to use batch normalization in the Conv2DReLU modules.
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
    A U-Net-like decoder specifically for a "count" branch (e.g. density/centroid regression).
    It expects multiple features from an encoder, optionally passes the deepest feature
    through a CenterBlock, and then successively decodes with skip connections.

    Attributes:
        blocks (nn.ModuleList): A list of DecoderBlock modules forming the decoding path.
        center (nn.Module): Either a CenterBlock or an Identity, depending on the 'center' flag.
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
        Initialize the UnetDecoder_Count.

        Args:
            encoder_channels (list): A list of channel dimensions for encoder outputs at different stages.
            decoder_channels (list): A list of channel dimensions for decoder outputs (from deep to shallow).
            n_blocks (int): Number of decoding blocks (usually 5 for a full U-Net).
            use_batchnorm (bool): Whether to use batch normalization in convolution blocks.
            attention_type (str, optional): If set, the type of attention to apply in each block.
            center (bool): If True, a CenterBlock is used at the bottom of the U-Net (between encoder and decoder).
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but `decoder_channels` has {len(decoder_channels)} entries."
            )

        # Remove the first skip which has the same spatial resolution as the input
        # and reverse the rest to match the top-down order
        encoder_channels = encoder_channels[1:][::-1]

        # Prepare in/out channels for each decoder block
        head_channels = encoder_channels[0]  # the deepest feature map channels
        in_channels = [head_channels] + decoder_channels[:-1]  # in channels of each block
        skip_channels = encoder_channels[1:] + [0]  # skip channels for each block, last is 0
        out_channels = decoder_channels

        # Center block if needed
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
        Forward pass of the UnetDecoder_Count.

        The features are expected in increasing spatial resolution order. 
        The first skip (same resolution as input) is removed. The rest are reversed so
        that the last feature in the list is the deepest (head) feature.

        Args:
            *features (torch.Tensor): Encoder outputs, e.g. from a ResNet or other backbone.

        Returns:
            torch.Tensor: The final decoded feature map from the last decoder block.
        """
        # Remove first skip (same resolution) and reverse
        features = features[1:][::-1]

        # The first is the head (deepest feature map), the rest are skip connections
        head = features[0]
        skips = features[1:]

        # Optionally pass head through the center block
        x = self.center(head)

        # Decode from deep to shallow
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
