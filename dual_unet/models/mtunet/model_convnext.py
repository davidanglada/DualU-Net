# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------


from typing import Optional, List

from .decoder_unet_convnext import UnetDecoder

from ..encoders import get_encoder
from ..base import SegmentationModel, SegmentationHead, CountHead


class DualUNetConvNext(SegmentationModel):
    """
    DualUNetConvNext is a fully convolutional neural network for image segmentation and count/density estimation,
    using a ConvNeXt-inspired decoder architecture. It creates two decoder instances from the same class
    (one for segmentation, one for counting), hence "dual" decoders.

    By default:
      - The encoder is "resnext50_32x4d" with ImageNet weights (if available).
      - Both decoders share the same UnetDecoder class but are separate instances.
      - No activation functions are applied in the heads (raw logits output).

    Args:
        encoder_name (str): Name of the backbone encoder. Default "resnext50_32x4d".
        encoder_depth (int): Number of downsampling stages in the encoder [3..5]. Default is 5.
        encoder_weights (Optional[str]): Pretrained weights for the encoder (e.g. "imagenet") or None.
        decoder_use_batchnorm (bool): If True, use BatchNorm2d in decoder blocks.
        decoder_channels (List[int]): Number of channels at each decoder stage. Must match `encoder_depth`.
        in_channels (int): Number of channels in the input image, default 3 (RGB).
        classes_s (int): Output channels for segmentation.
        classes_c (int): Output channels for count/density.
        aux_params (Optional[dict]): If provided, builds an optional classification head on top of the encoder.

    Returns:
        A PyTorch nn.Module implementing a dual U-Net design with a ConvNeXt-like decoder architecture.
    """

    def __init__(
        self,
        encoder_name: str = "resnext50_32x4d",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes_s: int = 1,
        classes_c: int = 1,
        aux_params: Optional[dict] = None,
    ):
        """
        Initialize DualUNetConvNext with separate decoders for segmentation and counting tasks,
        both using the same ConvNeXt-like UnetDecoder architecture.

        Args:
            encoder_name: Default "resnext50_32x4d".
            encoder_depth: Default 5, controlling number of downsampling stages.
            encoder_weights: Pretrained weights or None.
            decoder_use_batchnorm: Whether to use batch normalization (True/False/"inplace").
            decoder_channels: List of channels for each decoder stage.
            in_channels: Number of input channels (3 for RGB).
            classes_s: Segmentation output channels.
            classes_c: Counting/density output channels.
            aux_params: Optional dict for building a classification head.
        """
        super().__init__()

        # -----------------
        # 1) Encoder
        # -----------------
        self.encoder = get_encoder(
            encoder_name=encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # -----------------
        # 2) Segmentation Decoder & Head
        # -----------------
        self.decoder_seg = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_s,
            activation=None,  # Raw logits
            kernel_size=3,
        )

        # -----------------
        # 3) Count Decoder & Head
        # -----------------
        self.decoder_count = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
        )
        self.count_head = CountHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_c,
            activation=None,  # Raw logits
            kernel_size=3,
        )

        self.classification_head = None

        self.name = f"dualunetconvnext-{encoder_name}"
        self.initialize()  # Provided by the SegmentationModel base class
