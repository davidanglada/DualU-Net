# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------


from typing import Optional, List

import torch.nn as nn
from .decoder_unet import UnetDecoder

from ..encoders import get_encoder
from ..base import SegmentationModel, SegmentationHead, CountHead


class DualUNet(SegmentationModel):
    """
    DualUNet is a fully convolutional neural network for cell image segmentation
    and count (density) estimation, using a single UnetDecoder architecture
    for both tasks (two separate decoder instances).

    1) **Segmentation Decoder**: Produces a semantic segmentation mask.
    2) **Count Decoder**: Produces a count/density map (e.g., for cells or nuclei).

    By default:
      - The encoder is "resnext50_32x4d" with ImageNet weights (if available).
      - Both segmentation and count decoders share the same UnetDecoder class (but are separate instances).
      - No activation functions are applied in the heads (raw logits output).

    Args:
        encoder_name (str): Name of the backbone encoder, default "resnext50_32x4d".
        encoder_depth (int): Number of downsampling stages in the encoder [3..5], default 5.
        encoder_weights (Optional[str]): Pretrained weights for the encoder (e.g. "imagenet") or None.
        decoder_use_batchnorm (bool): Use BatchNorm2d in decoder blocks if True (or "inplace" for InplaceABN).
        decoder_channels (List[int]): Number of channels in each decoder stage. Must match `encoder_depth`.
        in_channels (int): Number of input channels, default 3 for RGB.
        classes_s (int): Number of output segmentation channels.
        classes_c (int): Number of output count/density channels.
        aux_params (Optional[dict]): If provided, a classification head is built on top of the encoder.

    Returns:
        nn.Module: A PyTorch model implementing a dual-decoder U-Net design (segmentation + count).
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
        Initialize the DualUNet model, creating two separate UnetDecoder instances from the
        same architecture: one for segmentation, one for counting.

        Args:
            encoder_name: Default "resnext50_32x4d".
            encoder_depth: Default 5, controlling number of downsampling stages.
            encoder_weights: Pretrained weights or None.
            decoder_use_batchnorm: Whether to use batch normalization (True/False/"inplace") in decoders.
            decoder_channels: List defining the channels in each decoder stage.
            in_channels: Number of input channels (e.g., 3 for RGB).
            classes_s: Number of segmentation output channels.
            classes_c: Number of counting/density output channels.
            aux_params: Dictionary for an optional classification head (e.g. {"classes": 2, ...}).
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
            activation=None,  # No activation (raw logits)
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
            activation=None,  # No activation (raw logits)
            kernel_size=3,
        )

        self.classification_head = None

        self.name = f"dualunet-{encoder_name}"
        self.initialize()  # Provided by the base SegmentationModel
