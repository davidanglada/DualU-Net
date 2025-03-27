from typing import Optional, Union, List, Callable
import torch
import torch.nn as nn

from .decoder_segment_convnext import UnetDecoder_Segment
from .decoder_count_convnext import UnetDecoder_Count
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead, CountHead


class DualUNet_ConvNext(SegmentationModel):
    """
    DualUNet_ConvNext is a fully convolutional network for image segmentation and counting tasks.

    It consists of:
      1. An encoder (backbone) that extracts multi-scale feature maps.
      2. Two separate decoders:
         - A segmentation decoder (UnetDecoder_Segment) that produces segmentation masks.
         - A count (density) decoder (UnetDecoder_Count) that produces density/centroid maps.
      3. Two corresponding heads:
         - A segmentation head (SegmentationHead) for the segmentation output.
         - A count head (CountHead) for the density/centroid output.
      4. An optional classification head (ClassificationHead) if `aux_params` is provided.

    Args:
        encoder_name (str): Name of the encoder backbone model (e.g., "convnext_base", "convnext_large").
        encoder_depth (int): Depth of the encoder in [3..5]. Deeper => more feature maps.
        encoder_weights (str, optional): Pretrained weights for the encoder. 
            Can be None or e.g. "imagenet" or other specific strings depending on the encoder.
        decoder_use_batchnorm (bool): If True, use batch normalization in the decoders.
        decoder_channels (List[int]): Channel dimension for each decoder stage (top to bottom). 
            Length = `encoder_depth`.
        decoder_attention_type (str, optional): If set, type of attention to apply in each decoder block (e.g. "scse").
        in_channels (int): Number of input channels for the model. Default=3 (RGB).
        classes_s (int): Number of output segmentation classes.
        classes_c (int): Number of output channels for the count/density map.
        activation_s (Union[str, Callable, None], optional): Activation after the segmentation head 
            (e.g., "sigmoid", "softmax2d", or a callable). Default is None.
        activation_c (Union[str, Callable, None], optional): Activation after the count head.
        aux_params (dict, optional): If provided, creates a classification head with specified params:
            {
              'classes': (int) number of classes,
              'pooling': (str) "max" or "avg",
              'dropout': (float) dropout ratio in [0,1),
              'activation': (str or None) e.g. "softmax", "sigmoid"
            }

    Returns:
        DualUNet_ConvNext model (nn.Module): A PyTorch module with:
          - self.encoder (backbone)
          - self.decoder (segmentation decoder)
          - self.decoder_count (count/density decoder)
          - self.segmentation_head
          - self.count_head
          - self.classification_head (if aux_params is not None)
    """

    def __init__(
        self,
        encoder_name: str = "convnext_base",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes_s: int = 1,
        classes_c: int = 1,
        activation_s: Optional[Union[str, Callable]] = None,
        activation_c: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        # 1. Build encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # 2. Segmentation decoder & head
        self.decoder = UnetDecoder_Segment(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
            center=False,  # Center block for convnext not typically used
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_s,
            activation=activation_s,
            kernel_size=3,
        )

        # 3. Count/density decoder & head
        self.decoder_count = UnetDecoder_Count(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
            center=False,  # Similarly, no center block
        )
        self.count_head = CountHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_c,
            activation=activation_c,
            kernel_size=3,
        )

        # 4. Optional classification head
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"dual_unet_{encoder_name}"
        self.initialize()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass. Returns segmentation and count predictions (and optionally classification).

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).

        Returns:
            A tuple or list of:
              - segmentation output of shape (N, classes_s, H, W)
              - count output of shape (N, classes_c, H, W)
              - classification output (if classification_head is present)
        """
        # 1. Pass input through encoder
        features = self.encoder(x)

        # 2. Decode features for segmentation
        seg_decoded = self.decoder(*features)
        seg_output = self.segmentation_head(seg_decoded)

        # 3. Decode features for counting/density
        cnt_decoded = self.decoder_count(*features)
        cnt_output = self.count_head(cnt_decoded)

        # 4. Classification head if present
        if self.classification_head is not None:
            # Typically the classification is built off the last encoder feature
            cls_output = self.classification_head(features[-1])
            return seg_output, cnt_output, cls_output

        return seg_output, cnt_output
