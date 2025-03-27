from typing import Optional, Union, List, Callable
import torch
import torch.nn as nn

from .decoder_segment import UnetDecoder_Segment
from .decoder_count import UnetDecoder_Count
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead, CountHead


class DualUNet(SegmentationModel):
    """
    DualUNet is a fully convolutional network for image segmentation and counting tasks, built in a U-Net-like fashion.

    It consists of:
      1. An encoder (backbone) that extracts multi-scale feature maps.
      2. Two separate decoders:
         - A segmentation decoder (UnetDecoder_Segment) that produces segmentation masks.
         - A count (density) decoder (UnetDecoder_Count) that produces density/centroid maps.
      3. Two corresponding heads:
         - A segmentation head (SegmentationHead) for segmentation.
         - A count head (CountHead) for density or count outputs.
      4. An optional classification head (ClassificationHead), if aux_params is provided.

    Args:
        encoder_name (str): Name of the backbone encoder (e.g. "resnet34", "resnet50"), 
            used to extract multi-scale features.
        encoder_depth (int): Depth of the encoder in the range [3, 5]. Controls how many 
            downsampling stages the encoder has.
        encoder_weights (str, optional): Pretrained weights for the encoder. 
            E.g. "imagenet", or None for random initialization.
        decoder_use_batchnorm (bool): If True, use batch normalization in the decoder blocks.
        decoder_channels (List[int]): The output channels for each decoder stage, 
            from deepest to shallow. Must have length == encoder_depth.
        decoder_attention_type (str, optional): Type of attention to use (e.g. "scse") 
            in the decoder blocks. If None, no attention is applied.
        in_channels (int): Number of channels in the input image. Default is 3 (RGB).
        classes_s (int): Number of segmentation output channels (e.g. segmentation classes).
        classes_c (int): Number of counting/density output channels.
        activation_s (Union[str, Callable, None], optional): Activation function applied after 
            the segmentation head. Examples: "sigmoid", "softmax", or a callable. Default None.
        activation_c (Union[str, Callable, None], optional): Activation function applied after 
            the count head. Default None.
        aux_params (dict, optional): Parameters for an optional classification head. If None,
            no classification head is created. Expected keys include:
                - "classes" (int): Number of classes for classification.
                - "pooling" (str): "avg" or "max" pooling.
                - "dropout" (float): Dropout ratio in [0,1).
                - "activation" (str or None): "sigmoid", "softmax", or None for raw logits.

    Returns:
        A PyTorch nn.Module representing the DualUNet, with:
          - self.encoder
          - self.decoder (for segmentation)
          - self.decoder_count (for counting)
          - self.segmentation_head
          - self.count_head
          - self.classification_head (if aux_params is provided)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
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
        """
        Initialize the DualUNet model.

        The model uses an encoder to extract features, and has two separate decoders 
        for segmentation and count tasks, respectively. Each decoder has its own 
        final head for output.

        Args:
            encoder_name (str): Name of the backbone encoder.
            encoder_depth (int): Depth of the encoder, e.g. 3..5.
            encoder_weights (str, optional): Pretrained weights (e.g. "imagenet") or None.
            decoder_use_batchnorm (bool): If True, use batchnorm in decoder blocks.
            decoder_channels (List[int]): Output channels for each stage in the decoders.
                Length must match encoder_depth.
            decoder_attention_type (str, optional): Type of attention for the decoder. E.g. "scse".
            in_channels (int): Number of input channels (default 3).
            classes_s (int): Number of segmentation classes.
            classes_c (int): Number of count/density output channels.
            activation_s (Union[str, Callable, None], optional): Activation after segmentation head.
            activation_c (Union[str, Callable, None], optional): Activation after count head.
            aux_params (dict, optional): Parameters for an auxiliary classification head, or None.

        """
        super().__init__()

        # Initialize the encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # Build the segmentation decoder and head
        self.decoder = UnetDecoder_Segment(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_s,
            activation=activation_s,
            kernel_size=3,
        )

        # Build the count/density decoder and head
        self.decoder_count = UnetDecoder_Count(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.count_head = CountHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_c,
            activation=activation_c,
            kernel_size=3,
        )

        # Optionally, add a classification head
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"dual_unet_{encoder_name}"
        self.initialize()
