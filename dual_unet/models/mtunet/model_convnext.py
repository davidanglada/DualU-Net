from typing import Optional, Union, List
from .decoder_segment_convnext import UnetDecoder_Segment
from .decoder_count_convnext import UnetDecoder_Count
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead, CountHead # import dels segmentation heads


class MTUnet_ConvNext(SegmentationModel):
    # Corregir
    """DualUnet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and two *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by the decoders to define an accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes_s: A number of classes for segmentation output mask (or you can think as a number of channels of output mask)
        classes_c: A number of classes for count output
        activation_s: An activation function to apply after the final convolution layer in the Segmentation decoder.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        activation_c: An activation function to apply after the final convolution layer in the Count decoder.
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

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
        activation_s: Optional[Union[str, callable]] = None,
        activation_c: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,  # s'hauria de canviar?? -> revisar
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # SEGMENTATION DECODER AND HEAD:

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

        # COUNT DECODER AND HEAD

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

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

# import torch

# # Step 1: Initialize the model
# # Let's assume we want 3 segmentation classes and 1 output for centroid count
# model = MTUnet(
#     encoder_name="resnet34",
#     encoder_depth=5,
#     encoder_weights=None,  # No pretraining for this example
#     in_channels=3,  # RGB images
#     classes_s=3,  # 3 segmentation classes
#     classes_c=1,
#     activation_s='softmax2d'  # 1 output class for centroid count
# )

# # Step 2: Prepare a toy input tensor (Batch of RGB images)
# # Example: Batch of 2 RGB images, size 128x128
# toy_input = torch.randn(2, 3, 128, 128)  # (batch_size, in_channels, height, width)

# # Step 3: Forward pass through the model
# segmentation_output, count_output = model(toy_input)

# # Step 4: Output the shapes of the segmentation and count outputs
# print("Segmentation Output Shape:", segmentation_output.shape)  # Expected: (2, 3, 128, 128)
# print("Count Output Shape:", count_output.shape)  # Expected: (2, 1, 128, 128)

# # Optionally, check the values (since it's a toy example, they'll be random)
# print("Segmentation Output (Example):", segmentation_output[0, 0, :5, :5])  # Print a small region
# print("Count Output (Example):", count_output[0, 0, :5, :5])  # Print a small region
