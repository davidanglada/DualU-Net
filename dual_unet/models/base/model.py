from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from . import initialization as init


class SegmentationModel(nn.Module):
    """
    A base segmentation model structure that includes an encoder, decoder, and
    optional heads for segmentation, counting, and classification.

    The forward pass processes the input through:
      1. An encoder that produces multi-scale feature maps.
      2. A decoder (or multiple decoders) that outputs feature maps for segmentation or counting.
      3. Heads that convert the decoded features into final predictions:
         - segmentation_head for per-pixel class predictions
         - count_head (optional) for density or count maps
         - classification_head (optional) for classification logits

    Attributes:
        encoder (nn.Module): The encoder network that produces feature maps.
        decoder (nn.Module): The main decoder for segmentation.
        segmentation_head (nn.Module): The segmentation head for producing masks.
        count_head (Optional[nn.Module]): Optional head for counting/density maps.
        decoder_count (Optional[nn.Module]): Optional separate decoder for count_head.
        classification_head (Optional[nn.Module]): Optional classification head.
    """

    def __init__(self) -> None:
        super().__init__()
        # These attributes should be set by subclasses or after instantiation:
        # self.encoder = ...
        # self.decoder = ...
        # self.segmentation_head = ...
        # self.count_head = ...
        # self.decoder_count = ...
        # self.classification_head = ...
        pass

    def initialize(self) -> None:
        """
        Initialize the decoder and heads with predefined schemes.

        This uses the initialization functions from `init.initialize_decoder`
        and `init.initialize_head`.
        """
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

        if hasattr(self, "classification_head") and self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the model.

        This method:
          1. Encodes the input using `self.encoder`.
          2. Decodes the features for segmentation with `self.decoder`.
          3. Passes the decoded features through `self.segmentation_head` to produce segmentation masks.
          4. Optionally, if a counting head exists, decodes features with `self.decoder_count` and
             passes them through `self.count_head`.
          5. Optionally, if a classification head exists, applies it to the last feature map for classification.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
                - `masks_s` if only segmentation is used.
                - `(masks_s, labels)` if segmentation and classification heads are used.
                - `(masks_s, masks_c)` if segmentation and count heads are used.
                - `(masks_s, masks_c, labels)` if all three heads are used.
        """
        # Encode input
        features = self.encoder(x)

        # Decode for segmentation
        decoder_output_s = self.decoder(*features)
        masks_s = self.segmentation_head(decoder_output_s)

        # Count head is optional
        if hasattr(self, 'count_head') and self.count_head is not None:
            decoder_output_c = self.decoder_count(*features)
            masks_c = self.count_head(decoder_output_c)

            # Classification head is also optional
            if hasattr(self, 'classification_head') and self.classification_head is not None:
                labels = self.classification_head(features[-1])
                return masks_s, masks_c, labels
            return masks_s, masks_c

        else:
            # If count head is absent, check classification head
            if hasattr(self, 'classification_head') and self.classification_head is not None:
                labels = self.classification_head(features[-1])
                return masks_s, labels

        return masks_s

    def predict(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Inference method. Switches the model to eval mode, then calls `forward(x)`
        under a no-grad context.

        Args:
            x (torch.Tensor): 4D input tensor of shape (batch_size, channels, height, width).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: The model outputs, which may be
            segmentation masks, optionally along with count maps and/or classification logits.
        """
        training_mode = self.training
        if training_mode:
            self.eval()

        with torch.no_grad():
            outputs = self.forward(x)

        if training_mode:
            self.train()  # revert to original mode if needed

        return outputs
