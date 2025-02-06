# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    """
    A base class for segmentation models that includes:
      - Encoder
      - One or more decoders (e.g. for segmentation, counting)
      - Optional classification head
      - Initialization routines for decoders/heads
      - Forward and predict methods
    """

    def initialize(self) -> None:
        """
        Initialize decoder and head(s) using standard weight initialization schemes.
        """
        init.initialize_decoder(self.decoder_seg)
        init.initialize_decoder(self.decoder_count)
        init.initialize_head(self.segmentation_head)
        init.initialize_head(self.count_head)

        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the model: encoder -> decoders -> heads.

        Returns:
            - If count head is present: (segmentation_mask, count_mask) or
              (segmentation_mask, count_mask, classification_labels).
            - Otherwise: segmentation_mask or (segmentation_mask, classification_labels).
        """
        # 1) Encode
        features = self.encoder(x)

        # 2) Decode for segmentation
        decoder_output_s = self.decoder_seg(*features)
        masks_s = self.segmentation_head(decoder_output_s)

        # 3) Decode for count if available
        if hasattr(self, 'count_head') and self.count_head is not None:
            decoder_output_c = self.decoder_count(*features)
            masks_c = self.count_head(decoder_output_c)

            # 4) Classification if available
            if hasattr(self, 'classification_head') and self.classification_head is not None:
                labels = self.classification_head(features[-1])
                return masks_s, masks_c, labels

            return masks_s, masks_c

        else:
            # If no count head, optionally do classification
            if hasattr(self, 'classification_head') and self.classification_head is not None:
                labels = self.classification_head(features[-1])
                return masks_s, labels

        return masks_s

    def predict(self, x: torch.Tensor):
        """
        Inference mode. Switches model to `eval` mode, does a forward pass with no grad.

        Args:
            x (torch.Tensor): A 4D tensor (N, C, H, W)

        Returns:
            Depending on the model architecture, can be a single segmentation mask,
            or multiple outputs (seg + count, seg + labels, etc.).
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
