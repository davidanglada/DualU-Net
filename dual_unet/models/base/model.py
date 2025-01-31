import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    # MODIFICAT PER INCLOURE DUAL UNET
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output_s = self.decoder(*features)
        masks_s = self.segmentation_head(decoder_output_s)
        

        if hasattr(self, 'count_head') and self.count_head is not None:
            decoder_output_c = self.decoder_count(*features)
            masks_c = self.count_head(decoder_output_c)
            
            if hasattr(self, 'classification_head') and self.classification_head is not None:
                labels = self.classification_head(features[-1])
                return masks_s, masks_c, labels
            return masks_s, masks_c
        else:
            if hasattr(self, 'classification_head') and self.classification_head is not None:
                labels = self.classification_head(features[-1])
                return masks_s, labels

        return masks_s

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
