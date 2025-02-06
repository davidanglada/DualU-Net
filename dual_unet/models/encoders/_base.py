# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict

from . import _utils as utils


class EncoderMixin:
    """
    Mixin that provides additional functionality for encoders, including:
      - Specifying the output channels for each layer's feature map.
      - Dynamically changing the first convolution to handle arbitrary input channels.
    """

    @property
    def out_channels(self) -> List[int]:
        """
        Returns:
            List[int]: The channel dimensions for each feature map produced by the encoder, 
                       up to the specified depth (`_depth`).
        """
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels: int) -> None:
        """
        Adjust the encoder to accept images with a custom number of input channels (in_channels).

        By default, if in_channels=3, no change is made.
        Otherwise, modifies the first convolution and updates `_out_channels` accordingly.
        """
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        utils.patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self):
        """
        Return a list/tuple of encoder stages (e.g., layers or blocks).
        Must be overridden by any child encoder class.

        Raises:
            NotImplementedError: If not overridden in the subclass.
        """
        raise NotImplementedError

    def make_dilated(self, stage_list: List[int], dilation_list: List[int]) -> None:
        """
        Convert specified stages in the encoder to use dilated convolutions
        instead of strided convolutions. Useful for tasks requiring larger
        receptive fields.

        Args:
            stage_list (List[int]): Indices of stages to be modified.
            dilation_list (List[int]): Corresponding dilation rates for each stage.
        """
        stages = self.get_stages()
        for stage_index, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_index],
                dilation_rate=dilation_rate,
            )
