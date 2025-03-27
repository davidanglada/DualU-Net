import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict

from . import _utils as utils


class EncoderMixin:
    """
    A mixin class that adds encoder functionality such as:
      - Managing output channels for feature tensors produced by the encoder.
      - Modifying the first convolution to handle an arbitrary number of input channels.
      - Providing a mechanism to dilate certain encoder stages.

    Attributes:
        _out_channels (Tuple[int, ...]): A tuple containing the number of output channels
            for each level of the encoder. The first element is typically the number of
            input channels, followed by the channels of subsequent stages.
        _depth (int): The depth of the encoder, controlling how many feature maps are produced.
        _in_channels (int): The number of input channels for the first convolution.
    """

    @property
    def out_channels(self) -> List[int]:
        """
        Return the channel dimensions for each tensor in the forward output of the encoder.

        Returns:
            List[int]: A list of the number of channels for each feature map level
                       (from low-level to high-level features).
        """
        # We slice the out_channels tuple to get up to the encoder depth + 1 
        # because the index 0 is often the input stage, and each additional
        # stage corresponds to one deeper level of features.
        return list(self._out_channels[: self._depth + 1])

    def set_in_channels(self, in_channels: int) -> None:
        """
        Change the number of input channels for the encoder's first convolutional layer.

        This method adjusts the internal metadata about `_in_channels` and `_out_channels`
        and calls a utility function to actually patch (modify) the first convolution
        layer of the encoder to accept `in_channels` rather than the default (e.g. 3 for RGB).

        Args:
            in_channels (int): The new number of input channels for the first convolution.
        """
        # If it's already 3, there's nothing to change
        if in_channels == 3:
            return

        self._in_channels = in_channels

        # Adjust the first out_channels entry if it was originally 3
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        # Patch the actual model layers
        utils.patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self) -> List[nn.Module]:
        """
        Retrieve encoder stages as a list of modules.

        This method should be overridden in the specific encoder implementation.
        It typically returns each "stage" (or block) of the encoder as a separate element.

        Returns:
            List[nn.Module]: A list of encoder stages (e.g. [stage0, stage1, stage2, ...]).
        """
        raise NotImplementedError(
            "Please implement get_stages() method in your encoder class."
        )

    def make_dilated(self, stage_list: List[int], dilation_list: List[int]) -> None:
        """
        Convert specified encoder stages to use dilated convolutions instead of strides.

        This is useful for semantic segmentation tasks where you want the encoder
        to maintain higher spatial resolution by replacing strides with dilations.

        Args:
            stage_list (List[int]): A list of stage indices that should be dilated.
            dilation_list (List[int]): The corresponding dilation rates for each stage in `stage_list`.

        Example:
            To dilate stages 3 and 4 by factors of 2 and 4 respectively:
                make_dilated(stage_list=[3,4], dilation_list=[2,4])
        """
        stages = self.get_stages()
        for stage_index, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_index],
                dilation_rate=dilation_rate,
            )
