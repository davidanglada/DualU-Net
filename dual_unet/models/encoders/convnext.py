# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

import timm
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional


class ConvNeXtEncoder(nn.Module):
    """
    A custom ConvNeXt encoder wrapper that:
      1. Creates a ConvNeXt model via timm.
      2. Removes the classification head.
      3. Exposes a `get_stages()` method for U-Net style feature extraction.
      4. Can load pretrained weights via `load_state_dict`.

    This encoder can produce multiple levels of features (stages) typically used in U-Net-like
    models for skip connections.

    Attributes:
        timm_model (nn.Module): The underlying ConvNeXt model from timm.
        depth (int): Number of encoder stages to output.
        out_channels (List[int]): A list specifying the number of channels at each stage.
        in_channels (int): The number of input channels.
    """

    def __init__(
        self,
        out_channels: List[int],
        depth: int = 5,
        **kwargs
    ) -> None:
        """
        Initialize the ConvNeXt encoder.

        Args:
            out_channels (List[int]): Channel sizes for each stage.
            depth (int): Number of stages to produce. If `depth=5`, it yields 6 feature maps (indices 0..5).
            **kwargs: Additional arguments forwarded to timm's `create_model` function
                      (e.g., `in_chans` for the input image channels).
        """
        super().__init__()
        self.timm_model = timm.create_model(
            "convnext_base",  # or another variant, e.g. "convnext_large"
            pretrained=False,
            **kwargs
        )
        self.depth = depth
        self.out_channels = out_channels
        self.in_channels = kwargs.get("in_chans", 3)

        # Remove the classification head if present
        if hasattr(self.timm_model, "head"):
            del self.timm_model.head

    def get_stages(self) -> List[nn.Module]:
        """
        Return the list of modules (stages) in the underlying ConvNeXt model, including stem.

        Returns:
            List[nn.Module]: Each element is a portion of the model representing a "stage."
                             The length of this list is typically 5 (indices 0..4) for a base ConvNeXt,
                             but the method is flexible for different `depth` values.
        """
        return [
            nn.Identity(),  # Stage 0: identity (no change, passing x as-is)
            nn.Sequential(self.timm_model.stem),
            self.timm_model.stages[0],
            self.timm_model.stages[1],
            self.timm_model.stages[2],
            self.timm_model.stages[3],
        ]

    def make_dilated(self, stage_list: List[int], dilation_list: List[int]) -> None:
        """
        Attempt to convert specified encoder stages to dilated mode.

        ConvNeXt encoders do not support dilated convolution mode, so this function raises an error.

        Args:
            stage_list (List[int]): Indices of stages to dilate (unused).
            dilation_list (List[int]): Dilation rates corresponding to each stage (unused).

        Raises:
            ValueError: Always, because dilation is not supported in ConvNeXt.
        """
        raise ValueError("ConvNeXt encoders do not support dilated mode")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through each stage, collecting outputs for e.g. U-Net skip connections.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W).

        Returns:
            List[torch.Tensor]: A list of feature maps from each stage (indices 0..depth).
                                For `depth=5`, there are 6 feature maps returned.
        """
        stages = self.get_stages()
        features = []
        for i in range(self.depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict: Dict[str, Dict[str, torch.Tensor]], strict: bool = True) -> None:
        """
        Load pretrained weights from a dict with the format: {'model': <ConvNeXt weights>}.

        The method removes any classifier head keys (e.g., 'head.bias', 'head.weight')
        before performing the load.

        Args:
            state_dict (Dict[str, Dict[str, torch.Tensor]]): The state dict,
                typically with a key `'model'` containing the actual weights.
            strict (bool): Whether to strictly enforce that the keys in `state_dict` match
                the keys returned by this module's `state_dict()` function.
        """
        # Remove classifier head keys, if present
        model_dict = state_dict.get("model", {})
        model_dict.pop("head.bias", None)
        model_dict.pop("head.weight", None)

        super().load_state_dict(model_dict, strict=strict)


def load_pretrained_convnext_weights(
    encoder: ConvNeXtEncoder,
    model_name: str
) -> None:
    """
    Load timm-pretrained weights into the given ConvNeXtEncoder.

    Steps:
      1. Create a timm model with pretrained weights.
      2. Retrieve its state dict.
      3. Rename each key to have a "timm_model." prefix.
      4. Remove classifier head keys (if any).
      5. Wrap so it matches encoder.load_state_dict() signature and load it.

    Args:
        encoder (ConvNeXtEncoder): The target encoder to load weights into.
        model_name (str): A valid ConvNeXt variant recognized by timm (e.g., 'convnext_base').
    """
    # Create a timm model with pretrained weights
    pretrained_model = timm.create_model(model_name, pretrained=True)
    pretrained_dict = pretrained_model.state_dict()

    # Prefix the keys with "timm_model."
    renamed_dict = {}
    for old_key, val in pretrained_dict.items():
        new_key = f"timm_model.{old_key}"
        renamed_dict[new_key] = val

    # Remove classifier head keys, e.g., "head.bias", "head.weight"
    keys_to_remove = [k for k in renamed_dict if "head." in k]
    for k in keys_to_remove:
        del renamed_dict[k]

    # Prepare the final state dict in encoder.load_state_dict() format
    ckpt = {"model": renamed_dict}
    encoder.load_state_dict(ckpt)


# URLs for pretrained weights for different ConvNeXt variants
convnext_weights = {
    "timm-convnext_base": {
        "imagenet": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth"
    },
    "timm-convnext_large": {
        "imagenet": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth"
    },
}

# Create a structure for pretrained settings
pretrained_settings: Dict[str, Dict[str, Dict[str, Any]]] = {}
for model_key, sources in convnext_weights.items():
    pretrained_settings[model_key] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_key][source_name] = {
            "url": source_url,
            "input_size": [3, 384, 384],
            "input_range": [0.0, 1.0],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }

# Definition of timm-based ConvNeXt encoders
timm_convnext_encoders = {
    "timm-convnext_base": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["timm-convnext_base"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024),  # Example channel config
            "depths": [3, 3, 27, 3],
            "dims": [128, 256, 512, 1024],
        },
    },
    "timm-convnext_large": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["timm-convnext_large"],
        "params": {
            "out_channels": (3, 192, 384, 768, 1536),
            "depths": [3, 3, 27, 3],
            "dims": [192, 384, 768, 1536],
        },
    },
}
