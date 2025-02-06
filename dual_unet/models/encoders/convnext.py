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
from typing import List

class ConvNeXtEncoder(nn.Module):
    """
    A custom ConvNeXt encoder wrapper that:
      - Creates a ConvNeXt model via timm.
      - Removes the classification head.
      - Exposes a `get_stages()` method for U-Net style feature extraction.
      - Can load pretrained weights using `load_state_dict`.
    """

    def __init__(
        self,
        out_channels: List[int],
        depth: int = 5,
        **kwargs
    ):
        """
        Args:
            out_channels (List[int]): Channel sizes for each stage.
            depth (int): Number of stages to output. 
                         e.g. if depth=5, we return 6 feature maps (0..5).
            **kwargs: Additional args forwarded to timm's create_model (e.g., `in_chans`).
        """
        super().__init__()
        self.timm_model = timm.create_model(
            "convnext_base",  # or another variant
            pretrained=False,
            **kwargs
        )
        self.depth = depth
        self.out_channels = out_channels
        self.in_channels = kwargs.get('in_chans', 3)

        # Remove classification head if present
        del self.timm_model.head

    def get_stages(self):
        """
        Returns:
            List of modules (stages) in the ConvNeXt model, including stem.
        """
        return [
            nn.Identity(),
            nn.Sequential(self.timm_model.stem),
            self.timm_model.stages[0],
            self.timm_model.stages[1],
            self.timm_model.stages[2],
            self.timm_model.stages[3],
        ]

    def make_dilated(self, stage_list, dilation_list):
        """
        Not supported by ConvNeXt encoders. Raises ValueError.
        """
        raise ValueError("ConvNeXt encoders do not support dilated mode")

    def forward(self, x: torch.Tensor):
        """
        Forward pass through each stage, collecting outputs for U-Net skip connections.

        Args:
            x (torch.Tensor): Input of shape (N, C, H, W)

        Returns:
            List[torch.Tensor]: Feature maps from each stage (0..depth).
        """
        stages = self.get_stages()
        features = []
        for i in range(self.depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        Loads pretrained weights from a dict with the format: {'model': <actual ConvNeXt weights>}.

        Removes any classifier head keys (e.g. 'head.bias', 'head.weight') before loading.
        """
        # Pop classifier weights if present
        state_dict['model'].pop("head.bias", None)
        state_dict['model'].pop("head.weight", None)

        # Load only the model sub-dict
        super().load_state_dict(state_dict['model'], strict=strict)


def load_pretrained_convnext_weights(encoder: ConvNeXtEncoder, model_name: str) -> None:
    """
    Load timm-pretrained weights into the given ConvNeXtEncoder.

    Args:
        encoder (ConvNeXtEncoder): The custom encoder object.
        model_name (str): A valid ConvNeXt variant recognized by timm (e.g., 'convnext_base').
    """
    # 1. Create a timm model with pretrained weights
    pretrained_model = timm.create_model(model_name, pretrained=True)
    pretrained_dict = pretrained_model.state_dict()

    # 2. Rename each key to have a "timm_model." prefix
    renamed_dict = {}
    for old_key, val in pretrained_dict.items():
        new_key = f"timm_model.{old_key}"
        renamed_dict[new_key] = val

    # 3. Remove classifier head keys
    keys_to_remove = [k for k in renamed_dict if "head." in k]
    for k in keys_to_remove:
        del renamed_dict[k]

    # 4. Wrap so it matches encoder.load_state_dict signature
    ckpt = {"model": renamed_dict}
    encoder.load_state_dict(ckpt)


convnext_weights = {
    'timm-convnext_base': {
        'imagenet': 'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth'
    },
    'timm-convnext_large': {
        'imagenet': 'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth'
    },
}

pretrained_settings = {}
for model_name, sources in convnext_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_size": [3, 384, 384],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }

timm_convnext_encoders = {
    'timm-convnext_base': {
        'encoder': ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["timm-convnext_base"],
        'params': {
            'out_channels': (3, 128, 256, 512, 1024),
            'depths': [3, 3, 27, 3],
            'dims': [128, 256, 512, 1024]
        },
    },
    'timm-convnext_large': {
        'encoder': ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["timm-convnext_large"],
        'params': {
            'out_channels': (3, 192, 384, 768, 1536),
            'depths': [3, 3, 27, 3],
            'dims': [192, 384, 768, 1536]
        },
    },
}
