from copy import deepcopy
import torch
import torch.nn as nn
from typing import Dict, Any, List
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    resnext50_32x4d, resnext101_32x8d
)
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,
    ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights
)

from ._base import EncoderMixin


class ResNetEncoder(ResNet, EncoderMixin):
    """
    A ResNet-based encoder that supports multiple depths of feature extraction
    for use in segmentation or other multi-stage tasks (e.g., U-Net).

    This class inherits from:
      - torchvision.models.ResNet as the base architecture,
      - EncoderMixin to add functionality like out_channels, adjustable input channels,
        and stage extraction.

    Attributes:
        _depth (int): Controls the number of feature maps (stages) returned in forward().
        _out_channels (List[int]): Channels of each extracted feature map, including the first conv layer.
        _in_channels (int): Number of input channels for the network (default is 3).

    Example:
        # Constructing a resnet34-based encoder
        encoder = ResNetEncoder(
            out_channels=[3, 64, 64, 128, 256, 512],
            depth=5,
            block=BasicBlock,
            layers=[3, 4, 6, 3]
        )
    """

    def __init__(
        self,
        out_channels: List[int],
        depth: int = 5,
        **kwargs: Any
    ) -> None:
        """
        Initialize the ResNetEncoder.

        Args:
            out_channels (List[int]): Number of channels for each encoder feature tensor,
                starting from the input layer to the deepest layer.
            depth (int): Number of stages to extract from the encoder. If depth=5,
                forward() returns 6 feature maps (indices 0..5).
            **kwargs: Additional arguments passed to the torchvision.models.ResNet base class
                (e.g., block, layers, groups, width_per_group).
        """
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3  # Default for standard ResNet
        # Remove unused classification layers
        del self.fc
        del self.avgpool

    def get_stages(self) -> List[nn.Module]:
        """
        Retrieve the encoder stages in a list form suitable for feature extraction.

        Returns:
            List[nn.Module]: A list of modules corresponding to encoder stages.
              Indices:
                0: identity (no operation, passes input as-is)
                1: initial conv/bn/relu layers
                2: maxpool + layer1
                3: layer2
                4: layer3
                5: layer4
        """
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass, extracting features from multiple stages for e.g. U-Net skip connections.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            List[torch.Tensor]: A list of feature maps from each stage.
                If depth=5, returns 6 feature maps (indices 0..5).
        """
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], **kwargs: Any) -> None:
        """
        Load weights into the model, ignoring the fully connected layer.

        Args:
            state_dict (Dict[str, torch.Tensor]): A dictionary of parameter name -> parameter tensor.
            **kwargs: Additional arguments to super().load_state_dict (e.g., `strict`).
        """
        # Remove classifier weights if they exist
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


# Define ResNet and ResNeXt encoders with their configurations
resnet_encoders: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                # Instead of a URL, store the actual state_dict from torchvision for simplicity
                "url": resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict(),
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict(),
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict(),
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).state_dict(),
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).state_dict(),
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1).state_dict(),
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1).state_dict(),
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
}
