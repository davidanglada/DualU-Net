from copy import deepcopy
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights
from ._base import EncoderMixin


class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels: list[int], depth: int = 5, **kwargs):
        """
        Initialize the ResNetEncoder.

        Args:
            out_channels (list of int): Number of channels for each encoder feature tensor.
            depth (int): Number of stages in the encoder.
            **kwargs: Additional arguments for the ResNet base class.
        """
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        # Remove fully connected layer and average pooling layer
        del self.fc
        del self.avgpool

    def get_stages(self) -> list[nn.Module]:
        """
        Get the stages of the ResNet model.

        Returns:
            list of nn.Module: List of stages in the ResNet model.
        """
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            list of torch.Tensor: List of feature tensors of different spatial resolutions.
        """
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict: dict, **kwargs):
        """
        Load state dictionary into the model.

        Args:
            state_dict (dict): State dictionary to load.
            **kwargs: Additional arguments for the base class method.
        """
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


# Define ResNet encoders with their parameters
resnet_encoders = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": resnet18(weights=ResNet18_Weights.DEFAULT).state_dict(),
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
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
                "url": resnet34(weights=ResNet34_Weights.DEFAULT).state_dict(),
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
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
                "url": resnet50(weights=ResNet50_Weights.DEFAULT).state_dict(),
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
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
                "url": resnet101(weights=ResNet101_Weights.DEFAULT).state_dict(),
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
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
                "url": resnet152(weights=ResNet152_Weights.DEFAULT).state_dict(),
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
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
                "url": resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT).state_dict(),
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
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
                "url": resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT).state_dict(),
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
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
