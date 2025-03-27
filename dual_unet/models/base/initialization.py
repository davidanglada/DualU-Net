import torch
import torch.nn as nn


def initialize_decoder(module: nn.Module) -> None:
    """
    Initialize the parameters of a decoder module.

    This function iterates over each sub-module in the given decoder module
    and applies the following initialization strategies:
      - For nn.Conv2d: Kaiming uniform initialization (fan_in, relu) for weights,
        constant 0 for biases if present.
      - For nn.BatchNorm2d: Constant 1 for weights, constant 0 for biases.
      - For nn.Linear: Xavier uniform initialization for weights, constant 0 for biases.

    Args:
        module (nn.Module): The decoder module whose sub-modules will be initialized.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module: nn.Module) -> None:
    """
    Initialize the parameters of a head module.

    This function iterates over each sub-module in the given head module
    and applies the following initialization strategy:
      - For nn.Linear and nn.Conv2d: Xavier uniform initialization for weights,
        constant 0 for biases if present.

    Args:
        module (nn.Module): The head module whose sub-modules will be initialized.
    """
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
