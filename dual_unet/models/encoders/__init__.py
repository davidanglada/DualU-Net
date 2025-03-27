# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

import functools
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders
from .convnext import timm_convnext_encoders, load_pretrained_convnext_weights, ConvNeXtEncoder
from ._preprocessing import preprocess_input

encoders: Dict[str, Dict[str, Any]] = {}
encoders.update(resnet_encoders)
encoders.update(timm_convnext_encoders)


def get_encoder(
    encoder_name: str,
    in_channels: int = 3,
    depth: int = 5,
    weights: Optional[str] = None
) -> torch.nn.Module:
    """
    Retrieve an encoder (e.g., ResNet or ConvNeXt) by name with optional pretrained weights.

    Args:
        encoder_name (str): Encoder name. Must be a key in the global `encoders` dict
            (e.g. "resnet34", "convnext_base").
        in_channels (int): Number of input channels. Defaults to 3 for RGB.
        depth (int): Depth of the encoder in [3..5]. Controls how many stages are actually used.
        weights (Optional[str]): Identifier for pretrained weights (e.g. "imagenet").
            If None, the encoder is randomly initialized.

    Returns:
        torch.nn.Module: The encoder module configured with the specified parameters.

    Raises:
        KeyError: If the specified encoder name is invalid, or if the specified weights are
            not available for the given encoder.

    Example:
        >>> encoder = get_encoder("resnet34", in_channels=1, weights="imagenet")
        >>> print(encoder)
    """
    # Special handling for ConvNeXt encoders
    if "convnext" in encoder_name:
        # Hard-coded out_channels for an example ConvNeXt configuration
        out_channels = [3, 128, 128, 256, 512, 1024]
        encoder = ConvNeXtEncoder(out_channels=out_channels, depth=5, in_chans=3)

        # For example, we load "convnext_base" pretrained weights by default
        load_pretrained_convnext_weights(encoder, "convnext_base")
        return encoder

    else:
        # Lookup the encoder entry
        if encoder_name not in encoders:
            raise KeyError(
                f"Wrong encoder name `{encoder_name}`, supported encoders: {list(encoders.keys())}"
            )

        Encoder = encoders[encoder_name]["encoder"]
        params: Dict[str, Any] = encoders[encoder_name]["params"]
        params.update(depth=depth)
        encoder = Encoder(**params)

        if weights is not None:
            # Retrieve pretrained settings
            pretrained_settings = encoders[encoder_name].get("pretrained_settings", {})
            if weights not in pretrained_settings:
                raise KeyError(
                    f"Wrong pretrained weights `{weights}` for encoder `{encoder_name}`. "
                    f"Available options are: {list(pretrained_settings.keys())}"
                )

            settings = pretrained_settings[weights]
            url = settings.get("url", None)

            # If the URL is a string, load via model_zoo
            if isinstance(url, str):
                state_dict = model_zoo.load_url(url)
            else:
                # Possibly the state_dict is directly provided
                state_dict = url

            encoder.load_state_dict(state_dict)

        # Adjust encoder input channels if needed (e.g. to 1 for grayscale)
        encoder.set_in_channels(in_channels)

        return encoder


def get_encoder_names() -> List[str]:
    """
    Return a list of all available encoder names, as keys in the `encoders` dictionary.

    Returns:
        List[str]: A list of supported encoder names.

    Example:
        >>> names = get_encoder_names()
        >>> print(names)
        ["resnet18", "resnet34", "convnext_base", ...]
    """
    return list(encoders.keys())


def get_preprocessing_params(
    encoder_name: str,
    pretrained: str = "imagenet"
) -> Dict[str, Any]:
    """
    Retrieve preprocessing parameters (mean, std, etc.) for a given encoder and its weight name.

    Args:
        encoder_name (str): Name of the encoder, must be a key in `encoders`.
        pretrained (str): Name of the pretrained weights (e.g., "imagenet").

    Returns:
        Dict[str, Any]: A dictionary containing "input_space", "input_range", "mean", and "std".

    Raises:
        ValueError: If the encoder is not found, or if the given pretrained weights aren't available.
    """
    if encoder_name not in encoders:
        raise ValueError(
            f"Encoder `{encoder_name}` not found. Available: {list(encoders.keys())}"
        )

    settings = encoders[encoder_name].get("pretrained_settings", {})
    if pretrained not in settings:
        raise ValueError(f"Available pretrained options: {list(settings.keys())}")

    # Extract relevant fields
    s = settings[pretrained]
    return {
        "input_space": s.get("input_space", "RGB"),
        "input_range": s.get("input_range", [0.0, 1.0]),
        "mean": s.get("mean", [0.485, 0.456, 0.406]),
        "std": s.get("std", [0.229, 0.224, 0.225]),
    }


def get_preprocessing_fn(
    encoder_name: str,
    pretrained: str = "imagenet"
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a preprocessing function configured according to the encoder's preprocessing parameters.

    Args:
        encoder_name (str): Name of the encoder, must be a key in `encoders`.
        pretrained (str): Pretrained weights identifier (e.g., "imagenet").

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: A function that takes a torch.Tensor (image)
            and returns a processed tensor (normalized according to the encoder's mean/std, etc.).

    Example:
        >>> preprocess_fn = get_preprocessing_fn("resnet34", pretrained="imagenet")
        >>> # Then use preprocess_fn on input images/tensors
        >>> input_tensor = torch.rand(1, 3, 224, 224)
        >>> processed = preprocess_fn(input_tensor)
    """
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    # functools.partial allows us to create a function with fixed arguments.
    return functools.partial(preprocess_input, **params)
