# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

import functools
import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders
from .convnext import timm_convnext_encoders
from .convnext import load_pretrained_convnext_weights, ConvNeXtEncoder
from ._preprocessing import preprocess_input

encoders = {}
encoders.update(resnet_encoders)
encoders.update(timm_convnext_encoders)


def get_encoder(encoder_name: str, in_channels: int = 3, depth: int = 5, weights: str = None):
    """
    Retrieve an encoder (e.g., ResNet/ConvNeXt) by name with optional pretrained weights.

    Args:
        name (str): Encoder name, must be a key in `encoders` dict (e.g. "resnet34", "convnext_base").
        in_channels (int): Number of input channels. Defaults to 3 for RGB.
        depth (int): Depth of the encoder in [3..5]. Controls how many stages are actually used.
        weights (str, optional): Pretrained weights identifier, e.g. "imagenet". If None, random init.

    Returns:
        encoder (nn.Module): The corresponding encoder with specified configuration.
    """

    # Special handling for convnext encoders
    if 'convnext' in encoder_name:
        out_channels = [3, 128, 128, 256, 512, 1024]
        encoder = ConvNeXtEncoder(out_channels=out_channels, depth=5, in_chans=3)
        load_pretrained_convnext_weights(encoder, 'convnext_base')  # Hard-coded for this example
        return encoder

    else:
        try:
            Encoder = encoders[encoder_name]["encoder"]
        except KeyError:
            raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(encoder_name, list(encoders.keys())))

        params = encoders[encoder_name]["params"]
        params.update(depth=depth)
        encoder = Encoder(**params)

        if weights is not None:
            try:
                settings = encoders[encoder_name]["pretrained_settings"][weights]
            except KeyError:
                raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights, encoder_name, list(encoders[encoder_name]["pretrained_settings"].keys()),
                ))
            if isinstance(settings["url"], str):
                state_dict = model_zoo.load_url(settings["url"])
            else:
                state_dict = settings["url"]
            encoder.load_state_dict(state_dict)

        encoder.set_in_channels(in_channels)

        return encoder


def get_encoder_names():
    """
    Return a list of all available encoder names.
    """
    return list(encoders.keys())


def get_preprocessing_params(encoder_name: str, pretrained: str = "imagenet"):
    """
    Retrieve preprocessing parameters (mean, std, etc.) for a given encoder and weight name.

    Args:
        encoder_name (str): Name of the encoder.
        pretrained (str): Name of the pretrained weights, e.g. "imagenet".

    Returns:
        dict: Contains "input_space", "input_range", "mean", "std".
    """
    if encoder_name not in encoders:
        raise ValueError(
            f"Encoder `{encoder_name}` not found. Available: {list(encoders.keys())}"
        )

    settings = encoders[encoder_name]["pretrained_settings"]
    if pretrained not in settings:
        raise ValueError(f"Available pretrained options: {list(settings.keys())}")

    # Extract relevant fields
    return {
        "input_space": settings[pretrained].get("input_space"),
        "input_range": settings[pretrained].get("input_range"),
        "mean": settings[pretrained].get("mean"),
        "std": settings[pretrained].get("std"),
    }


def get_preprocessing_fn(encoder_name: str, pretrained: str = "imagenet"):
    """
    Returns a preprocessing function that can normalize input images
    according to the encoder's mean/std or other params.

    Args:
        encoder_name (str): Name of the encoder.
        pretrained (str): Pretrained weights identifier (e.g., "imagenet").

    Returns:
        callable: A function that applies the appropriate preprocessing.
    """
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
