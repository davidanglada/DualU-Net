import functools
import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders
from .convnext import timm_convnext_encoders
from .convnext import load_pretrained_convnext_weights, ConvNeXtEncoder
from ._preprocessing import preprocess_input

encoders = {}
encoders.update(resnet_encoders)
encoders.update(timm_convnext_encoders)

def get_encoder(name, in_channels=3, depth=5, weights=None):

    if 'convnext' in name:
        out_channels = [3, 128, 128, 256, 512, 1024]

        # Create your encoder
        encoder = ConvNeXtEncoder(out_channels=out_channels, depth=5, in_chans=3)

        # Load pretrained weights
        load_pretrained_convnext_weights(encoder, 'convnext_base')

        return encoder

    else:

        try:
            Encoder = encoders[name]["encoder"]
        except KeyError:
            raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

        params = encoders[name]["params"]
        params.update(depth=depth)
        encoder = Encoder(**params)

        if weights is not None:
            try:
                settings = encoders[name]["pretrained_settings"][weights]
            except KeyError:
                raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights, name, list(encoders[name]["pretrained_settings"].keys()),
                ))
            encoder.load_state_dict(model_zoo.load_url(settings["url"]))

        encoder.set_in_channels(in_channels)

        return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):
    settings = encoders[encoder_name]["pretrained_settings"]

    if pretrained not in settings.keys():
        raise ValueError("Available pretrained options {}".format(settings.keys()))

    formatted_settings = {}
    formatted_settings["input_space"] = settings[pretrained].get("input_space")
    formatted_settings["input_range"] = settings[pretrained].get("input_range")
    formatted_settings["mean"] = settings[pretrained].get("mean")
    formatted_settings["std"] = settings[pretrained].get("std")
    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
