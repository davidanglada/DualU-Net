import torch
from .mtunet import DualUNet, DualUNet_ConvNext
from .losses import DualLoss_combined

def build_model(config: dict) -> torch.nn.Module:
    """
    Initialize a dual U-Net model (for segmentation + counting) based on the provided config.

    This function supports two types of encoders:
      1. ConvNeXt-based (if "convnext" in config["model"]["encoder_name"]),
      2. Other encoders (ResNet family, etc.) that use the standard DualUNet class.

    Args:
        config (dict): Configuration dictionary. Must contain "model" sub-dict with keys:
            - "encoder_name" (str): Name of the encoder (e.g., "resnet34", "convnext_base").
            - "encoder_weights" (str or None): Pretrained weights for the encoder (e.g. "imagenet").
            - "decoder_channels" (list of int): Channels for each decoder stage (top to bottom).
            - "decoder_use_batchnorm" (bool): If True, use BatchNorm in decoder.
            - "classes_s" (int): Number of segmentation classes (the final segmentation head channels).
            - "activation_s" (str or callable or None, optional): Activation for the segmentation head.
            - "activation_c" (str or callable or None, optional): Activation for the count head.
            - "aux_params" (dict, optional): Aux classification head parameters if needed.

    Returns:
        torch.nn.Module: A configured DualUNet or DualUNet_ConvNext model.
    """
    # Identify whether to use a ConvNeXt-based model or standard ResNet-based model
    if "convnext" in config["model"]["encoder_name"].lower():
        # Build ConvNeXt-based dual U-Net
        model = DualUNet_ConvNext(
            encoder_name=config["model"]["encoder_name"],
            classes_s=config["model"]["classes_s"] + 1,  # +1 for segmentation classes (often background + classes)
            classes_c=1,  # Single channel for centroid/density
            activation_s=None,  # Typically no final activation or as configured
            activation_c=config["model"].get("activation_c", None),
            encoder_weights=config["model"]["encoder_weights"],
            decoder_channels=config["model"]["decoder_channels"],
            decoder_use_batchnorm=config["model"]["decoder_use_batchnorm"],
            aux_params=config["model"].get("aux_params", None),
        )
    else:
        # Build standard (ResNet or other) dual U-Net
        model = DualUNet(
            encoder_name=config["model"]["encoder_name"],
            classes_s=config["model"]["classes_s"] + 1,
            classes_c=1,
            activation_s=None,
            activation_c=config["model"].get("activation_c", None),
            encoder_weights=config["model"]["encoder_weights"],
            decoder_channels=config["model"]["decoder_channels"],
            decoder_use_batchnorm=config["model"]["decoder_use_batchnorm"],
            aux_params=config["model"].get("aux_params", None),
        )

    return model


def load_state_dict(cfg: dict, model: torch.nn.Module) -> None:
    """
    Load model weights from a checkpoint file specified in the config.

    The config is expected to have:
      cfg["model"]["checkpoint"] -> path to the checkpoint file.

    Args:
        cfg (dict): Configuration dictionary containing the checkpoint path under "model" sub-dict.
        model (torch.nn.Module): The model into which the state dictionary will be loaded.
    """
    import torch

    checkpoint_path = cfg["model"].get("checkpoint")
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # If checkpoint is wrapped in a dict with 'model' key, unwrap it
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        # Load state dict into the model
        model.load_state_dict(checkpoint, strict=False)
        # Note: Setting strict=False allows partial load of the checkpoint 
        # if some layers don't match (e.g. different heads).
