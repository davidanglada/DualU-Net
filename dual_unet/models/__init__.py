from .mtunet import DualUNet, DualUNetConvNext
from .losses import DualLoss_combined

def build_model(config):
    """Initialize the MTUnet model based on the config file."""

    if "convnext" in config["model"]["encoder_name"]:
        model = DualUNetConvNext(
            encoder_name=config["model"]["encoder_name"],
            classes_s=config["model"]["classes_s"]+1,  # Segmentation classes
            classes_c=1,
            encoder_weights=config["model"]["encoder_weights"],
            decoder_channels=config["model"]["decoder_channels"],
            decoder_use_batchnorm=config["model"]["decoder_use_batchnorm"],
            aux_params=config["model"].get("aux_params", None),
        )
        # model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        return model

    else:

        model = DualUNet(
            encoder_name=config["model"]["encoder_name"],
            classes_s=config["model"]["classes_s"]+1,  # Segmentation classes
            classes_c=1,  # Centroid regression classes
            encoder_weights=config["model"]["encoder_weights"],
            decoder_channels=config["model"]["decoder_channels"],
            decoder_use_batchnorm=config["model"]["decoder_use_batchnorm"],
            aux_params=config["model"].get("aux_params", None),
        )
        # model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        return model

def load_state_dict(cfg, model):
    import torch
    
    if 'checkpoint' in cfg['model']:
        # load file
        checkpoint = torch.load(cfg['model']['checkpoint'], map_location='cpu')

        # get model, if checkpoint is a dict
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        # get number of keys in checkpoint
        num_ckpt_keys = len(checkpoint)

        model.load_state_dict(checkpoint, strict=False)

        # load checkpoint of the entire model
        # if cfg.model.name == 'deformable_detr':
        #     print("Loading checkpoint for deformable DETR...")
        #     from .deformable_detr import load_sd_deformable_detr

        #     # if we have a checkpoint for backbone, remove backbone now
        #     if cfg.model.backbone.has('checkpoint'):
        #         # remove backbone from checkpoint
        #         print("\t removing backbone from checkpoint...")
        #         checkpoint = {k: v for k, v in checkpoint.items() if 'backbone' not in k}

        #         # track number of keys removed
        #         print(f"\t {num_ckpt_keys - len(checkpoint)} keys removed from checkpoint...")
        #         num_ckpt_keys = len(checkpoint)

        #         # remove neck if number of levels and channels are different
        #         model_levels = model.num_feature_levels
        #         ckpt_levels  = len([k for k in checkpoint.keys() if k.startswith('input_proj') and k.endswith('0.weight')])
        #         model_channels = [model.input_proj[i][0].in_channels for i in range(model_levels)]
        #         ckpt_channels  = [checkpoint["input_proj.{}.0.weight".format(i)].size(1) for i in range(ckpt_levels)]
        #         if model_levels != ckpt_levels or not all([m==c for m,c in zip(model_channels, ckpt_channels)]):
        #             print("\t removing neck from checkpoint...")
        #             checkpoint = {k: v for k, v in checkpoint.items() if 'input_proj' not in k}

        #             print(f"\t {num_ckpt_keys - len(checkpoint)} keys removed from checkpoint...")
        #             num_ckpt_keys = len(checkpoint)
                
        #     load_sd_deformable_detr(model, checkpoint)
        # else:
        #     raise NotImplementedError(f"Model {cfg.model.name} not implemented")