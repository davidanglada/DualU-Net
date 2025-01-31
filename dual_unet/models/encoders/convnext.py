import timm
import torch
import torch.nn as nn
from typing import List

# -------------------------------------------------------------------------
# 1. Your custom encoder (as you provided)
# -------------------------------------------------------------------------
class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        out_channels: List[int],
        depth: int = 5,
        **kwargs
    ):
        """
        out_channels: list of channel sizes at each stage
        depth: how many stages to extract (0..depth inclusive => depth+1 stages)
        
        kwargs: forwarded to ConvNeXt parent constructor. 
                Typically includes config like in_chans, depths, dims, etc.
        """
        super().__init__()

        # ---------------------------
        # Instead of inheriting directly from `ConvNeXt`,
        # create a timm model inside your class
        # ---------------------------
        self.timm_model = timm.create_model(
            'convnext_base',  # or another variant
            pretrained=False,  # we'll load weights manually
            **kwargs
        )
        self.depth = depth
        self.out_channels = out_channels
        self.in_channels = kwargs.get('in_chans', 3)

        # remove classification head
        del self.timm_model.head

    def get_stages(self):
        # The timm convnext model typically has:
        # stem --> stages[0] --> stages[1] --> stages[2] --> stages[3]
        return [
            nn.Identity(),
            nn.Sequential(self.timm_model.stem),
            self.timm_model.stages[0],
            self.timm_model.stages[1],
            self.timm_model.stages[2],
            self.timm_model.stages[3],
        ]

    def make_dilated(self, stage_list, dilation_list):
        # Not supported by convnext
        raise ValueError("ConvNeXt encoders do not support dilated mode")

    def forward(self, x):
        # Go through each stage in succession, collecting features
        stages = self.get_stages()
        features = []
        for i in range(self.depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        This expects a dictionary like:
           {'model': <actual ConvNeXt weights>}
        where the classifier weights ("head.bias" / "head.weight") 
        have not been removed yet or can be removed in this method.
        """
        # pop classifier weights if present
        state_dict['model'].pop("head.bias",  None)
        state_dict['model'].pop("head.weight", None)

        # Now load only the "model" part into our timm_model
        super().load_state_dict(state_dict['model'], strict=strict)


# -------------------------------------------------------------------------
# 2. Helper function to load pretrained timm weights
# -------------------------------------------------------------------------
def load_pretrained_convnext_weights(encoder: ConvNeXtEncoder, model_name: str):
    """
    Load timm pretrained weights from `model_name` 
    (e.g., 'convnext_tiny') into your custom encoder.
    """
    # Create a timm model with pretrained weights
    pretrained_model = timm.create_model(model_name, pretrained=True)
    pretrained_dict = pretrained_model.state_dict()

     # 2. Rename each key to have a "timm_model." prefix
    renamed_dict = {}
    for old_key, val in pretrained_dict.items():
        new_key = f"timm_model.{old_key}"
        renamed_dict[new_key] = val

    # 3. Remove classifier head if desired
    #    e.g. keys that contain "head."
    keys_to_remove = [k for k in renamed_dict if "head." in k]
    for k in keys_to_remove:
        del renamed_dict[k]

    # Wrap in a dict so it matches your encoder's load_state_dict signature
    ckpt = {"model": renamed_dict}

    # Load into your custom encoder, which removes the head automatically
    encoder.load_state_dict(ckpt)


convnext_weights = {
    # 'timm-convnext_tiny': {
    #     'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-convnext/convnext_tiny_1k_224_ema.pth'
    # },
    # 'timm-convnext_small': {
    #     'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-convnext/convnext_small_1k_224_ema.pth'
    # },
    'timm-convnext_base': {
        'imagenet': 'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth'
    },
    'timm-convnext_large': {
        'imagenet': 'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth'
    }
}

pretrained_settings = {}
for model_name, sources in convnext_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 384, 384],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }


timm_convnext_encoders = {
    # 'timm-convnext_tiny': {
    #     'encoder': ConvNeXtEncoder,
    #     "pretrained_settings": pretrained_settings["timm-convnext_tiny"],
    #     'params': {
    #         'out_channels': (3, 96, 192, 384, 768),
    #         'depths': [3, 3, 9, 3],
    #         'dims': [96, 192, 384, 768]
    #     },
    # },
    # 'timm-convnext_small': {
    #     'encoder': ConvNeXtEncoder,
    #     "pretrained_settings": pretrained_settings["timm-convnext_small"],
    #     'params': {
    #         'out_channels': (3, 96, 192, 384, 768),
    #         'depths': [3, 3, 27, 3],
    #         'dims': [96, 192, 384, 768]
    #     },
    # },
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
    }
}
