import copy
import numpy as np

from skimage import color
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import _get_fill, _setup_fill_arg
from typing import List, Dict, Any, Union, Optional, Tuple


def build_transforms(cfg: Dict[str, Any], split: str, is_train: bool = True) -> v2.Compose:
    """
    Build a composed set of transforms based on the given configuration.

    Args:
        cfg (dict): Configuration containing transform specs under cfg['transforms'].
        split (str): Data split identifier (e.g., "train", "test").
        is_train (bool): If True, includes augmentation transforms.

    Returns:
        v2.Compose: A composed transform that can be applied to (image, target) pairs.
    """
    transforms = [v2.ToImage()]

    if is_train:
        transforms.append(build_augmentations(cfg))

    transforms.append(v2.ToDtype(torch.float32, scale=True))

    if "rescale" in cfg["transforms"]:
        transforms.append(
            Rescale(
                cfg["transforms"]["rescale"],
                antialias=True,
                interpolation=v2.InterpolationMode.BICUBIC,
            )
        )
    if "normalize" in cfg["transforms"]:
        mean = cfg["transforms"]["normalize"]["mean"]
        std = cfg["transforms"]["normalize"]["std"]
        transforms.append(v2.Normalize(mean=mean, std=std))

    transforms.append(GaussianCentroidMask(sigma=cfg["training"]["sigma"]))

    if split == "test":
        transforms.append(
            SegmentationMaskTensorWithBackground_masks(
                num_classes=cfg["dataset"][split]["num_classes"]
            )
        )
    else:
        transforms.append(
            SegmentationMaskTensorWithBackground(
                num_classes=cfg["dataset"][split]["num_classes"]
            )
        )

    return v2.Compose(transforms)


def build_augmentations(cfg: Dict[str, Any]) -> v2.Compose:
    """
    Build a composed transform of augmentations specified under cfg['transforms']['augmentations'].

    Args:
        cfg (dict): Configuration dictionary with an "augmentations" list specifying each augmentation's name and params.

    Returns:
        v2.Compose: A composed augmentation transform.
    """
    augs = []
    assert "augmentations" in cfg["transforms"], "Missing 'augmentations' in cfg['transforms']."

    for aug in cfg["transforms"]["augmentations"]:
        assert "name" in aug, "Each augmentation must have a 'name' key."
        transform = AugmentationFactory.build(
            aug["name"], **{k: v for k, v in aug.items() if k not in ["name", "p"]}
        )
        if "p" in aug:
            transform = RandomApply([transform], p=aug["p"])
        augs.append(transform)

    return v2.Compose(augs)


class AugmentationFactory:
    """
    A simple factory to build augmentation transforms by name.
    """

    @staticmethod
    def build(name: str, **kwargs) -> nn.Module:
        """
        Build a specified augmentation transform.

        Args:
            name (str): The name of the augmentation.
            **kwargs: Additional parameters for the transform.

        Returns:
            nn.Module: A TorchVision v2 transform or custom transform object.
        """
        if name == "hflip":
            return v2.RandomHorizontalFlip(p=1.0)
        elif name == "vflip":
            return v2.RandomVerticalFlip(p=1.0)
        elif name == "rotate90":
            return RandomRotation90()
        elif name == "cjitter":
            return v2.ColorJitter(**kwargs)
        elif name == "elastic":
            return v2.ElasticTransform(**kwargs)
        elif name == "blur":
            return v2.GaussianBlur(**kwargs)
        elif name == "resizedcrop":
            return v2.RandomResizedCrop(**kwargs, antialias=True)
        elif name == "resize":
            return v2.Resize(**kwargs, antialias=True)
        elif name == "randomcrop":
            return v2.RandomCrop(**kwargs)
        elif name == "hedjitter":
            return HEDJitter(**kwargs)
        else:
            raise ValueError(f"Unknown augmentation name: {name}")


class RandomRotation90(v2.Transform):
    """
    A transform that randomly rotates an image (and associated target data)
    by one of [0, 90, -90, 180] degrees.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fill = _setup_fill_arg(0)

    def _get_params(self, flat_inputs) -> Dict[str, float]:
        angles = torch.tensor([0, 90, -90, 180], dtype=torch.float32)
        angle = angles[torch.randperm(4)[0]].item()
        return {"angle": angle}

    def _transform(self, inpt, params):
        fill = _get_fill(self._fill, type(inpt))
        return self._call_kernel(
            F.rotate,
            inpt,
            **params,
            interpolation=v2.InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=fill,
        )


class HEDJitter(nn.Module):
    """
    A custom transform that applies random perturbations in the H&E color space (HED).
    """

    def __init__(
        self,
        alpha: Union[float, Tuple[float, float]] = (0.98, 1.02),
        beta: Union[float, Tuple[float, float]] = (-0.02, 0.02)
    ):
        super().__init__()
        if not isinstance(alpha, tuple):
            alpha = (1.0 - alpha, 1.0 + alpha)
        if not isinstance(beta, tuple):
            beta = (-beta, beta)
        self.alpha = alpha
        self.beta = beta

    def forward(self, image: torch.Tensor, target: Dict[str, Any]):
        alpha_H = float(torch.empty(1).uniform_(self.alpha[0], self.alpha[1]))
        alpha_E = float(torch.empty(1).uniform_(self.alpha[0], self.alpha[1]))
        beta_H = float(torch.empty(1).uniform_(self.beta[0], self.beta[1]))
        beta_E = float(torch.empty(1).uniform_(self.beta[0], self.beta[1]))

        orig_dtype = image.dtype
        image = F.convert_image_dtype(image, torch.float32)

        # Convert to numpy for skimage color conversion
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_hed = color.rgb2hed(img_np)
        img_hed[..., 0] = img_hed[..., 0] * alpha_H + beta_H
        img_hed[..., 1] = img_hed[..., 1] * alpha_E + beta_E
        # If D channel also needed, uncomment and apply alpha_D, beta_D

        img_rgb = color.hed2rgb(img_hed)
        image = torch.tensor(img_rgb).permute(2, 0, 1)
        image = F.convert_image_dtype(image, orig_dtype)

        return image, target


class GaussianCentroidMask(nn.Module):
    """
    Convert bounding boxes in the target into a Gaussian centroid mask.
    """

    def __init__(self, sigma: float = 5.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, image: torch.Tensor, target: Dict[str, Any]):
        """
        Generate a Gaussian centroid mask from the bounding boxes in the target.

        Args:
            image (torch.Tensor): Image tensor of shape (C, H, W).
            target (dict): Target dictionary containing at least 'boxes'.

        Returns:
            (torch.Tensor, dict): The unmodified image and the updated target with 'centroid_gaussian'.
        """
        height, width = image.shape[-2], image.shape[-1]
        gauss_mask, _ = self.generate_gaussian_masks(target["boxes"], height, width)
        target["centroid_gaussian"] = gauss_mask
        return image, target

    def generate_gaussian_masks(
        self, 
        boxes: torch.Tensor, 
        height: int, 
        width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Gaussian masks from bounding box centroids.

        Args:
            boxes (torch.Tensor): (N, 4) bounding boxes [xmin, ymin, xmax, ymax].
            height (int): Image height.
            width (int): Image width.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                1) A (1, H, W) Gaussian mask over centroids.
                2) A centroid mask (same shape) with 1 at centroid pixels.
        """
        mask = torch.zeros((1, height, width), dtype=torch.float32)
        for box in boxes:
            centroid_x = ((box[2] - box[0]) / 2) + box[0]
            centroid_y = ((box[3] - box[1]) / 2) + box[1]
            centroid_x = min(max(centroid_x, 0), width - 1)
            centroid_y = min(max(centroid_y, 0), height - 1)
            mask[0, int(centroid_y), int(centroid_x)] += 1.0

        centroids_mask = mask.clone()
        filtered_mask = gaussian_filter(mask.numpy(), sigma=self.sigma)
        filtered_mask = torch.tensor(filtered_mask, dtype=torch.float32)
        if filtered_mask.max() > 0:
            filtered_mask /= filtered_mask.max()

        return filtered_mask, centroids_mask


class SegmentationMaskTensorWithBackground(nn.Module):
    """
    Convert binary instance masks (one per object) into a multi-channel segmentation tensor,
    including background as the first channel.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, image: torch.Tensor, target: Dict[str, Any]):
        """
        Build a (C+1, H, W) segmentation mask from `masks` and `labels`.

        Args:
            image (torch.Tensor): The input image.
            target (dict): Target containing 'masks' (List of binary masks) and 'labels' (List of int class IDs).

        Returns:
            Tuple[torch.Tensor, dict]: The unmodified image and the target with 'segmentation_mask' added.
        """
        height, width = image.shape[-2], image.shape[-1]
        seg_mask = self.build_segmentation_tensor(target["masks"], target["labels"], height, width)
        target["segmentation_mask"] = seg_mask
        target.pop("masks", None)
        return image, target

    def build_segmentation_tensor(
        self, 
        masks: List[torch.Tensor], 
        labels: List[int],
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Create a multi-channel segmentation tensor with shape (C+1, H, W). 
        The first channel is background.

        Args:
            masks (list): List of binary masks (H, W).
            labels (list): Corresponding class labels (1..num_classes).
            height (int): Image height.
            width (int): Image width.

        Returns:
            torch.Tensor: The segmentation tensor with shape (num_classes+1, H, W).
        """
        seg_tensor = torch.zeros((self.num_classes + 1, height, width), dtype=torch.float32)
        for i, label in enumerate(labels):
            seg_tensor[label] += masks[i]
        seg_tensor[0] = 1.0 - seg_tensor[1:].sum(dim=0).clamp(max=1.0)
        return seg_tensor


class SegmentationMaskTensorWithBackground_masks(nn.Module):
    """
    Similar to SegmentationMaskTensorWithBackground, but does not remove 'masks' from the target.
    Useful if we need the original masks downstream.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, image: torch.Tensor, target: Dict[str, Any]):
        height, width = image.shape[-2], image.shape[-1]
        seg_mask = self.build_segmentation_tensor(target["masks"], target["labels"], height, width)
        target["segmentation_mask"] = seg_mask
        return image, target

    def build_segmentation_tensor(
        self, 
        masks: List[torch.Tensor], 
        labels: List[int],
        height: int,
        width: int
    ) -> torch.Tensor:
        seg_tensor = torch.zeros((self.num_classes + 1, height, width), dtype=torch.float32)
        for i, label in enumerate(labels):
            seg_tensor[label] += masks[i]
        seg_tensor[0] = 1.0 - seg_tensor[1:].sum(dim=0).clamp(max=1.0)
        return seg_tensor


class RandomApply(v2.Transform):
    """
    Randomly apply a sequence of transforms with probability p.
    """

    def __init__(self, transforms: List[nn.Module], p: float = 0.5) -> None:
        super().__init__()
        if not isinstance(transforms, (list, nn.ModuleList)):
            raise TypeError("Argument transforms should be a list or nn.ModuleList of callables.")
        self.transforms = transforms
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` must be in the interval [0.0, 1.0].")
        self.p = p

    def _extract_params_for_v1_transform(self):
        return {"transforms": self.transforms, "p": self.p}

    def forward(self, *inputs):
        needs_unpacking = len(inputs) > 1
        if torch.rand(1) >= self.p:
            return inputs if needs_unpacking else inputs[0]

        for tform in self.transforms:
            outputs = tform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)
        return outputs

    def extra_repr(self) -> str:
        lines = [f"    {t}" for t in self.transforms]
        return "\n".join(lines)


class Rescale(v2.Transform):
    """
    Rescales the image (and potentially the target) by a factor, with optional max_size constraint.
    """

    def __init__(
        self,
        scale: float,
        max_size: Optional[int] = None,
        interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
        antialias: bool = True
    ) -> None:
        super().__init__()
        self.scale = scale
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_params(self, flat_inputs) -> Dict[str, int]:
        # The first element is assumed to be the image
        h, w = flat_inputs[0].shape[-2:]
        size_val = int(self.scale * min(h, w))
        if self.max_size is not None:
            size_val = min(size_val, self.max_size)
        return {"size": size_val}

    def _transform(self, inpt: torch.Tensor, params: Dict[str, int]):
        return self._call_kernel(
            F.resize,
            inpt,
            params["size"],
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias,
        )