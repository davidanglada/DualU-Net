import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as F

from skimage import color
from scipy.ndimage import gaussian_filter
from typing import Any, Dict, List, Optional, Tuple, Union


def build_transforms(cfg: Dict[str, Any], split: str, is_train: bool = True) -> v2.Compose:
    """
    Build a sequence of image and target transformations based on the configuration.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing transform parameters.
        split (str): The dataset split, e.g., 'train', 'val', 'test'.
        is_train (bool): Whether the transformations are for training or not.

    Returns:
        v2.Compose: A composed torchvision v2 transformation that processes both image and target.
    """
    transforms = [v2.ToImage()]

    # Augmentation transforms for training
    if is_train:
        transforms.append(build_augmentations(cfg))

    # Convert image to float32 and scale from [0,255] to [0,1]
    transforms.append(v2.ToDtype(torch.float32, scale=True))

    # Optional rescaling
    if 'rescale' in cfg['transforms']:
        transforms.append(
            Rescale(
                cfg['transforms']['rescale'],
                antialias=True,
                interpolation=v2.InterpolationMode.BICUBIC
            )
        )

    # Optional image normalization
    if 'normalize' in cfg['transforms']:
        mean = cfg['transforms']['normalize']['mean']
        std = cfg['transforms']['normalize']['std']
        transforms.append(v2.Normalize(mean=mean, std=std))

    # Append GaussianCentroidMask transform
    if cfg['dataset'][split]['name'] == 'image_only':
        return v2.Compose(transforms)
    
    else:
        transforms.append(GaussianCentroidMask(sigma=cfg['training']['sigma']))

        # Depending on split, choose which segmentation mask transform to apply
        if split == 'test':
            transforms.append(
                SegmentationMaskTensorWithBackground_masks(
                    num_classes=cfg['dataset'][split]['num_classes']
                )
            )
        else:
            transforms.append(
                SegmentationMaskTensorWithBackground(
                    num_classes=cfg['dataset'][split]['num_classes']
                )
            )

        return v2.Compose(transforms)


def build_augmentations(cfg: Dict[str, Any]) -> v2.Compose:
    """
    Build a composed set of augmentations from the configuration.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing a 'transforms' key
                              with an 'augmentations' list.

    Returns:
        v2.Compose: A composed transform of specified augmentations.
    """
    augs = []
    assert 'augmentations' in cfg['transforms'], "No augmentations found in config."

    for aug in cfg['transforms']['augmentations']:
        assert 'name' in aug, "Augmentation must have a 'name' key."
        transform = AugmentationFactory.build(
            aug['name'],
            **{k: v for k, v in aug.items() if k not in ['name', 'p']}
        )
        # If probability p is specified, wrap in RandomApply
        if 'p' in aug:
            transform = RandomApply([transform], p=aug['p'])
        augs.append(transform)

    return v2.Compose(augs)


class AugmentationFactory:
    """
    A simple factory for creating augmentation transforms from a name key.
    """

    @staticmethod
    def build(name: str, **kwargs: Any) -> v2.Transform:
        """
        Build a transform instance based on the provided name.

        Args:
            name (str): The transform's name (e.g., "hflip", "vflip", "rotate90").
            **kwargs (Any): Additional arguments for the transform.

        Returns:
            v2.Transform: The corresponding transform.
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
            raise ValueError(f'Unknown augmentation: {name}')


class RandomRotation90(v2.Transform):
    """
    A transform that randomly rotates an image by 0, 90, -90, or 180 degrees.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fill = (0,)

    def make_params(self, flat_inputs: List[torch.Tensor]) -> Dict[str, float]:
        """
        Randomly pick one angle from [0, 90, -90, 180].

        Args:
            flat_inputs (List[torch.Tensor]): Flattened image inputs.

        Returns:
            Dict[str, float]: Dictionary containing the chosen angle.
        """
        angles = torch.tensor([0, 90, -90, 180])
        angle = angles[torch.randperm(4)[0]].item()
        return dict(angle=angle)

    def transform(self, inpt: Any, params: Dict[str, float]) -> Any:
        """
        Apply the random rotation to the input.

        Args:
            inpt (Any): Image or target to transform.
            params (Dict[str, float]): The parameters dict with the chosen angle.

        Returns:
            Any: Rotated image or transformed target.
        """
        fill = v2._utils._get_fill(self._fill, type(inpt))
        return self._call_kernel(
            F.rotate,
            inpt,
            params['angle'],
            interpolation=v2.InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=fill
        )


class HEDJitter(nn.Module):
    """
    A custom color jitter transform that operates in HED space.
    (Hematoxylin, Eosin, DAB color space).
    """

    def __init__(
        self,
        alpha: Union[float, Tuple[float, float]] = (0.98, 1.02),
        beta: Union[float, Tuple[float, float]] = (-0.02, 0.02)
    ):
        """
        Args:
            alpha (Union[float, Tuple[float, float]]): Multiplicative factor range for H&E channels.
            beta (Union[float, Tuple[float, float]]): Additive factor range for H&E channels.
        """
        super().__init__()
        if not isinstance(alpha, tuple):
            alpha = (1.0 - alpha, 1.0 + alpha)
        if not isinstance(beta, tuple):
            beta = (-beta, beta)
        self.alpha = alpha
        self.beta = beta

        self.rgb_from_hed = torch.tensor(
            [[0.65, 0.70, 0.29],
             [0.07, 0.99, 0.11],
             [0.27, 0.57, 0.78]],
            dtype=torch.float32,
            requires_grad=False
        )
        self.hed_from_rgb = torch.inverse(self.rgb_from_hed)

    def forward(
        self,
        image: torch.Tensor,
        target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply HED color jitter to the input image.

        Args:
            image (torch.Tensor): Image tensor in [C,H,W] format.
            target (Dict[str, Any]): Target dict (not modified).

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The jittered image and the unmodified target.
        """
        alpha_H = torch.empty(1).uniform_(self.alpha[0], self.alpha[1]).item()
        alpha_E = torch.empty(1).uniform_(self.alpha[0], self.alpha[1]).item()

        beta_H = torch.empty(1).uniform_(self.beta[0], self.beta[1]).item()
        beta_E = torch.empty(1).uniform_(self.beta[0], self.beta[1]).item()

        orig_dtype = image.dtype
        image = F.convert_image_dtype(image, torch.float32)

        # Convert to HED space
        img_np = image.permute(1, 2, 0).numpy()
        img_hed = color.rgb2hed(img_np)

        # Apply multiplicative/additive factors
        img_hed[..., 0] = img_hed[..., 0] * alpha_H + beta_H
        img_hed[..., 1] = img_hed[..., 1] * alpha_E + beta_E

        # Convert back to RGB
        image_rgb = torch.tensor(color.hed2rgb(img_hed)).permute(2, 0, 1)
        image_rgb = F.convert_image_dtype(image_rgb, orig_dtype)

        return image_rgb, target


class GaussianCentroidMask(nn.Module):
    """
    A transform that creates a Gaussian mask based on the centroids of bounding boxes in the target.
    """

    def __init__(self, sigma: float = 5.0) -> None:
        """
        Args:
            sigma (float): Standard deviation for the Gaussian filter.
        """
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        image: torch.Tensor,
        target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate a Gaussian centroid mask and add it to the target under 'centroid_gaussian'.

        Args:
            image (torch.Tensor): Image tensor in [C,H,W] format.
            target (Dict[str, Any]): Target dict containing 'boxes' with bounding boxes.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Unmodified image and updated target.
        """
        height, width = image.shape[-2], image.shape[-1]
        gaussian_mask, _ = self.generate_gaussian_masks(target['boxes'], height, width)
        target['centroid_gaussian'] = gaussian_mask
        return image, target

    def generate_gaussian_masks(
        self,
        boxes: torch.Tensor,
        height: int,
        width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert bounding boxes to Gaussian masks centered at each box's centroid.

        Args:
            boxes (torch.Tensor): Tensor of shape [N,4] with bounding boxes [xmin, ymin, xmax, ymax].
            height (int): Image height.
            width (int): Image width.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The filtered Gaussian mask and the centroids mask.
        """
        mask = torch.zeros((1, height, width), dtype=torch.float32)

        # Place a value of 1.0 at each bounding box centroid
        for box in boxes:
            centroid_x = ((box[2] - box[0]) / 2) + box[0]
            centroid_y = ((box[3] - box[1]) / 2) + box[1]

            centroid_x = min(max(centroid_x, 0), width - 1)
            centroid_y = min(max(centroid_y, 0), height - 1)

            mask[0, int(centroid_y), int(centroid_x)] += 1.0

        centroids_mask = mask.clone()
        mask = torch.tensor(
            gaussian_filter(mask.numpy(), sigma=self.sigma),
            dtype=torch.float32
        )
        if mask.max() > 0:
            mask = mask / mask.max()

        return mask, centroids_mask


class SegmentationMaskTensorWithBackground(nn.Module):
    """
    A transform that converts binary masks and labels into a segmentation mask
    with an additional background channel (channel 0).
    """

    def __init__(self, num_classes: int) -> None:
        """
        Args:
            num_classes (int): Number of classes (excluding background).
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        image: torch.Tensor,
        target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Build and attach the segmentation mask with background to the target.

        Args:
            image (torch.Tensor): Image tensor in [C,H,W] format.
            target (Dict[str, Any]): Target dict containing 'masks' (binary masks) and 'labels' (class IDs).

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Unmodified image and updated target.
        """
        height, width = image.shape[-2], image.shape[-1]
        segmentation_mask = self.build_segmentation_tensor(
            target['masks'],
            target['labels'],
            height,
            width
        )
        target['segmentation_mask'] = segmentation_mask
        target.pop('masks', None)
        return image, target

    def build_segmentation_tensor(
        self,
        masks: List[torch.Tensor],
        labels: List[int],
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Create a (num_classes+1) x H x W segmentation tensor,
        with channel 0 as background.

        Args:
            masks (List[torch.Tensor]): A list of binary masks for each object.
            labels (List[int]): Class labels for each object.
            height (int): Image height.
            width (int): Image width.

        Returns:
            torch.Tensor: A segmentation mask tensor.
        """
        segmentation_tensor = torch.zeros(
            (self.num_classes + 1, height, width),
            dtype=torch.float32
        )

        for mask, label in zip(masks, labels):
            segmentation_tensor[label] += mask

        # Background is defined as the complement of the sum of all class channels
        segmentation_tensor[0] = 1.0 - segmentation_tensor[1:].sum(dim=0).clamp(max=1.0)
        return segmentation_tensor


class SegmentationMaskTensorWithBackground_masks(nn.Module):
    """
    A transform that converts binary masks and labels into a segmentation mask
    with a background channel, but keeps 'masks' in the target (for test usage).
    """

    def __init__(self, num_classes: int) -> None:
        """
        Args:
            num_classes (int): Number of classes (excluding background).
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        image: torch.Tensor,
        target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Build and attach the segmentation mask with background to the target,
        preserving 'masks' in the target.

        Args:
            image (torch.Tensor): Image tensor in [C,H,W] format.
            target (Dict[str, Any]): Target dict containing 'masks' (binary masks) and 'labels' (class IDs).

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Unmodified image and updated target.
        """
        height, width = image.shape[-2], image.shape[-1]
        segmentation_mask = self.build_segmentation_tensor(
            target['masks'],
            target['labels'],
            height,
            width
        )
        target['segmentation_mask'] = segmentation_mask
        return image, target

    def build_segmentation_tensor(
        self,
        masks: List[torch.Tensor],
        labels: List[int],
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Create a (num_classes+1) x H x W segmentation tensor,
        with channel 0 as background.

        Args:
            masks (List[torch.Tensor]): A list of binary masks for each object.
            labels (List[int]): Class labels for each object.
            height (int): Image height.
            width (int): Image width.

        Returns:
            torch.Tensor: A segmentation mask tensor.
        """
        segmentation_tensor = torch.zeros(
            (self.num_classes + 1, height, width),
            dtype=torch.float32
        )

        for mask, label in zip(masks, labels):
            segmentation_tensor[label] += mask

        segmentation_tensor[0] = 1.0 - segmentation_tensor[1:].sum(dim=0).clamp(max=1.0)
        return segmentation_tensor


class RandomApply(v2.Transform):
    """
    A transform that randomly applies a list of transforms with probability p.
    """

    def __init__(
        self,
        transforms: List[v2.Transform],
        p: float = 0.5
    ) -> None:
        """
        Args:
            transforms (List[v2.Transform]): A list of transforms to potentially apply.
            p (float): Probability of applying the transforms.
        """
        super().__init__()
        if not isinstance(transforms, (list, nn.ModuleList)):
            raise TypeError(
                "Argument transforms should be a sequence of callables or a `nn.ModuleList`"
            )
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a float in the interval [0.0, 1.0].")

        self.transforms = transforms
        self.p = p

    def _extract_params_for_v1_transform(self) -> Dict[str, Any]:
        return {"transforms": self.transforms, "p": self.p}

    def forward(self, *inputs: Any) -> Any:
        """
        Apply the stored transforms with probability p.

        Args:
            *inputs (Any): The image and optionally the target.

        Returns:
            Any: The transformed inputs if applied, else the original.
        """
        needs_unpacking = len(inputs) > 1

        if torch.rand(1) >= self.p:
            return inputs if needs_unpacking else inputs[0]

        outputs = inputs
        for transform in self.transforms:
            outputs = transform(*outputs) if needs_unpacking else (transform(outputs),)
        return outputs

    def extra_repr(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return "\n".join(format_string)


class Rescale(v2.Transform):
    """
    A transform that rescales the shorter side of an image by a given factor,
    optionally capped by a max_size.
    """

    def __init__(
        self,
        scale: float,
        max_size: Optional[int] = None,
        interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
        antialias: bool = True
    ) -> None:
        """
        Args:
            scale (float): Factor by which to rescale the smaller dimension of the image.
            max_size (int, optional): Maximum possible size (on the smaller dimension).
            interpolation (v2.InterpolationMode): Interpolation method.
            antialias (bool): Use antialiasing if supported.
        """
        super().__init__()
        self.scale = scale
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def make_params(self, flat_inputs: List[torch.Tensor]) -> Dict[str, int]:
        """
        Determine the new size based on the scale factor and optional max_size.

        Args:
            flat_inputs (List[torch.Tensor]): Flattened image inputs.

        Returns:
            Dict[str, int]: Dictionary with the 'size' key specifying the new size.
        """
        h, w = flat_inputs[0].shape[-2:]
        sz = int(self.scale * min(h, w))
        if self.max_size is not None:
            sz = min(sz, self.max_size)
        return dict(size=sz)

    def transform(self, inpt: Any, params: Dict[str, int]) -> Any:
        """
        Perform the resizing transform.

        Args:
            inpt (Any): Image or target to transform.
            params (Dict[str, int]): Dictionary containing the new 'size'.

        Returns:
            Any: Resized image or target.
        """
        return self._call_kernel(
            F.resize,
            inpt,
            params['size'],
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias
        )
