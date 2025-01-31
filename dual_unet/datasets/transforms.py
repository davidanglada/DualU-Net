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

# from ..utils.box_ops import normalize_box, denormalize_box

def build_transforms(cfg, split, is_train=True):
    transforms = [v2.ToImage()]

    # sanity check
    # transforms.append(v2.ClampBoundingBoxes())
    # transforms.append(v2.SanitizeBoundingBoxes())

    # augmentation when training
    if is_train:
        # augmentations
        transforms.append(build_augmentations(cfg))

        # sanity check after augmentations
        # transforms.append(v2.ClampBoundingBoxes())
        # transforms.append(v2.SanitizeBoundingBoxes())
        
    # convert image and bbox format
    # transforms.append(v2.ConvertBoundingBoxFormat(format='CXCYWH'))
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    
    # resize
    if 'rescale' in cfg['transforms']:
        transforms.append(Rescale(cfg['transforms']['rescale'], 
                            antialias=True,
                            interpolation=v2.InterpolationMode.BICUBIC))
    # image normalization
    if 'normalize' in cfg['transforms']:
        mean = cfg['transforms']['normalize']['mean']
        std = cfg['transforms']['normalize']['std']
        transforms.append(v2.Normalize(mean=mean, std=std))
    # bounding box normalization
    # transforms.append(NormalizeBoundingBoxes())

    transforms.append(GaussianCentroidMask(sigma=cfg['training']['sigma']))
    if split == 'test':
        transforms.append(SegmentationMaskTensorWithBackground_masks(num_classes=cfg['dataset'][split]['num_classes']))
    else:
        transforms.append(SegmentationMaskTensorWithBackground(num_classes=cfg['dataset'][split]['num_classes']))

    return v2.Compose(transforms)

def build_augmentations(cfg):
    augs = list()
    assert 'augmentations' in cfg['transforms']
    for aug in cfg['transforms']['augmentations']:
        assert 'name' in aug, "Augmentation must have a name."

        # build the augmentation, don't send name and p params as kwargs
        a = AugmentationFactory.build(aug['name'],
                                **{k : v for k, v in aug.items() \
                                        if k not in ['name','p']})
        # if p in kwargs, random apply the transform
        if 'p' in aug:
            # TODO: we should use v2.RandomApply but it's crashing with 2 inputs idk why
            a = RandomApply([a], p=aug['p'])
        augs.append(a)
    return v2.Compose(augs)

class AugmentationFactory:
    def build(name, **kwargs):
        if name == "hflip":
            t = v2.RandomHorizontalFlip(p=1.0)
        elif name == "vflip":
            t = v2.RandomVerticalFlip(p=1.0)
        elif name == "rotate90":
            t = RandomRotation90(**kwargs)
        elif name == "cjitter":
            t = v2.ColorJitter(**kwargs)
        elif name == "elastic":
            t = v2.ElasticTransform(**kwargs)
        elif name == "blur":
            t = v2.GaussianBlur(**kwargs)
        elif name == "resizedcrop":
            t = v2.RandomResizedCrop(**kwargs, antialias=True)
        elif name == "resize":
            t = v2.Resize(**kwargs, antialias=True)
        elif name == "randomcrop":
            t = v2.RandomCrop(**kwargs)
        elif name == "hedjitter":
            t = HEDJitter(**kwargs)
        else:
            raise ValueError(f'Unknown augmentation: {name}')
        return t

# class NormalizeBoundingBoxes(nn.Module):
#     def forward(self, image, target):
#         h, w = target['boxes'].canvas_size
#         # boxes to float
#         boxes = copy.deepcopy(target['boxes'].data).float()
#         # update target
#         target['boxes'].data = normalize_box(boxes, (h,w))
#         return image, target

# class DenormalizeBoundingBoxes(nn.Module):
#     def forward(self, image, target):
#         h, w = target['boxes'].canvas_size
#         # boxes to float
#         boxes = copy.deepcopy(target['boxes'].data)
#         # update target
#         target['boxes'].data = denormalize_box(boxes, (h,w))
#         return image, target

class RandomRotation90(v2.Transform):
    f"""
        Extend RandomRotation by only allowing rotations of 90, 180 and -90 (270) degrees.
    """
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._fill = _setup_fill_arg(0)

    def _get_params(self, flat_inputs):
        #angle = torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item()
        # randomly select 90, -90 or 180 (as tensor)
        angles = torch.tensor([0, 90, -90, 180])
        angle  = angles[torch.randperm(4)[0]].item()
        return dict(angle=angle)

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
    def __init__(self, alpha = (0.98,1.02),
                       beta  = (-0.02, 0.02)):
        super().__init__()
        if not isinstance(alpha, tuple):
            alpha = (1.0-alpha, 1.0+alpha)
        if not isinstance(beta, tuple):
            beta = (-beta, beta)
        self.alpha = alpha
        self.beta = beta

        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                                        [0.07, 0.99, 0.11],
                                        [0.27, 0.57, 0.78]], 
                                        dtype=torch.float32, 
                                        requires_grad=False)
        self.hed_from_rgb = torch.inverse(self.rgb_from_hed)
    
    def forward(self, image, target):
        # get alpha and beta for H, E and D channels
        alpha_H = torch.empty(1).uniform_(self.alpha[0], self.alpha[1]).item()
        alpha_E = torch.empty(1).uniform_(self.alpha[0], self.alpha[1]).item()
        #alpha_D = torch.empty(1).uniform_(self.alpha[0], self.alpha[1]).item()
        beta_H = torch.empty(1).uniform_(self.beta[0], self.beta[1]).item()
        beta_E = torch.empty(1).uniform_(self.beta[0], self.beta[1]).item()
        #beta_D = torch.empty(1).uniform_(self.beta[0], self.beta[1]).item()

        # convert to float32
        orig_dtype = image.dtype
        image = F.convert_image_dtype(image, torch.float32)

        # 
        image = color.rgb2hed(image.permute(1,2,0).numpy())
        image[...,0] = image[...,0] * alpha_H + beta_H
        image[...,1] = image[...,1] * alpha_E + beta_E
        #image[...,2] = image[...,2] * alpha_D + beta_D

        # convert back to rgb tensor
        image = torch.tensor(color.hed2rgb(image)).permute(2,0,1)
        image = F.convert_image_dtype(image, orig_dtype)

        return image, target

class GaussianCentroidMask(nn.Module):
    def __init__(self, sigma=5):
        super().__init__()
        self.sigma = sigma  # Standard deviation for Gaussian distribution

    def forward(self, image, target):
        """
        Args:
            image: Input image (not modified by this transform)
            target: Target dict with 'boxes' (bounding boxes) and 'labels' (class IDs)
        
        Returns:
            image: Unmodified image
            target: Modified target with 'centroid_gaussian' added
        """
        height, width = image.shape[-2], image.shape[-1]
        # print(target['boxes'])
        # print(height, width)
        gaussian_mask, centroids_mask = self.generate_gaussian_masks(target['boxes'], height, width)
        target['centroid_gaussian'] = gaussian_mask
        # target['centroids'] = centroids_mask
        # target.pop('boxes', None)  # Remove the 'boxes' key
        # print(target['centroid_gaussian'].shape)
        # print(target['centroid_gaussian'])
        # print(target)
        return image, target

    def generate_gaussian_masks(self, boxes, height, width):
        """
        Generate Gaussian masks from bounding boxes.
        """
        mask = torch.zeros((1, height, width))  # Initialize empty mask (HxW)
        # print(boxes)
        for box in boxes:

            # Get the centroid coordinates in formatbbox=[xmin, ymin, xmax, ymax]
            # print(box)
            centroid_x = ((box[2] - box[0]) / 2) + box[0]
            centroid_y = ((box[3] - box[1]) / 2) + box[1]
            # print(centroid_x, centroid_y)
            # print(int(centroid_x), int(centroid_y))

            # Deal with edge cases
            centroid_x = min(max(centroid_x, 0), width - 1)
            centroid_y = min(max(centroid_y, 0), height - 1)
            
            mask[0, int(centroid_y), int(centroid_x)] += 1.0  # Set the centroid pixel to 1
        
        centroids_mask = mask
        mask = gaussian_filter(mask, sigma=self.sigma)
        gaussian_mask = mask/mask.max()
        # print(gaussian_mask)
        # print(np.unique(gaussian_mask))
        # gaussian_mask = mask

        return torch.Tensor(gaussian_mask), torch.Tensor(centroids_mask) # Add channel dimension

class SegmentationMaskTensorWithBackground(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes  # Number of classes (excluding background)

    def forward(self, image, target):
        """
        Args:
            image: Input image
            target: Target dict with 'masks' (binary masks for each object)

        Returns:
            image: Unmodified image
            target: Modified target with 'segmentation_mask' added (including background channel)
        """
        height, width = image.shape[-2], image.shape[-1]
        segmentation_mask = self.build_segmentation_tensor(target['masks'], target['labels'], height, width)
        target['segmentation_mask'] = segmentation_mask
        target.pop('masks', None)  # Remove the 'masks' key
        return image, target

    def build_segmentation_tensor(self, masks, labels, height, width):
        """
        Build a (C+1)xHxW tensor from binary masks where C is the number of object classes.
        The first channel is the background.
        """
        # Initialize the segmentation tensor (C+1) for the background and the object classes
        segmentation_tensor = torch.zeros((self.num_classes + 1, height, width), dtype=torch.float32)

        # Fill the foreground channels
        for i in range(len(labels)):
            mask = masks[i]
            label = labels[i]
            segmentation_tensor[label] += mask

        # Calculate the background as the complement of all other channels
        segmentation_tensor[0] = 1.0 - segmentation_tensor[1:].sum(dim=0).clamp(max=1.0)

        return segmentation_tensor


class SegmentationMaskTensorWithBackground_masks(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes  # Number of classes (excluding background)

    def forward(self, image, target):
        """
        Args:
            image: Input image
            target: Target dict with 'masks' (binary masks for each object)

        Returns:
            image: Unmodified image
            target: Modified target with 'segmentation_mask' added (including background channel)
        """
        height, width = image.shape[-2], image.shape[-1]
        segmentation_mask = self.build_segmentation_tensor(target['masks'], target['labels'], height, width)
        target['segmentation_mask'] = segmentation_mask
        # target.pop('masks', None)  # Remove the 'masks' key
        return image, target

    def build_segmentation_tensor(self, masks, labels, height, width):
        """
        Build a (C+1)xHxW tensor from binary masks where C is the number of object classes.
        The first channel is the background.
        """
        # Initialize the segmentation tensor (C+1) for the background and the object classes
        segmentation_tensor = torch.zeros((self.num_classes + 1, height, width), dtype=torch.float32)

        # Fill the foreground channels
        for i in range(len(labels)):
            mask = masks[i]
            label = labels[i]
            segmentation_tensor[label] += mask

        # Calculate the background as the complement of all other channels
        segmentation_tensor[0] = 1.0 - segmentation_tensor[1:].sum(dim=0).clamp(max=1.0)

        return segmentation_tensor

class RandomApply(v2.Transform):
    def __init__(self, transforms, p = 0.5) -> None:
        super().__init__()

        if not isinstance(transforms, (list, nn.ModuleList)):
            raise TypeError("Argument transforms should be a sequence of callables or a `nn.ModuleList`")
        self.transforms = transforms

        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        self.p = p

    def _extract_params_for_v1_transform(self):
        return {"transforms": self.transforms, "p": self.p}

    def forward(self, *inputs):
        needs_unpacking = len(inputs) > 1

        if torch.rand(1) >= self.p:
            return inputs if needs_unpacking else inputs[0]

        for transform in self.transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)
        return outputs

    def extra_repr(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return "\n".join(format_string)
    
class Rescale(v2.Transform):
    def __init__(self, scale, 
                 max_size=None,
                 interpolation=v2.InterpolationMode.BILINEAR,
                 antialias=True) -> None:
        super().__init__()
        self.scale = scale
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_params(self, flat_inputs):
        # get image size
        #h, w = F._get_image_size(flat_inputs)
        # get image size
        h, w = flat_inputs[0].shape[-2:]
        # calculate new size
        sz = int(self.scale * min(h, w))
        if self.max_size is not None:
            sz = min(sz, self.max_size)
        return dict(size=sz)

    def _transform(self, inpt, params):
        return self._call_kernel(
            F.resize,
            inpt,
            params['size'],
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias,
        )