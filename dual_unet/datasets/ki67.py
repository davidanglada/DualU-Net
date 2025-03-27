import os
import os.path as osp
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import v2
import cv2
from PIL import Image

from .base import BaseCellCOCO, DetectionWrapper

KI67_CLASSES = ['Positive', 'Negative', 'Stroma']


class Ki67(torchvision.datasets.CocoDetection, BaseCellCOCO):
    """
    A dataset class for handling Ki67 images and annotations in a COCO-like format.
    Extends PyTorch's `CocoDetection` and a custom `BaseCellCOCO` interface.
    """

    def __init__(
        self,
        root: str,
        fold: Union[str, int],
        transforms: Optional[Any] = None,
        sigma: int = 5
    ) -> None:
        """
        Initialize the Ki67 dataset.

        Args:
            root (str): The root directory of the dataset.
            fold (Union[str, int]): The dataset split or fold, e.g. 'train', 'val', or an integer for a fold number.
            transforms (Optional[Any]): Transformations to be applied to each image-target pair.
            sigma (int): Standard deviation used for generating Gaussian masks.
        """
        self.root = root
        self.sigma = sigma

        # If fold is an integer, convert it to a string like "fold1"
        fold = fold if isinstance(fold, str) else f'fold{fold}'
        self.fold = fold

        img_folder = osp.join(root, fold, 'images')
        ann_file = osp.join(root, fold, 'annotations.json')
        super(Ki67, self).__init__(img_folder, ann_file, transforms=transforms)

    @property
    def num_classes(self) -> int:
        """
        Number of classes (excluding background).

        Returns:
            int: 3 total classes (Positive, Negative, Stroma).
        """
        return 3

    @property
    def class_names(self) -> List[str]:
        """
        Names of the Ki67 classes.

        Returns:
            List[str]: ['Positive', 'Negative', 'Stroma'].
        """
        return KI67_CLASSES

    def image_size(
        self,
        image_id: Optional[int] = None,
        idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Provides the size of an image in the dataset.

        Note:
            Currently returns a fixed size of (1024, 1024).

        Args:
            image_id (int, optional): Unique image identifier.
            idx (int, optional): Index of the image in the dataset.

        Returns:
            torch.Tensor: A tensor containing [height, width].
        """
        return torch.tensor([1024, 1024])

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples in the dataset.
        """
        return super(Ki67, self).__len__()

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Retrieve an image-target pair by index, filtering out invalid targets.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Image.Image, List[Dict[str, Any]]]: A tuple of the image and
            its corresponding annotations.
        """
        img, tgt = super(Ki67, self).__getitem__(idx)

        # Invalid targets (area <= 0 or insufficient segmentation points)
        if len(tgt) > 0:
            tgt = [
                t for t in tgt
                if t['area'] > 0 and len(t['segmentation'][0]) > 4
            ]

        # If all targets are invalid, insert a placeholder target
        if len(tgt) == 0:
            tgt = [dict(
                id=-1,
                image_id=idx,
                category_id=-1,
                bbox=[-1, -1, -1, -1],
                area=1024 * 1024,
                segmentation=[[0, 0, 0, 1024, 1024, 1024, 1024, 0]],
                centroid=[512, 512],
                iscrowd=0,
            )]

        return img, tgt

    def generate_gaussian_mask(
        self,
        tgt: List[Dict[str, Any]],
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Generate a Gaussian mask with peaks at each centroid in the provided targets.

        Args:
            tgt (List[Dict[str, Any]]): The COCO-style target annotations for one image.
            height (int): Height of the output mask.
            width (int): Width of the output mask.

        Returns:
            torch.Tensor: A 2D tensor representing the Gaussian mask.
        """
        mask = np.zeros((height, width), dtype=np.float32)

        for t in tgt:
            if 'centroid' in t:
                x, y = t['centroid']
                xx, yy = np.meshgrid(np.arange(width), np.arange(height))
                gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))
                mask += gaussian

        return torch.tensor(mask, dtype=torch.float32)

    def get_raw_image(
        self,
        image_id: Optional[int] = None,
        idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Retrieve the raw image from disk without additional dataset transforms.

        Args:
            image_id (int, optional): Unique image identifier.
            idx (int, optional): Index of the sample in the dataset.

        Returns:
            torch.Tensor: A tensor representation of the image in [C, H, W] format.
        """
        assert image_id is not None or idx is not None, \
            "Either image_id or idx must be provided."

        if image_id is None:
            image_id = self.ids[idx]

        img = self._load_image(image_id)

        transforms_pipeline = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        img = transforms_pipeline(img)
        return img


def build_ki67_dataset(cfg: Dict[str, Any], split: str = 'train') -> Any:
    """
    Build a Ki67 dataset object according to the specified configuration and split.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing dataset and transform info.
        split (str): Which dataset split to load (e.g., 'train', 'val', 'test').

    Returns:
        Any: A dataset instance, possibly wrapped for detection if 'num_classes' == 1.
    """
    from .transforms import build_transforms
    root = cfg['dataset'][split]['root']
    num_classes = cfg['dataset'][split]['num_classes']

    transforms = build_transforms(cfg, split, is_train=(split == 'train'))

    if num_classes == 1:
        dataset = DetectionWrapper(Ki67)(
            root,
            cfg['dataset'][split]['fold'],
            transforms=transforms
        )
    else:
        dataset = Ki67(
            root,
            cfg['dataset'][split]['fold'],
            transforms=transforms
        )

    # Wrap dataset to use torchvision's transforms v2
    dataset = datasets.wrap_dataset_for_transforms_v2(
        dataset,
        target_keys=('image_id', 'masks', 'boxes', 'labels')
    )
    return dataset


def ki67_to_coco(data_dir: str, fold: Union[str, int], out_dir: str) -> None:
    """
    Convert a Ki67 dataset from its original format (Numpy arrays) to COCO format.

    Args:
        data_dir (str): Directory containing the Ki67 data (images.npy and masks.npy).
        fold (Union[str, int]): The dataset fold or split (e.g., 'train', 'val', or an integer).
        out_dir (str): Output directory for the COCO-formatted dataset.
    """
    print("Converting Ki67 to COCO format...")

    img_path = osp.join(data_dir, f"{fold}", "images", "images.npy")
    mask_path = osp.join(data_dir, f"{fold}", "masks", "masks.npy")

    # Load images and masks
    images = np.load(img_path)
    # Ignore the background mask channel by slicing off the last channel
    masks = np.load(mask_path)[:, :, :, :-1]

    out_fold_dir = osp.join(out_dir, f"fold{fold}")
    if not osp.exists(osp.join(out_fold_dir, "images")):
        os.makedirs(osp.join(out_fold_dir, "images"))

    ls_images: List[Dict[str, Any]] = []
    ls_annots: List[Dict[str, Any]] = []
    instance_count = 1

    # Process each image
    for idx in range(images.shape[0]):
        filename = f"im{idx:04d}.png"
        image_i = images[idx]

        # Save the image
        Image.fromarray(image_i.astype(np.uint8)).save(
            osp.join(out_fold_dir, "images", filename)
        )

        height, width = image_i.shape[:2]
        ls_images.append(
            dict(id=idx, file_name=filename, height=height, width=width)
        )

        # Process each class mask
        mask_i = masks[idx]
        for lbl in range(mask_i.shape[-1]):
            uq_instance_ids = np.unique(mask_i[:, :, lbl])[1:]  # Skip zero
            for instance_id in uq_instance_ids:
                coords = np.where(mask_i[:, :, lbl] == instance_id)

                xmin = int(np.min(coords[1]))
                ymin = int(np.min(coords[0]))
                xmax = int(np.max(coords[1]))
                ymax = int(np.max(coords[0]))

                centroid_x = int(np.mean(coords[1]))
                centroid_y = int(np.mean(coords[0]))
                centroid = [centroid_x, centroid_y]

                # Extract contours
                mask_i_bin = (mask_i[:, :, lbl] == instance_id)
                contours, _ = cv2.findContours(
                    mask_i_bin.astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                contour = []
                for p in contours[0].reshape(-1, 2):
                    contour.append(int(p[0]))
                    contour.append(int(p[1]))
                contour = [contour]

                ls_annots.append(
                    dict(
                        id=instance_count,
                        image_id=idx,
                        category_id=int(lbl + 1),
                        bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                        area=(xmax - xmin) * (ymax - ymin),
                        segmentation=contour,
                        centroid=centroid,
                        iscrowd=0,
                    )
                )
                instance_count += 1

    # Create the categories for COCO
    categories = [
        dict(id=k + 1, name=v) for k, v in enumerate(KI67_CLASSES)
    ]

    # Final COCO format dictionary
    coco_format_json = dict(
        images=ls_images,
        annotations=ls_annots,
        categories=categories
    )

    # Save annotations to a JSON file
    with open(osp.join(out_fold_dir, "annotations.json"), 'w') as f:
        json.dump(coco_format_json, f)

    print(f"COCO format annotations with centroids saved to {osp.join(out_fold_dir, 'annotations.json')}")
