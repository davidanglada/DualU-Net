import os
import os.path as osp
import json
import sys
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2
import scipy.io as sio

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import v2

from dual_unet.datasets.base import BaseCellCOCO, DetectionWrapper

CONSEP_NUCLEI = ['miscellaneous', 'inflammatory', 'epithelial', 'spindleshaped']
ACTUAL_CONSEP_NUCLEI_MAP = {1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 4}


class Consep(torchvision.datasets.CocoDetection, BaseCellCOCO):
    """
    A dataset class for the Consep dataset, extending PyTorch's CocoDetection and BaseCellCOCO.

    This class handles image loading, target (annotation) filtering, and provides
    methods to retrieve raw images and dataset size information.
    """

    def __init__(
        self,
        root: str,
        fold: str,
        transforms: Optional[Any] = None
    ) -> None:
        """
        Initialize the Consep dataset.

        Args:
            root (str): Path to the root directory of the dataset.
            fold (str): The data subset to use (e.g., 'train', 'val', 'test').
            transforms (Optional[Any]): Optional transformations to be applied to each sample.
        """
        self.root = root
        self.fold = fold
        img_folder = osp.join(root, f'{fold}', 'images')
        ann_file = osp.join(root, f'{fold}', 'annotations.json')
        super(Consep, self).__init__(img_folder, ann_file, transforms=transforms)

    @property
    def num_classes(self) -> int:
        """
        Returns:
            int: The total number of classes (excluding background).
        """
        return 4

    @property
    def class_names(self) -> List[str]:
        """
        Returns:
            List[str]: A list of class names used in the Consep dataset.
        """
        return CONSEP_NUCLEI

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
        return super(Consep, self).__len__()

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Retrieve an image-target pair by index.

        Filters out invalid targets (area <= 0 or insufficient points in segmentation).
        Returns a placeholder target if no valid annotations are found.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Image.Image, List[Dict[str, Any]]]: The image and its corresponding annotations.
        """
        img, tgt = super(Consep, self).__getitem__(idx)

        # Remove invalid targets
        if len(tgt) > 0:
            tgt = [
                t for t in tgt
                if t['area'] > 0 and len(t['segmentation'][0]) > 4
            ]

        # If no valid targets remain, insert a placeholder
        if len(tgt) == 0:
            tgt = [dict(
                id=-1,
                image_id=idx,
                category_id=-1,
                bbox=[-1, -1, -1, -1],
                area=-1,
                segmentation=[-1],
                centroid=[-1, -1],
                iscrowd=0,
            )]

        return img, tgt

    def get_raw_image(
        self,
        image_id: Optional[int] = None,
        idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Retrieve a raw image from the dataset (resized to (1024, 1024)).

        Args:
            image_id (int, optional): Unique image identifier.
            idx (int, optional): Index of the image in the dataset.

        Returns:
            torch.Tensor: A tensor containing the raw image data.
        """
        assert image_id is not None or idx is not None, \
            "Either image_id or idx must be provided."

        if image_id is None:
            image_id = self.ids[idx]

        img = self._load_image(image_id)

        transforms_pipeline = v2.Compose([
            v2.ToImage(),
            v2.Resize((1024, 1024),
                      interpolation=v2.InterpolationMode.BICUBIC,
                      antialias=True),
            v2.ToDtype(torch.float32, scale=True)
        ])

        img = transforms_pipeline(img)
        return img


def build_consep_dataset(cfg: Dict[str, Any], split: str = 'train') -> Any:
    """
    Build a Consep dataset object according to the specified configuration and split.

    Args:
        cfg (Dict[str, Any]): A configuration dictionary containing dataset and transform info.
        split (str): Which split to load (e.g., 'train', 'val', 'test').

    Returns:
        Any: The constructed dataset, potentially wrapped for detection or segmentation.
    """
    from .transforms import build_transforms

    root = cfg['dataset'][split]['root']
    num_classes = cfg['dataset'][split]['num_classes']
    is_train = (split == 'train')

    # Build the transforms
    transforms = build_transforms(cfg, split, is_train=is_train)
    transforms.transforms.insert(
        3,
        v2.Resize(
            (1024, 1024),
            interpolation=v2.InterpolationMode.BICUBIC,
            antialias=True
        )
    )

    # Create the dataset
    if num_classes == 1:
        DatasetClass = DetectionWrapper(Consep)
        dataset = DatasetClass(
            root,
            cfg['dataset'][split]['fold'],
            transforms=transforms
        )
    else:
        dataset = Consep(
            root,
            cfg['dataset'][split]['fold'],
            transforms=transforms
        )

    # Wrap dataset to use transforms v2 (for new torchvision)
    dataset = datasets.wrap_dataset_for_transforms_v2(
        dataset,
        target_keys=('image_id', 'masks', 'boxes', 'labels')
    )
    return dataset


def consep2coco(
    data_dir: str,
    fold: str,
    out_dir: str
) -> None:
    """
    Convert the Consep dataset from its original format to COCO format.

    This function reads images and corresponding .mat labels, extracts instance information,
    and saves the images and annotation files in COCO-compatible JSON format.

    Args:
        data_dir (str): The base directory containing the Consep images and labels.
        fold (str): The data subset (e.g., 'train', 'val', 'test').
        out_dir (str): The destination directory for the COCO-formatted dataset.
    """
    img_dir = osp.join(data_dir, fold, "images")
    lbl_dir = osp.join(data_dir, fold, "labels")

    out_dir = osp.join(out_dir, f"{fold}")
    if not osp.exists(osp.join(out_dir, "images")):
        os.makedirs(osp.join(out_dir, "images"))

    ls_images = []
    ls_annots = []
    instance_count = 1  # Instance ID starts from 1

    # Iterate over images
    for img_idx, img_name in enumerate(os.listdir(img_dir)):
        img_path = osp.join(img_dir, img_name)
        lbl_path = osp.join(lbl_dir, img_name.split(".")[0] + ".mat")
        assert osp.exists(lbl_path), f"Label file not found: {lbl_path}"

        # Process the image
        img = Image.open(img_path)
        width, height = img.size  # PIL format: (width, height)
        img_filename = osp.join(out_dir, "images", img_name)
        img.save(img_filename)

        ls_images.append(
            dict(
                id=img_idx,
                file_name=img_filename,
                height=height,
                width=width
            )
        )

        # Process the corresponding label
        lbl = sio.loadmat(lbl_path)
        inst_map = lbl['inst_map']
        type_map = lbl['type_map']   # noqa: F841
        inst_type = lbl['inst_type']  # noqa: F841
        uq_inst_ids = np.unique(inst_map)

        # For each unique instance (excluding 0/background)
        for inst_id in uq_inst_ids:
            if inst_id == 0:
                continue

            inst_mask_i = (inst_map == inst_id)
            coords = np.where(inst_mask_i)

            xmin = int(np.min(coords[1]))
            ymin = int(np.min(coords[0]))
            xmax = int(np.max(coords[1]))
            ymax = int(np.max(coords[0]))

            centroid_x = int(np.mean(coords[1]))
            centroid_y = int(np.mean(coords[0]))
            centroid = [centroid_x, centroid_y]

            # Extract contours for the segmentation polygon
            contours, _ = cv2.findContours(
                inst_mask_i.astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
            contour = []
            for p in contours[0].reshape(-1, 2):
                contour.append(int(p[0]))
                contour.append(int(p[1]))
            contour = [contour]

            # Map instance type to one of the 4 categories
            inst_type_i = int(inst_type[int(inst_id) - 1])
            inst_type_i = ACTUAL_CONSEP_NUCLEI_MAP[inst_type_i]

            ls_annots.append(
                dict(
                    id=instance_count,
                    image_id=img_idx,
                    category_id=inst_type_i,
                    bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                    area=(xmax - xmin) * (ymax - ymin),
                    segmentation=contour,
                    centroid=centroid,
                    iscrowd=0,
                )
            )
            instance_count += 1

    # Build the COCO categories
    categories = [dict(id=k + 1, name=v) for k, v in enumerate(CONSEP_NUCLEI)]

    # Prepare and save the COCO-format JSON
    coco_format_json = dict(
        images=ls_images,
        annotations=ls_annots,
        categories=categories
    )
    with open(osp.join(out_dir, "annotations.json"), 'w') as f:
        json.dump(coco_format_json, f)

    print(f"COCO format annotations with centroids saved to {osp.join(out_dir, 'annotations.json')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Consep COCO Conversion')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    parser.add_argument('--fold', type=str, default='train', help='Data subset (train, val, test)')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    consep2coco(args.data_dir, args.fold, args.out_dir)
