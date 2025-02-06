import os
import os.path as osp
import json
import sys

import numpy as np
from PIL import Image
import cv2
import scipy.io as sio

import torch
import torchvision
from torchvision.transforms import v2

from .base import BaseCellCOCO, DetectionWrapper

# Available nucleus classes for CoNSeP
CONSEP_NUCLEI = [
    "miscellaneous",
    "inflammatory",
    "epithelial",
    "spindleshaped"
]

# Mapping from original MAT labels to our 4-class scheme
ACTUAL_CONSEP_NUCLEI_MAP = {
    1: 1, 
    2: 2, 
    3: 3, 
    4: 3, 
    5: 4, 
    6: 4, 
    7: 4
}


class Consep(torchvision.datasets.CocoDetection, BaseCellCOCO):
    """
    A dataset class for the CoNSeP dataset, extending PyTorch's CocoDetection and BaseCellCOCO.

    Args:
        root (str): Root directory of the dataset.
        fold (str): Subset/fold to use (e.g. "train", "test").
        transforms: Optional transformations to apply to the data.
    """

    def __init__(
        self, 
        root: str, 
        fold: str, 
        transforms=None
    ) -> None:
        self.root = root
        self.fold = fold
        img_folder = osp.join(root, f"{fold}", "images")
        ann_file = osp.join(root, f"{fold}", "annotations.json")
        super().__init__(img_folder, ann_file, transforms=transforms)

    @property
    def num_classes(self) -> int:
        """Number of nucleus classes in the CoNSeP dataset."""
        return 4

    @property
    def class_names(self) -> list:
        """List of class names for the CoNSeP dataset."""
        return CONSEP_NUCLEI

    def image_size(self, image_id=None, idx=None) -> torch.Tensor:
        """
        Returns the default image size (1024x1024) for CoNSeP images.

        Args:
            image_id (int, optional): An image ID in the dataset.
            idx (int, optional): Index of the sample in the dataset.

        Returns:
            torch.Tensor: A 1D tensor containing (height, width).
        """
        return torch.tensor([1024, 1024])

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return super().__len__()

    def __getitem__(self, idx: int):
        """
        Retrieve the (image, target) pair for a given index.

        Args:
            idx (int): Index of the desired sample.

        Returns:
            (PIL.Image.Image, List[dict]): Image and its annotation list.
        """
        img, tgt = super().__getitem__(idx)

        # Filter out invalid targets
        if len(tgt) > 0:
            tgt = [t for t in tgt if t["area"] > 0 and len(t["segmentation"][0]) > 4]

        # Insert a dummy placeholder if no valid targets
        if len(tgt) == 0:
            tgt = [
                dict(
                    id=-1,
                    image_id=idx,
                    category_id=-1,
                    bbox=[-1, -1, -1, -1],
                    area=-1,
                    segmentation=[-1],
                    centroid=[-1, -1],
                    iscrowd=0,
                )
            ]

        return img, tgt

    def get_raw_image(self, image_id: int = None, idx: int = None) -> torch.Tensor:
        """
        Load and return a raw image from the dataset (resized to 1024x1024).

        Args:
            image_id (int, optional): ID of the image in COCO format.
            idx (int, optional): Index of the image in the dataset.

        Returns:
            torch.Tensor: A resized image tensor of shape (3, 1024, 1024).
        """
        assert image_id is not None or idx is not None, "Provide either image_id or idx."

        if image_id is None:
            image_id = self.ids[idx]

        img = self._load_image(image_id)

        transforms_ = v2.Compose([
            v2.ToImage(),
            v2.Resize(
                (1024, 1024),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True
            ),
            v2.ToDtype(torch.float32, scale=True),
        ])
        img = transforms_(img)
        return img


def build_consep_dataset(cfg: dict, split: str = "train"):
    """
    Build a CoNSeP dataset instance with given config and split.

    Args:
        cfg (dict): A configuration dictionary.
        split (str): Data split to build ("train", "test", etc.).

    Returns:
        torchvision.datasets.wrap_dataset_for_transforms_v2: 
            The resulting dataset object, wrapped for new transforms.
    """
    from .transforms import build_transforms

    root = cfg["dataset"][split]["root"]
    num_classes = cfg["dataset"][split]["num_classes"]

    transforms_ = build_transforms(cfg, split, is_train=(split == "train"))
    transforms_.transforms.insert(
        3, 
        v2.Resize(
            (1024, 1024), 
            interpolation=v2.InterpolationMode.BICUBIC,
            antialias=True
        )
    )

    if num_classes == 1:
        dataset = DetectionWrapper(Consep)(
            root, cfg["dataset"][split]["fold"], transforms=transforms_
        )
    else:
        dataset = Consep(
            root, cfg["dataset"][split]["fold"], transforms=transforms_
        )

    dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(
        dataset, target_keys=("image_id", "masks", "boxes", "labels")
    )

    return dataset


def consep2coco(
    data_dir: str, 
    fold: str, 
    out_dir: str
) -> None:
    """
    Convert CoNSeP data into COCO format with centroid information.

    Args:
        data_dir (str): Directory containing {fold}/images and {fold}/labels.
        fold (str): Data split to process ("train" or "test").
        out_dir (str): Output directory to store images and annotations in COCO format.
    """
    img_dir = osp.join(data_dir, fold, "images")
    lbl_dir = osp.join(data_dir, fold, "labels")

    out_dir = osp.join(out_dir, f"{fold}")
    if not osp.exists(osp.join(out_dir, "images")):
        os.makedirs(osp.join(out_dir, "images"))

    ls_images = []
    ls_annots = []
    instance_count = 1

    for img_idx, img_name in enumerate(os.listdir(img_dir)):
        img_path = osp.join(img_dir, img_name)
        lbl_path = osp.join(lbl_dir, img_name.split(".")[0] + ".mat")

        assert osp.exists(lbl_path), f"Label file not found at {lbl_path}"

        # Read and save image
        img = Image.open(img_path)
        width, height = img.size
        img_filename = osp.join(out_dir, "images", img_name)
        img.save(img_filename)

        ls_images.append(
            dict(id=img_idx, file_name=img_filename, height=height, width=width)
        )

        lbl = sio.loadmat(lbl_path)
        inst_map, type_map, inst_type = lbl["inst_map"], lbl["type_map"], lbl["inst_type"]
        uq_inst_ids = np.unique(inst_map)

        for inst_id in uq_inst_ids:
            if inst_id == 0:
                continue

            inst_mask_i = (inst_map == inst_id)
            coords = np.where(inst_mask_i)
            xmin, ymin = int(np.min(coords[1])), int(np.min(coords[0]))
            xmax, ymax = int(np.max(coords[1])), int(np.max(coords[0]))
            centroid_x = int(np.mean(coords[1]))
            centroid_y = int(np.mean(coords[0]))
            centroid = [centroid_x, centroid_y]

            contours, _ = cv2.findContours(
                inst_mask_i.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = []
            for p in contours[0].reshape(-1, 2):
                contour.append(int(p[0]))
                contour.append(int(p[1]))
            contour = [contour]

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

    categories = [dict(id=k + 1, name=v) for k, v in enumerate(CONSEP_NUCLEI)]
    coco_format_json = dict(
        images=ls_images,
        annotations=ls_annots,
        categories=categories,
    )

    out_json_path = osp.join(out_dir, "annotations.json")
    with open(out_json_path, "w") as f:
        json.dump(coco_format_json, f)

    print(f"COCO format annotations with centroids saved to {out_json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoNSeP to COCO conversion")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory.")
    parser.add_argument("--fold", type=str, default="train", help="Split (train/test).")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    args = parser.parse_args()
    consep2coco(args.data_dir, args.fold, args.out_dir)
