import os
import os.path as osp

import numpy as np
import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import v2
import cv2
from PIL import Image
import json

from .base import BaseCellCOCO, DetectionWrapper

KI67_CLASSES = ["Positive", "Negative", "Stroma"]


class Ki67(torchvision.datasets.CocoDetection, BaseCellCOCO):
    """
    Dataset class for Ki67 data in COCO format.

    Args:
        root (str): Root directory for the dataset.
        fold (Union[str, int]): Name or number of the fold (e.g. "train", "test", or fold index).
        transforms (optional): PyTorch or custom transformations to apply.
        sigma (float, optional): Standard deviation for generating Gaussian centroid masks, default=5.
    """

    def __init__(
        self, 
        root: str, 
        fold, 
        transforms=None, 
        sigma: float = 5
    ) -> None:
        self.root = root
        self.sigma = sigma

        fold = fold if isinstance(fold, str) else f"fold{fold}"
        self.fold = fold

        img_folder = osp.join(root, fold, "images")
        ann_file = osp.join(root, fold, "annotations.json")

        super(Ki67, self).__init__(img_folder, ann_file, transforms=transforms)

    @property
    def num_classes(self) -> int:
        """
        Number of classes (3 for Ki67: Positive, Negative, Stroma).
        """
        return 3

    @property
    def class_names(self) -> list:
        """
        Returns the list of Ki67 class names.
        """
        return KI67_CLASSES

    def image_size(self, image_id=None, idx=None) -> torch.Tensor:
        """
        Returns a default image size (1024x1024) for Ki67 data.

        Args:
            image_id (int, optional): COCO image ID.
            idx (int, optional): Dataset index.

        Returns:
            torch.Tensor: Shape (2,) containing (height, width).
        """
        return torch.tensor([1024, 1024])

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return super(Ki67, self).__len__()

    def __getitem__(self, idx: int):
        """
        Get the image and target annotations at index `idx`.

        Args:
            idx (int): Index of the desired sample.

        Returns:
            Tuple[Image.Image, List[dict]]: PIL image and the annotations.
        """
        img, tgt = super(Ki67, self).__getitem__(idx)
        height, width = img.size

        # Filter out invalid targets
        if len(tgt) > 0:
            tgt = [t for t in tgt if t["area"] > 0 and len(t["segmentation"][0]) > 4]

        # If no valid targets, create a dummy placeholder annotation
        if len(tgt) == 0:
            tgt = [
                dict(
                    id=-1,
                    image_id=idx,
                    category_id=-1,
                    bbox=[-1, -1, -1, -1],
                    area=1024 * 1024,
                    segmentation=[[0, 0, 0, 1024, 1024, 1024, 1024, 0]],
                    centroid=[512, 512],
                    iscrowd=0,
                )
            ]

        return img, tgt

    def generate_gaussian_mask(
        self, 
        tgt: list, 
        height: int, 
        width: int
    ) -> torch.Tensor:
        """
        Generate a 2D Gaussian mask for each centroid in the annotations.

        Args:
            tgt (List[dict]): List of COCO annotations with 'centroid' keys.
            height (int): Image height.
            width (int): Image width.

        Returns:
            torch.Tensor: A 2D float32 tensor (height, width) with Gaussian peaks at centroids.
        """
        mask = np.zeros((height, width), dtype=np.float32)

        for t in tgt:
            if "centroid" in t:
                x, y = t["centroid"]
                xx, yy = np.meshgrid(np.arange(width), np.arange(height))
                gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))
                mask += gaussian

        return torch.tensor(mask, dtype=torch.float32)

    def get_raw_image(self, image_id=None, idx=None) -> torch.Tensor:
        """
        Load and return a raw image from disk, converting to float32 tensor [0..1] range.

        Args:
            image_id (int, optional): COCO image ID.
            idx (int, optional): Dataset index.

        Returns:
            torch.Tensor: A tensor of shape (C, H, W), with float32 type in [0..1].
        """
        assert image_id is not None or idx is not None, "Provide either image_id or idx."
        if image_id is None:
            image_id = self.ids[idx]

        img = self._load_image(image_id)

        transform_pipeline = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        img = transform_pipeline(img)
        return img


def build_ki67_dataset(cfg: dict, split: str = "train"):
    """
    Build a Ki67 dataset instance according to the config.

    Args:
        cfg (dict): A configuration dictionary.
        split (str): The dataset split ("train" or "test").

    Returns:
        torchvision.datasets.wrap_dataset_for_transforms_v2: Wrapped dataset with transformations.
    """
    from .transforms import build_transforms

    root = cfg["dataset"][split]["root"]
    num_classes = cfg["dataset"][split]["num_classes"]

    transforms_ = build_transforms(cfg, split, is_train=(split == "train"))

    if num_classes == 1:
        dataset = DetectionWrapper(Ki67)(
            root, 
            cfg["dataset"][split]["fold"], 
            transforms=transforms_
        )
    else:
        dataset = Ki67(
            root, 
            cfg["dataset"][split]["fold"], 
            transforms=transforms_
        )

    dataset = datasets.wrap_dataset_for_transforms_v2(
        dataset, target_keys=("image_id", "masks", "boxes", "labels")
    )
    return dataset


def ki67_to_coco(data_dir: str, fold, out_dir: str) -> None:
    """
    Convert the Ki67 data into COCO format.

    Args:
        data_dir (str): Path to the directory containing images/masks subfolders.
        fold: Either fold index or name (e.g. "train").
        out_dir (str): Path to store the COCO-formatted data.
    """
    print("Converting Ki67 to COCO format...")

    img_path = osp.join(data_dir, f"{fold}", "images", "images.npy")
    mask_path = osp.join(data_dir, f"{fold}", "masks", "masks.npy")

    images = np.load(img_path)
    masks = np.load(mask_path)[:, :, :, :-1]  # exclude background channel

    out_fold_dir = osp.join(out_dir, f"fold{fold}")
    img_out_dir = osp.join(out_fold_dir, "images")
    if not osp.exists(img_out_dir):
        os.makedirs(img_out_dir)

    ls_images = []
    ls_annots = []
    instance_count = 1

    for idx in range(images.shape[0]):
        filename = f"im{idx:04d}.png"
        image_i = images[idx]

        Image.fromarray(image_i.astype(np.uint8)).save(osp.join(img_out_dir, filename))

        height, width = image_i.shape[:2]
        ls_images.append(
            dict(id=idx, file_name=filename, height=height, width=width)
        )

        mask_i = masks[idx]
        for lbl in range(mask_i.shape[-1]):
            uq_instance_ids = np.unique(mask_i[:, :, lbl])[1:]  # skip background
            for instance_id in uq_instance_ids:
                coords = np.where(mask_i[:, :, lbl] == instance_id)

                xmin = int(np.min(coords[1]))
                ymin = int(np.min(coords[0]))
                xmax = int(np.max(coords[1]))
                ymax = int(np.max(coords[0]))

                centroid_x = int(np.mean(coords[1]))
                centroid_y = int(np.mean(coords[0]))
                centroid = [centroid_x, centroid_y]

                bin_mask = (mask_i[:, :, lbl] == instance_id)
                contours, _ = cv2.findContours(
                    bin_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
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

    categories = [dict(id=k + 1, name=v) for k, v in enumerate(KI67_CLASSES)]
    coco_format_json = dict(
        images=ls_images,
        annotations=ls_annots,
        categories=categories,
    )

    out_json_path = osp.join(out_fold_dir, "annotations.json")
    with open(out_json_path, "w") as f:
        json.dump(coco_format_json, f)

    print(f"COCO format annotations with centroids saved to {out_json_path}")
