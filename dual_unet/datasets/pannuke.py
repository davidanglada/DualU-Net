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

PANNUKE_TISSUE = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix", "Colon",
    "Esophagus", "HeadNeck", "Kidney", "Liver", "Lung", "Ovarian", "Pancreatic",
    "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus"
]

PANNUKE_NUCLEI = [
    "neoplastic",
    "inflammatory",
    "connective",
    "necrosis",
    "epithelial"
]


class Pannuke(torchvision.datasets.CocoDetection, BaseCellCOCO):
    """
    Pannuke dataset class that extends CocoDetection and BaseCellCOCO.

    Args:
        root (str): Root directory of the dataset.
        fold (Union[str, int]): Name or number of the fold (e.g., "train", "val", "test").
        transforms: Optional transformations to apply to the dataset samples.
        sigma (float, optional): Standard deviation used if generating Gaussian masks for centroids.
    """

    def __init__(
        self,
        root: str,
        fold,
        transforms=None,
        sigma: float = 5.0
    ) -> None:
        self.root = root
        self.sigma = sigma

        fold = fold if isinstance(fold, str) else f"fold{fold}"
        self.fold = fold

        img_folder = osp.join(root, fold, "images")
        ann_file = osp.join(root, fold, "annotations.json")

        super().__init__(img_folder, ann_file, transforms=transforms)

    @property
    def num_classes(self) -> int:
        """Number of classes in PanNuke (5)."""
        return 5

    @property
    def class_names(self) -> list:
        """Returns the PanNuke nucleus class names."""
        return PANNUKE_NUCLEI

    def image_size(self, image_id=None, idx=None) -> torch.Tensor:
        """
        Returns the image size (256x256) for PanNuke data.

        Args:
            image_id (int, optional): ID of the image in COCO format.
            idx (int, optional): Dataset index.

        Returns:
            torch.Tensor: A 1D tensor containing (height, width).
        """
        return torch.tensor([256, 256])

    def __len__(self) -> int:
        """Total number of samples in the dataset."""
        return super().__len__()

    def __getitem__(self, idx: int):
        """
        Retrieve the (image, target) pair at the given index.

        Args:
            idx (int): Index in the dataset.

        Returns:
            (PIL.Image.Image, List[Dict]): The image and its annotations.
        """
        img, tgt = super().__getitem__(idx)
        height, width = img.size

        if len(tgt) > 0:
            tgt = [t for t in tgt if t["area"] > 0 and len(t["segmentation"][0]) > 4]

        if len(tgt) == 0:
            tgt = [
                dict(
                    id=-1,
                    image_id=idx,
                    category_id=-1,
                    bbox=[-1, -1, -1, -1],
                    area=255 * 255,
                    segmentation=[[0, 0, 0, 255, 255, 255, 255, 0]],
                    centroid=[128, 128],
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
        Generates a Gaussian mask for each centroid in the target annotations.

        Args:
            tgt (List[Dict]): List of COCO-style annotations that may contain 'centroid' fields.
            height (int): Height of the resulting mask.
            width (int): Width of the resulting mask.

        Returns:
            torch.Tensor: A (H, W) float32 tensor with Gaussian peaks at centroids.
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
        Load and return a raw image from disk, converting it to a float32 tensor.

        Args:
            image_id (int, optional): COCO image ID.
            idx (int, optional): Dataset index.

        Returns:
            torch.Tensor: Image tensor with shape (C, H, W) and float32 type.
        """
        assert image_id is not None or idx is not None, "Provide either image_id or idx."
        if image_id is None:
            image_id = self.ids[idx]

        img = self._load_image(image_id)
        transforms_ = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        img = transforms_(img)
        return img


def build_pannuke_dataset(cfg: dict, split: str = "train"):
    """
    Build a PanNuke dataset instance based on the given config and split.

    Args:
        cfg (dict): Configuration dictionary with dataset details.
        split (str): Dataset split ("train" or "test").

    Returns:
        Dataset wrapped for transforms_v2 usage.
    """
    from .transforms import build_transforms

    root = cfg["dataset"][split]["root"]
    num_classes = cfg["dataset"][split]["num_classes"]
    transforms_ = build_transforms(cfg, split, is_train=(split == "train"))

    if num_classes == 1:
        dataset = DetectionWrapper(Pannuke)(root, cfg["dataset"][split]["fold"], transforms=transforms_)
    else:
        dataset = Pannuke(root, cfg["dataset"][split]["fold"], transforms=transforms_)

    dataset = datasets.wrap_dataset_for_transforms_v2(
        dataset,
        target_keys=("image_id", "masks", "boxes", "labels")
    )
    return dataset


def pannuke2coco(data_dir: str, fold, out_dir: str) -> None:
    """
    Convert the PanNuke data into COCO format with centroid annotations.

    Args:
        data_dir (str): Directory containing fold images/masks subfolders.
        fold: Either fold number or name (e.g., "fold1").
        out_dir (str): Output directory to store images and COCO JSON.
    """
    print("Converting Pannuke to COCO format...")

    img_path = osp.join(data_dir, f"{fold}", "images", "images.npy")
    mask_path = osp.join(data_dir, f"{fold}", "masks", "masks.npy")

    images = np.load(img_path)
    masks = np.load(mask_path)[:, :, :, :-1]  # Exclude background channel

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
            uq_instance_ids = np.unique(mask_i[:, :, lbl])[1:]
            for instance_id in uq_instance_ids:
                coords = np.where(mask_i[:, :, lbl] == instance_id)
                xmin = int(np.min(coords[1]))
                ymin = int(np.min(coords[0]))
                xmax = int(np.max(coords[1]))
                ymax = int(np.max(coords[0]))

                centroid_x = int(np.mean(coords[1]))
                centroid_y = int(np.mean(coords[0]))
                centroid = [centroid_x, centroid_y]

                bin_mask = mask_i[:, :, lbl] == instance_id
                contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    categories = [dict(id=k + 1, name=v) for k, v in enumerate(PANNUKE_NUCLEI)]
    coco_format_json = dict(
        images=ls_images,
        annotations=ls_annots,
        categories=categories
    )

    out_json_path = osp.join(out_fold_dir, "annotations.json")
    with open(out_json_path, "w") as f:
        json.dump(coco_format_json, f)

    print(f"COCO format annotations with centroids saved to {out_json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--fold", type=str, default="1")
    parser.add_argument("--out-dir", type=str, required=True)

    args = parser.parse_args()
    pannuke2coco(args.data_dir, args.fold, args.out_dir)
