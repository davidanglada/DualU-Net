import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from torchvision import datasets

from .pannuke import build_pannuke_dataset
from .consep import build_consep_dataset
from .ki67 import build_ki67_dataset
from ..utils.misc import nested_tensor_from_tensor_list


def build_dataset(cfg: dict, split: str = "train"):
    """
    Build a dataset according to the name specified in `cfg['dataset'][split]['name']`.

    Args:
        cfg (dict): Configuration dictionary containing dataset info.
        split (str): Split to build ("train", "val", "test", etc.).

    Returns:
        Dataset: An instance of a dataset object.
    """
    dataset_name = cfg["dataset"][split]["name"]
    if dataset_name == "pannuke":
        dataset = build_pannuke_dataset(cfg, split=split)
    elif dataset_name == "consep":
        dataset = build_consep_dataset(cfg, split=split)
    elif dataset_name == "ki67":
        dataset = build_ki67_dataset(cfg, split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Expects batch to be a list of (image, target) pairs. Returns a stack of images 
    (assuming they have the same shape) and a list of targets.

    Args:
        batch (List[Tuple[Tensor, Dict]]): A list of image-target pairs.

    Returns:
        (Tensor, List[Dict]): Stacked images and list of targets.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)  # Convert list of images to a single tensor
    return images, targets


def build_loader(
    cfg: dict, 
    dataset, 
    split: str = "train", 
    collate_fn=collate_fn
):
    """
    Build a DataLoader for the given dataset.

    Args:
        cfg (dict): Configuration dictionary with 'loader' and distribution info.
        dataset: The dataset instance to load.
        split (str): "train", "val", "test", etc.
        collate_fn (callable): Function to collate a batch, default is collate_fn above.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    _loader_cfg = cfg["loader"][split]
    sampler = None

    if cfg["distributed"]:
        if split in ["train", "val", "infer"]:
            sampler = DistributedSampler(
                dataset,
                shuffle=_loader_cfg["shuffle"],
                num_replicas=cfg["world_size"],
                rank=cfg["rank"],
            )
        else:
            from .loader import DistributedSamplerNoDuplicate
            sampler = DistributedSamplerNoDuplicate(
                dataset,
                shuffle=_loader_cfg["shuffle"],
                num_replicas=cfg["world_size"],
                rank=cfg["rank"],
            )
    else:
        if _loader_cfg["shuffle"]:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=_loader_cfg["batch_size"],
        num_workers=_loader_cfg["num_workers"],
        drop_last=_loader_cfg["drop_last"],
        collate_fn=collate_fn
    )

    return loader


def compute_class_weights_with_background(
    dataset, 
    num_classes: int, 
    background_importance_factor: float = 1.0
) -> torch.Tensor:
    """
    Compute class weights (including background) with emphasis on the background.

    Args:
        dataset: Dataset (e.g., Consep) providing segmentation data.
        num_classes (int): Number of classes (excluding background).
        background_importance_factor (float): Weight multiplier for the background class.

    Returns:
        torch.Tensor: A 1D tensor of class weights, including background as the 0th index.
    """
    total_classes = num_classes + 1
    class_counts = np.zeros(total_classes, dtype=np.float64)

    for _, t in dataset:
        mask = torch.argmax(t["segmentation_mask"], dim=0)
        mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        for i in range(total_classes):
            class_counts[i] += np.sum(mask == i)

    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (total_classes * class_counts + 1e-6)

    # Emphasize background
    class_weights[0] *= background_importance_factor

    # Normalize to sum to 1
    class_weights /= class_weights.sum()

    return torch.tensor(class_weights, dtype=torch.float32)


def compute_class_weights_no_background(
    dataset, 
    num_classes: int, 
    background_importance_factor: float = 1.0
) -> torch.Tensor:
    """
    Compute class weights for a dataset that focuses on the labeled classes (no background mask).

    Args:
        dataset: Dataset providing segmentation or class label data.
        num_classes (int): Number of actual classes.
        background_importance_factor (float): Not used here, but kept for API consistency.

    Returns:
        torch.Tensor: A 1D tensor of class weights for the specified classes.
    """
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for _, t in dataset:
        for label in t["labels"]:
            class_counts[label - 1] += 1

    total_count = np.sum(class_counts)
    class_weights = total_count / (num_classes * class_counts + 1e-6)
    class_weights /= class_weights.sum()

    return torch.tensor(class_weights, dtype=torch.float32)
