from typing import Any, Dict, Tuple, List

import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from torchvision import datasets  # noqa: F401

from .pannuke import build_pannuke_dataset
from .consep import build_consep_dataset
from .ki67 import build_ki67_dataset
from ..utils.misc import nested_tensor_from_tensor_list, seed_worker


def build_dataset(cfg: Dict[str, Any], split: str = 'train') -> Any:
    """
    Build the dataset specified in the configuration for a given split.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing dataset information.
        split (str): The dataset split ('train', 'val', 'test', etc.).

    Returns:
        Any: The constructed dataset object.
    """
    if cfg['dataset'][split]['name'] == 'pannuke':
        dataset = build_pannuke_dataset(cfg, split=split)
    elif cfg['dataset'][split]['name'] == 'consep':
        dataset = build_consep_dataset(cfg, split=split)
    elif cfg['dataset'][split]['name'] == 'ki67':
        dataset = build_ki67_dataset(cfg, split=split)
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']['split']['name']}")
    return dataset


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]]
) -> Tuple[torch.Tensor, Tuple[Dict[str, Any], ...]]:
    """
    Custom collate function for combining a list of samples into a batch.

    Args:
        batch (List[Tuple[torch.Tensor, Dict[str, Any]]]): A batch of data, where each element
            is a tuple (image_tensor, target_dict).

    Returns:
        Tuple[torch.Tensor, Tuple[Dict[str, Any], ...]]: A tuple containing a stacked
        tensor of images and the corresponding tuple of target dictionaries.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)  # Assumes all images are the same size
    return images, targets


def build_loader(
    cfg: Dict[str, Any],
    dataset: Any,
    split: str = 'train',
    collate_fn: Any = collate_fn
) -> DataLoader:
    """
    Build a DataLoader for the provided dataset and configuration.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing loader settings.
        dataset (Any): The dataset to load.
        split (str): Which dataset split to load (e.g., 'train', 'val', 'test').
        collate_fn (Callable): Collate function used to combine individual samples.

    Returns:
        DataLoader: A PyTorch DataLoader configured with the specified sampler, batch size,
        and other loader settings.
    """
    _loader_cfg = cfg['loader'][split]

    # Create sampler
    sampler = None
    if cfg['distributed']:
        if split in ['train', 'val', 'infer']:
            print("Using DistributedSampler")
            sampler = DistributedSampler(
                dataset,
                shuffle=_loader_cfg['shuffle'],
                num_replicas=cfg['world_size'],
                rank=cfg['rank']
            )
        else:
            from .loader import DistributedSamplerNoDuplicate
            sampler = DistributedSamplerNoDuplicate(
                dataset,
                shuffle=_loader_cfg['shuffle'],
                num_replicas=cfg['world_size'],
                rank=cfg['rank']
            )
    else:
        sampler = RandomSampler(dataset) if _loader_cfg['shuffle'] else SequentialSampler(dataset)

    g = torch.Generator()
    g.manual_seed(42)

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=_loader_cfg['batch_size'],
        num_workers=_loader_cfg['num_workers'],
        drop_last=_loader_cfg['drop_last'],
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g
    )
    return loader


def compute_class_weights_with_background(
    dataset: Any,
    num_classes: int,
    background_importance_factor: float = 1.0
) -> torch.Tensor:
    """
    Compute class weights that include the background as its own class, with an optional
    emphasis on the background class.

    This function calculates pixel-wise counts of each class (including background) across
    the dataset segmentation masks and then computes weights inversely proportional to
    those counts. The background class is multiplied by a specified factor to increase
    its importance.

    Args:
        dataset (Any): The dataset containing images and segmentation masks.
        num_classes (int): Number of cell classes (excluding background).
        background_importance_factor (float): Weight multiplier for the background class.

    Returns:
        torch.Tensor: A 1D tensor of class weights for (background + num_classes).
    """
    total_classes = num_classes + 1
    class_counts = np.zeros(total_classes)

    # Count pixel occurrences for each class index
    for _, t in dataset:
        mask = torch.argmax(t['segmentation_mask'], dim=0)
        mask = mask.numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        for i in range(total_classes):
            class_counts[i] += np.sum(mask == i)

    # Compute initial weights (inverse of frequency)
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (total_classes * class_counts + 1e-6)

    # Apply background importance factor
    class_weights[0] *= background_importance_factor

    # Normalize so that the weights sum to 1
    class_weights /= class_weights.sum()

    return torch.tensor(class_weights, dtype=torch.float32)


def compute_class_weights_no_background(
    dataset: Any,
    num_classes: int,
    background_importance_factor: float = 1.0  # Not used, but kept for signature consistency
) -> torch.Tensor:
    """
    Compute class weights for classes only (no separate background class).

    This function calculates label occurrences for each class in the dataset and then
    computes weights inversely proportional to those counts.

    Args:
        dataset (Any): The dataset containing images and label information.
        num_classes (int): Number of cell classes (excluding background).
        background_importance_factor (float): Unused parameter, kept for consistency.

    Returns:
        torch.Tensor: A 1D tensor of class weights for the given number of classes.
    """
    class_counts = np.zeros(num_classes)

    # Count how many times each label occurs
    for _, t in dataset:
        for label in t['labels']:
            class_counts[label - 1] += 1

    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)

    class_weights /= class_weights.sum()
    return torch.tensor(class_weights, dtype=torch.float32)
