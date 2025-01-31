
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from torchvision import datasets

from .pannuke import build_pannuke_dataset
from .consep import build_consep_dataset
from .kumar import build_kumar_dataset
from .ki67 import build_ki67_dataset
# from .monuseg import build_monuseg_dataset
# from .dataset import build_cell_dataset
from ..utils.misc import nested_tensor_from_tensor_list
import torch
import numpy as np

def build_dataset(cfg, split='train'):
    if cfg['dataset'][split]['name'] == 'pannuke':
        dataset = build_pannuke_dataset(cfg, split=split)
    elif cfg['dataset'][split]['name'] == 'consep':
        dataset = build_consep_dataset(cfg, split=split)
    elif cfg['dataset'][split]['name'] ==  'kumar':
        dataset = build_kumar_dataset(cfg, split=split)
    elif cfg['dataset'][split]['name'] == 'ki67':
        dataset = build_ki67_dataset(cfg, split=split)
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']['split']['name']}")
    
    return dataset

#collate_fn = lambda batch : tuple(zip(*batch))
# def collate_fn(batch):
#     batch = list(zip(*batch))
#     batch[0] = nested_tensor_from_tensor_list(batch[0])
#     return batch

def collate_fn(batch):
    # Split the batch into images and targets
    images, targets = zip(*batch)
    
    # Make sure the images are tensors, and return them without nesting
    images = torch.stack(images)  # Stack them into a single tensor (assumes same size)
    
    return images, targets

def build_loader(cfg, dataset, split='train', collate_fn=collate_fn):
    _loader_cfg = cfg['loader'][split]
    
    # create sampler
    sampler = None
    if cfg['distributed']:
        if split in ['train','val','infer']:
            print("Using DistributedSampler")
            sampler = DistributedSampler(dataset, 
                                     shuffle=_loader_cfg['shuffle'],
                                     num_replicas=cfg['world_size'],
                                     rank=cfg['rank'])
        else:
            from .loader import DistributedSamplerNoDuplicate
            sampler = DistributedSamplerNoDuplicate(dataset, 
                                     shuffle=_loader_cfg['shuffle'],
                                     num_replicas=cfg['world_size'],
                                     rank=cfg['rank'])
    else:
        sampler = RandomSampler(dataset) if _loader_cfg['shuffle'] else SequentialSampler(dataset)
    # create data loader
    # print("Sampler created")
    loader = DataLoader(dataset, sampler=sampler,
                        batch_size=_loader_cfg['batch_size'],
                        num_workers=_loader_cfg['num_workers'],
                        drop_last=_loader_cfg['drop_last'],
                        collate_fn=collate_fn)#, pin_memory=split=="infer")

    # print("Loader created")

    # loader = DataLoader(dataset, shuffle=_loader_cfg['shuffle'],
    #                     batch_size=_loader_cfg['batch_size'],
    #                     num_workers=_loader_cfg['num_workers'],
    #                     drop_last=_loader_cfg['drop_last'],
    #                     collate_fn=collate_fn)#, pin_memory=split=="infer")
    # pin memory only for inference as if done in train or val, converts tv_tensors to standard torch tensors.
    return loader


import numpy as np
import torch

def compute_class_weights_with_background(dataset, num_classes, background_importance_factor=1.0):
    """
    Compute class weights with emphasis on the background, while keeping class-dependent weights for each cell class.
    
    Args:
        dataset: The dataset (e.g., Consep dataset).
        num_classes: Number of cell classes in the dataset (excluding background).
        background_importance_factor: Weight multiplier for the background class to increase its importance.
    
    Returns:
        class_weights: A tensor of weights for each class, with increased importance for the background.
    """
    # Total classes include the background as the first class
    num_classes = num_classes + 1
    class_counts = np.zeros(num_classes)  # Store counts for each class, including background

    # Count occurrences of each class in the dataset
    for _, t in dataset:
        mask = torch.argmax(t['segmentation_mask'], dim=0)  # Assuming segmentation mask
        mask = mask.numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        for i in range(num_classes):  # Count pixels for each class
            class_counts[i] += np.sum(mask == i)
    
    # Total pixel count for normalization
    total_pixels = np.sum(class_counts)
    
    # Compute initial weights inversely proportional to each class's pixel count
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)

    # Apply background importance factor to the background class
    class_weights[0] *= background_importance_factor  # Increase background weight

    # Normalize weights to sum to 1
    class_weights /= class_weights.sum()
    
    return torch.tensor(class_weights, dtype=torch.float32)


def compute_class_weights_no_background(dataset, num_classes, background_importance_factor=1.0):
    """
    Compute class weights with emphasis on the background, while keeping class-dependent weights for each cell class.
    
    Args:
        dataset: The dataset (e.g., Consep dataset).
        num_classes: Number of cell classes in the dataset (excluding background).
        background_importance_factor: Weight multiplier for the background class to increase its importance.
    
    Returns:
        class_weights: A tensor of weights for each class, with increased importance for the background.
    """
    # Total classes include the background as the first class
    num_classes = num_classes
    class_counts = np.zeros(num_classes)  # Store counts for each class, including background

    # Count occurrences of each class in the dataset
    for _, t in dataset:
        for label in t['labels']:
            class_counts[label-1] += 1
        
    # Total pixel count for normalization
    total_pixels = np.sum(class_counts)
    
    # Compute initial weights inversely proportional to each class's pixel count
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)

    # Normalize weights to sum to 1
    class_weights /= class_weights.sum()
    
    return torch.tensor(class_weights, dtype=torch.float32)
