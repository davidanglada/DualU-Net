# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""

import math
import sys
import time
from typing import Iterable, List, Dict, Any

import torch.nn as nn
import torch.distributed as dist

import torch
from torch.cuda.amp import autocast, GradScaler

import dual_unet.utils.misc as utils
from dual_unet.eval import MultiTaskEvaluationMetric_all

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """ Reduce a tensor across all processes to get the average loss. """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def train_one_epoch(
    cfg: Dict[str, Any],
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    thresholds: List[float] = [0.5],
    max_pair_distance: float = 12.0,
    max_norm: float = 0
) -> Dict[str, float]:
    """
    Train the model for one epoch with optimized distributed training.

    Args:
        cfg: Configuration dictionary.
        model: The model to train.
        criterion: The loss function.
        data_loader: DataLoader for training data.
        optimizer: Optimizer for training.
        device: Device to run the training on.
        epoch: Current epoch number.
        thresholds: List of thresholds for evaluation.
        max_pair_distance: Maximum distance for pairing centroids.
        max_norm: Maximum norm for gradient clipping.

    Returns:
        Dictionary of training losses and metrics.
    """
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 1

    scaler = GradScaler()

    # Ensure unique data per GPU
    if isinstance(data_loader.sampler, torch.utils.data.distributed.DistributedSampler):
        data_loader.sampler.set_epoch(epoch)

    metrics = {
        'f': MultiTaskEvaluationMetric_all(
            num_classes=data_loader.dataset.num_classes,
            thresholds=thresholds,
            max_pair_distance=max_pair_distance,
            class_names=data_loader.dataset.class_names,
            dataset=cfg['dataset']['train']['name'],
            train=True
        )
    }

    all_predictions = []
    all_targets = []

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        # Prevent NaN propagation
        if not torch.isfinite(loss):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        # Reduce loss across all GPUs
        loss_value = reduce_tensor(loss).item()

        # Use no_sync() to reduce communication overhead, sync every 4 steps
        if i % 4 != 0 and dist.get_world_size() > 1:
            with model.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        if max_norm > 0:
            scaler.unscale_(optimizer)
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # Collect predictions for evaluation at the end of the epoch
        if epoch in [cfg['optimizer']['epochs']] or epoch % cfg['evaluation']['interval'] == 0:
            all_predictions.extend([
                {
                    'segmentation_mask': torch.softmax(outputs[0][i], dim=0),
                    'centroid_gaussian': outputs[1][i],
                    'image': samples[i],
                }
                for i in range(len(outputs[0]))
            ])
            all_targets.extend(targets)

    # Synchronize metrics only once per epoch
    if epoch in [cfg['optimizer']['epochs']] or epoch % cfg['evaluation']['interval'] == 0:
        for k in metrics:
            metrics[k].update(all_predictions, all_targets)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # Reduce losses across all GPUs before returning
    losses = {k: reduce_tensor(torch.tensor(meter.global_avg, device=device)).item() for k, meter in metric_logger.meters.items()}

    if epoch in [cfg['optimizer']['epochs']] or epoch % cfg['evaluation']['interval'] == 0:
        metrics = {k: metrics[k].compute() for k in metrics}
        return {**losses, **metrics}

    return {**losses}

@torch.no_grad()
def reduce_tensor_eval(tensor: torch.Tensor) -> torch.Tensor:
    """ Reduce a tensor across all processes to get the average loss. """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

@torch.no_grad()
def evaluate(
    cfg: Dict[str, Any],
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    thresholds: List[float] = [0.5],
    max_pair_distance: float = 12.0,
    th: float = 0.15
) -> Dict[str, float]:
    """
    Evaluate the model with optimized distributed evaluation.

    Args:
        cfg: Configuration dictionary.
        model: The model to evaluate.
        criterion: The loss function.
        data_loader: DataLoader for evaluation data.
        device: Device to run the evaluation on.
        thresholds: List of thresholds for evaluation.
        max_pair_distance: Maximum distance for pairing centroids.
        th: Threshold for local maxima detection.

    Returns:
        Dictionary of evaluation losses and metrics.
    """
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Ensure unique data per GPU
    if isinstance(data_loader.sampler, torch.utils.data.distributed.DistributedSampler):
        data_loader.sampler.set_epoch(0)

    metrics = {
        'f': MultiTaskEvaluationMetric_all(
            num_classes=data_loader.dataset.num_classes,
            thresholds=thresholds,
            max_pair_distance=max_pair_distance,
            class_names=data_loader.dataset.class_names,
            dataset=cfg['dataset']['val']['name'],
            train=True,
            th=th
        )
    }

    all_predictions = []
    all_targets = []
    total_loss = torch.tensor(0.0, device=device)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss = criterion(outputs, targets)
        total_loss += loss  # Accumulate loss before reduction

        predictions = [
            {
                'segmentation_mask': torch.softmax(outputs[0][i], dim=0),
                'centroid_gaussian': outputs[1][i],
                'image': samples[i],
            }
            for i in range(len(outputs[0]))
        ]
        all_predictions.extend(predictions)
        all_targets.extend(targets)

    # Synchronize and reduce loss across GPUs
    total_loss = reduce_tensor_eval(total_loss) / len(data_loader)

    # Compute metrics once at the end of evaluation
    for k in metrics:
        metrics[k].update(all_predictions, all_targets)

    metric_logger.synchronize_between_processes()
    losses = {k: reduce_tensor(torch.tensor(meter.global_avg, device=device)).item() for k, meter in metric_logger.meters.items()}
    metrics = {k: metrics[k].compute() for k in metrics}

    return {**losses, "loss": total_loss.item(), **metrics}

@torch.no_grad()
def evaluate_test(
    cfg: Dict[str, Any],
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    thresholds: List[float] = [0.5],
    max_pair_distance: float = 12.0,
    output_sufix: str = '',
    train: bool = False,
    th: float = 0.15
) -> Dict[str, float]:
    """
    Evaluate the model for testing with optimized distributed evaluation.

    Args:
        cfg: Configuration dictionary.
        model: The model to evaluate.
        criterion: The loss function.
        data_loader: DataLoader for evaluation data.
        device: Device to run the evaluation on.
        thresholds: List of thresholds for evaluation.
        max_pair_distance: Maximum distance for pairing centroids.
        output_sufix: Suffix for output files.
        train: Whether the model is in training mode.
        th: Threshold for local maxima detection.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Ensure unique data per GPU
    if isinstance(data_loader.sampler, torch.utils.data.distributed.DistributedSampler):
        data_loader.sampler.set_epoch(0)

    metrics = {
        'f': MultiTaskEvaluationMetric_all(
            num_classes=data_loader.dataset.num_classes,
            thresholds=thresholds,
            max_pair_distance=max_pair_distance,
            class_names=data_loader.dataset.class_names,
            dataset=cfg['dataset']['test']['name'],
            train=train,
            th=th,
            output_sufix=output_sufix
        )
    }

    all_predictions = []
    all_targets = []
    total_loss = torch.tensor(0.0, device=device)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss = criterion(outputs, targets)
        total_loss += loss  # Accumulate loss before reduction

        predictions = [
            {
                'segmentation_mask': torch.softmax(outputs[0][i], dim=0),
                'centroid_gaussian': outputs[1][i],
                'image': samples[i],
            }
            for i in range(len(outputs[0]))
        ]
        all_predictions.extend(predictions)
        all_targets.extend(targets)

    # Synchronize and reduce loss across GPUs
    total_loss = reduce_tensor(total_loss) / len(data_loader)

    # Compute metrics once at the end of evaluation
    for k in metrics:
        metrics[k].update(all_predictions, all_targets)

    metric_logger.synchronize_between_processes()
    metrics = {k: metrics[k].compute() for k in metrics}

    return {"loss": total_loss.item(), **metrics}


@torch.no_grad()
def evaluate_test_time(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    thresholds: List[float] = [0.5],
    max_pair_distance: float = 12.0,
    output_sufix: str = '',
    train: bool = False,
    th: float = 0.15
) -> (Dict[str, float], float):
    """
    Evaluate the model and measure the time taken.

    Args:
        model: The model to evaluate.
        criterion: The loss function.
        data_loader: DataLoader for evaluation data.
        device: Device to run the evaluation on.
        thresholds: List of thresholds for evaluation.
        max_pair_distance: Maximum distance for pairing centroids.
        output_sufix: Suffix for output files.
        train: Whether the model is in training mode.
        th: Threshold for local maxima detection.

    Returns:
        Dictionary of evaluation metrics and total time taken.
    """
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    metrics = {
        'f': MultiTaskEvaluationMetric_all(
            num_classes=data_loader.dataset.num_classes,
            thresholds=thresholds,
            max_pair_distance=max_pair_distance,
            class_names=data_loader.dataset.class_names,
            train=train,
            th=th,
            output_sufix=output_sufix
        )
    }

    start_time = time.time()
    for samples, _ in data_loader:
        samples = samples.to(device)
        outputs = model(samples)

    end_time = time.time()
    total_time = end_time - start_time

    return {**metrics}, total_time

@torch.no_grad()
def evaluate_test_p2c(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    thresholds: List[float] = [0.5],
    max_pair_distance: float = 12.0,
    output_sufix: str = '',
    train: bool = False,
    th: float = 0.15
) -> Dict[str, float]:
    """
    Evaluate the model for testing with p2c metric.

    Args:
        model: The model to evaluate.
        criterion: The loss function.
        data_loader: DataLoader for evaluation data.
        device: Device to run the evaluation on.
        thresholds: List of thresholds for evaluation.
        max_pair_distance: Maximum distance for pairing centroids.
        output_sufix: Suffix for output files.
        train: Whether the model is in training mode.
        th: Threshold for local maxima detection.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    metrics = {
        'f': MultiTaskEvaluationMetric_all(
            num_classes=data_loader.dataset.num_classes,
            thresholds=thresholds,
            max_pair_distance=max_pair_distance,
            class_names=data_loader.dataset.class_names,
            train=train,
            th=th,
            output_sufix=output_sufix
        )
    }

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)

        predictions = [
            {
                'segmentation_mask': torch.softmax(outputs[0][i], dim=0),
                'centroid_gaussian': outputs[1][i],
                'image': samples[i],
            }
            for i in range(len(outputs[0]))
        ]
        for k in metrics:
            metrics[k].update(predictions, targets)

    metric_logger.synchronize_between_processes()
    metrics = {k: metrics[k].compute() for k in metrics}

    return {**metrics}