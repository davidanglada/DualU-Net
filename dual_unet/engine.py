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

import torch
from torch.cuda.amp import autocast, GradScaler

import dual_unet.utils.misc as utils
from dual_unet.eval import MultiTaskEvaluationMetric_all

def train_one_epoch(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    thresholds: List[float] = [0.5],
    max_pair_distance: float = 12.0,
    max_norm: float = 0
) -> Dict[str, float]:
    """
    Train the model for one epoch.

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

    metrics = {
        'f': MultiTaskEvaluationMetric_all(
            num_classes=data_loader.dataset.num_classes,
            thresholds=thresholds,
            max_pair_distance=max_pair_distance,
            class_names=data_loader.dataset.class_names,
            train=True
        )
    }

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        scaler.scale(loss).backward()
        if max_norm > 0:
            scaler.unscale_(optimizer)
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        if epoch in [cfg['optimizer']['epochs']] or epoch % cfg['evaluation']['interval'] == 0:
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
    print("Averaged stats:", metric_logger)
    losses = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if epoch in [cfg['optimizer']['epochs']] or epoch % cfg['evaluation']['interval'] == 0:
        metrics = {k: metrics[k].compute() for k in metrics}
        return {**losses, **metrics}

    return {**losses}

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    thresholds: List[float] = [0.5],
    max_pair_distance: float = 12.0,
    th: float = 0.15
) -> Dict[str, float]:
    """
    Evaluate the model.

    Args:
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

    metrics = {
        'f': MultiTaskEvaluationMetric_all(
            num_classes=data_loader.dataset.num_classes,
            thresholds=thresholds,
            max_pair_distance=max_pair_distance,
            class_names=data_loader.dataset.class_names,
            train=True,
            th=th
        )
    }

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss = criterion(outputs, targets)
        metric_logger.update(loss=loss.item())

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
    losses = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metrics = {k: metrics[k].compute() for k in metrics}

    return {**losses, **metrics}

@torch.no_grad()
def evaluate_test(
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
    Evaluate the model for testing.

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