import math
import sys
import time
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import dual_unet.utils.misc as utils
from dual_unet.eval import MultiTaskEvaluationMetric, MultiTaskEvaluationMetric_all  # Import other evaluation metrics if needed


def train_one_epoch(
    cfg: dict,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    thresholds: List[float] = [0.5],
    max_pair_distance: float = 12.0,
    max_norm: float = 0
) -> dict:
    """
    Train the model for one epoch.

    This function:
      - Sets the model and criterion to training mode,
      - Iterates over the provided data_loader,
      - Computes the forward pass and loss,
      - Performs backpropagation and optimizer step,
      - Tracks metrics in a MetricLogger.

    Optionally, it computes certain evaluation metrics at the end of the epoch 
    (depending on the config settings).

    Args:
        cfg (dict): Configuration dictionary with fields like:
            - optimizer.epochs (int): Total number of epochs.
            - evaluation.interval (int): Interval at which to compute / log metrics.
        model (nn.Module): The model to train.
        criterion (nn.Module): Loss function / criterion.
        data_loader (Iterable): Data loader providing (samples, targets).
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): The device (CPU/GPU) to run computations on.
        epoch (int): Current epoch index (for logging).
        thresholds (List[float]): Thresholds for metrics that require binarization.
        max_pair_distance (float): Max pairing distance used by the evaluation metric.
        max_norm (float): If > 0, max norm for gradient clipping.

    Returns:
        dict: Dictionary containing average losses (and possibly some metrics) for this epoch.
    """
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 1

    # Optionally define metrics to track
    metrics = {
        "f": MultiTaskEvaluationMetric(
            num_classes=data_loader.dataset.num_classes,
            thresholds=thresholds,
            max_pair_distance=max_pair_distance,
            class_names=data_loader.dataset.class_names,
            train=True  # We're in training mode; some metrics may differ if train=False
        )
    }

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        optimizer.zero_grad()

        # Forward pass
        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss_value = loss.item()

        # Backward pass
        loss.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        optimizer.step()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Update logs
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # Optionally update metrics (e.g., to track them within the epoch)
        # Usually done less frequently or at epoch's end. 
        # Example (if we want per-iteration metrics, but that can slow training):
        # predictions = [
        #     {
        #         "segmentation_mask": torch.softmax(outputs[0][i], dim=0),
        #         "centroid_gaussian": outputs[1][i],
        #         "image": samples[i],
        #     }
        #     for i in range(len(outputs[0]))
        # ]
        # for k in metrics:
        #     metrics[k].update(predictions, targets)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    losses = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Optionally compute metrics at the end of the epoch
    # e.g., if epoch is final or at an evaluation interval
    if epoch == cfg["optimizer"]["epochs"] or (epoch % cfg["evaluation"]["interval"] == 0):
        # Example usage of the metrics if needed:
        # final_metrics = {k: metrics[k].compute() for k in metrics}
        # return {**losses, **final_metrics}
        pass

    return losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    thresholds: List[float] = [0.5],
    max_pair_distance: float = 12.0,
    th: float = 0.15
) -> dict:
    """
    Evaluate the model on a given data_loader, computing a multi-task metric.

    Args:
        model (nn.Module): Model to evaluate (segmentation + counting).
        criterion (nn.Module): The loss function, if we need to compute a validation loss.
        data_loader (Iterable): The validation/test data loader.
        device (torch.device): CPU or GPU device.
        thresholds (List[float]): Threshold list for metrics that require binarization.
        max_pair_distance (float): Pairing distance for centroid matching in metric.
        th (float): Threshold used internally in the metric (e.g. for local maxima detection).

    Returns:
        dict: Dictionary of losses and metrics (e.g., F-score).
    """
    model.eval()
    criterion.eval()
    print("Model and criterion set to eval mode")

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # We might track e.g. a MultiTaskEvaluationMetric
    metrics = {
        "f": MultiTaskEvaluationMetric(
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
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        outputs = model(samples)
        loss = criterion(outputs, targets)
        metric_logger.update(loss=loss.item())

        # Build prediction dict for metric
        predictions = [
            {
                "segmentation_mask": torch.softmax(outputs[0][i], dim=0),
                "centroid_gaussian": outputs[1][i],
                "image": samples[i],
            }
            for i in range(len(outputs[0]))
        ]
        for k in metrics:
            metrics[k].update(predictions, targets)

    metric_logger.synchronize_between_processes()
    losses = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # Compute metrics
    metrics = {k: metrics[k].compute() for k in metrics}

    return {**losses, **metrics}


@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    thresholds: List[float] = [0.5],
    max_pair_distance: float = 12.0,
    output_sufix: str = "",
    train: bool = False,
    th: float = 0.15
) -> dict:
    """
    Similar to `evaluate`, but uses a MultiTaskEvaluationMetric_all for extended metrics or different logging.

    Args:
        model (nn.Module): Model to evaluate.
        criterion (nn.Module): Loss function (if we compute a validation loss).
        data_loader (Iterable): Data loader for evaluation set.
        device (torch.device): CPU or GPU.
        thresholds (List[float]): Threshold for binarization in the metric.
        max_pair_distance (float): Pair distance for matching centroids.
        output_sufix (str): A suffix used for naming output logs or files if needed.
        train (bool): If True, might adjust how metrics are computed internally.
        th (float): Additional threshold used in the evaluation metric.

    Returns:
        dict: Contains evaluation metrics (like F-score, etc.).
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    metrics = {
        "f": MultiTaskEvaluationMetric_all(
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
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]
        outputs = model(samples)
        # We can collect or log the criterion loss if needed:
        # loss = criterion(outputs, targets)
        # metric_logger.update(loss=loss.item())

        predictions = [
            {
                "segmentation_mask": torch.softmax(outputs[0][i], dim=0),
                "centroid_gaussian": outputs[1][i],
                "image": samples[i],
            }
            for i in range(len(outputs[0]))
        ]
        for k in metrics:
            metrics[k].update(predictions, targets)

    metric_logger.synchronize_between_processes()
    # losses = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metrics = {k: metrics[k].compute() for k in metrics}

    return metrics