import torch
from typing import Optional, List


def _take_channels(*xs: torch.Tensor, ignore_channels: Optional[List[int]] = None):
    """
    Utility function to select specific channels from the input tensors.
    If `ignore_channels` is provided, the given indices are removed.

    Args:
        *xs (torch.Tensor): One or more tensors with shape (N, C, H, W).
        ignore_channels (List[int], optional): List of channel indices to ignore.

    Returns:
        tuple: A tuple of the input tensors with the specified channels removed.
    """
    if ignore_channels is None:
        return xs
    else:
        # We assume all input tensors have the same shape and channel dimension
        channels = [
            ch for ch in range(xs[0].shape[1]) if ch not in ignore_channels
        ]
        # For each tensor, select only the allowed channels
        xs = [
            torch.index_select(
                x, dim=1, index=torch.tensor(channels).to(x.device)
            )
            for x in xs
        ]
        return xs


def _threshold(x: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
    """
    Apply a threshold to the input tensor, turning values > threshold into 1, and others into 0.
    If `threshold` is None, return x unchanged.

    Args:
        x (torch.Tensor): Input tensor.
        threshold (float, optional): Threshold value for binarization.

    Returns:
        torch.Tensor: Binarized tensor if threshold is specified, else the original tensor.
    """
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(
    pr: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-7,
    threshold: Optional[float] = None,
    ignore_channels: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU or Jaccard) between ground truth and prediction.

    Args:
        pr (torch.Tensor): Predicted tensor (N, C, H, W) or (N, H, W).
        gt (torch.Tensor): Ground truth tensor (N, C, H, W) or (N, H, W).
        eps (float): Epsilon to avoid zero division, default=1e-7.
        threshold (float, optional): Threshold for binarization of predictions.
        ignore_channels (List[int], optional): Channels to ignore in the calculation.

    Returns:
        torch.Tensor: IoU score (scalar).
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps

    return (intersection + eps) / union


jaccard = iou


def f_score(
    pr: torch.Tensor,
    gt: torch.Tensor,
    beta: float = 1,
    eps: float = 1e-7,
    threshold: Optional[float] = None,
    ignore_channels: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Calculate the F-score (F-beta score) between ground truth and prediction.

    Args:
        pr (torch.Tensor): Predicted tensor (N, C, H, W) or (N, H, W).
        gt (torch.Tensor): Ground truth tensor (N, C, H, W) or (N, H, W).
        beta (float): Positive constant controlling the relative importance of precision vs. recall.
        eps (float): Epsilon to avoid zero division, default=1e-7.
        threshold (float, optional): Threshold for binarization of predictions.
        ignore_channels (List[int], optional): Channels to ignore.

    Returns:
        torch.Tensor: F-score (scalar).
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) / (
        (1 + beta ** 2) * tp + (beta ** 2) * fn + fp + eps
    )
    return score


def f_score_w(
    pr: torch.Tensor,
    gt: torch.Tensor,
    beta: float = 1,
    eps: float = 1e-7,
    threshold: Optional[float] = None,
    ignore_channels: Optional[List[int]] = None,
    class_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculate a weighted F-score (useful for weighted Dice or multi-class weighting).

    Args:
        pr (torch.Tensor): Predicted tensor (N, C, H, W).
        gt (torch.Tensor): Ground truth tensor (N, C, H, W).
        beta (float): Positive constant controlling the relative importance of precision vs. recall.
        eps (float): Epsilon for numerical stability, default=1e-7.
        threshold (float, optional): Threshold for binarization of predictions.
        ignore_channels (List[int], optional): Channels to ignore.
        class_weights (torch.Tensor, optional): Weights for each class, shape (C,).

    Returns:
        torch.Tensor: Weighted F-score, averaged across classes.
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    if class_weights is None:
        class_weights = torch.ones(gt.shape[1], device=gt.device)

    # Summation across spatial dimensions
    tp = torch.sum(gt * pr, dim=(0, 2, 3))
    fp = torch.sum(pr, dim=(0, 2, 3)) - tp
    fn = torch.sum(gt, dim=(0, 2, 3)) - tp

    weighted_tp = class_weights * tp
    weighted_fp = class_weights * fp
    weighted_fn = class_weights * fn

    score = ((1 + beta ** 2) * weighted_tp + eps) / (
        (1 + beta ** 2) * weighted_tp + (beta ** 2) * weighted_fn + weighted_fp + eps
    )

    return score.mean()


def accuracy(
    pr: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.5,
    ignore_channels: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Calculate the accuracy score between ground truth and predictions.

    Args:
        pr (torch.Tensor): Predicted tensor (N, C, H, W) or (N, H, W).
        gt (torch.Tensor): Ground truth tensor (N, C, H, W) or (N, H, W).
        threshold (float): Binarization threshold, default=0.5.
        ignore_channels (List[int], optional): Channels to ignore.

    Returns:
        torch.Tensor: Accuracy (scalar).
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


def precision(
    pr: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-7,
    threshold: Optional[float] = None,
    ignore_channels: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Calculate the precision score between ground truth and predictions.

    Args:
        pr (torch.Tensor): Predicted tensor (N, C, H, W).
        gt (torch.Tensor): Ground truth tensor (N, C, H, W).
        eps (float): Epsilon to avoid zero division, default=1e-7.
        threshold (float, optional): Threshold for binarization.
        ignore_channels (List[int], optional): Channels to ignore.

    Returns:
        torch.Tensor: Precision score (scalar).
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)
    return score


def recall(
    pr: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-7,
    threshold: Optional[float] = None,
    ignore_channels: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Calculate the recall score between ground truth and predictions.

    Args:
        pr (torch.Tensor): Predicted tensor (N, C, H, W).
        gt (torch.Tensor): Ground truth tensor (N, C, H, W).
        eps (float): Epsilon to avoid zero division, default=1e-7.
        threshold (float, optional): Threshold for binarization.
        ignore_channels (List[int], optional): Channels to ignore.

    Returns:
        torch.Tensor: Recall score (scalar).
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)
    return score
