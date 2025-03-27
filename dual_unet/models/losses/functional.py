import torch
from typing import Optional, List, Tuple

def _take_channels(
    *xs: torch.Tensor, 
    ignore_channels: Optional[List[int]] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Select (i.e., keep) only the channels not listed in `ignore_channels` from each tensor in `xs`.

    Args:
        xs (torch.Tensor): One or more tensors from which to select channels.
        ignore_channels (List[int], optional): Indices of channels to ignore. 
            If None, return the original tensors.

    Returns:
        Tuple[torch.Tensor, ...]: A tuple of tensors with ignored channels removed.
    """
    if ignore_channels is None:
        return xs
    else:
        device = xs[0].device
        channels_to_keep = [
            ch for ch in range(xs[0].shape[1]) if ch not in ignore_channels
        ]
        idx_tensor = torch.tensor(channels_to_keep, device=device)
        return tuple(torch.index_select(x, dim=1, index=idx_tensor) for x in xs)


def _threshold(
    x: torch.Tensor, 
    threshold: Optional[float] = None
) -> torch.Tensor:
    """
    Binarize a tensor using a specified threshold. If threshold is None, no binarization is applied.

    Args:
        x (torch.Tensor): Tensor to binarize.
        threshold (float, optional): Threshold value. If None, return `x` unchanged.

    Returns:
        torch.Tensor: Binarized tensor if threshold is given, otherwise the original tensor.
    """
    if threshold is not None:
        return (x > threshold).type_as(x)
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
    Calculate the Intersection over Union (IoU, also known as the Jaccard index) 
    between ground truth and prediction tensors.

    Args:
        pr (torch.Tensor): Predicted tensor of shape (N, C, H, W) or (C, H, W).
        gt (torch.Tensor): Ground truth tensor of the same shape.
        eps (float): Small epsilon to avoid zero division.
        threshold (float, optional): Threshold for binarizing predictions. 
            If None, no binarization is performed.
        ignore_channels (List[int], optional): Channels to ignore.

    Returns:
        torch.Tensor: Scalar IoU score.
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps

    return (intersection + eps) / union


jaccard = iou  # Alias for `iou`


def f_score(
    pr: torch.Tensor, 
    gt: torch.Tensor, 
    beta: float = 1.0, 
    eps: float = 1e-7, 
    threshold: Optional[float] = None,
    ignore_channels: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Calculate the F-score (Dice or F-beta) between ground truth and prediction tensors.

    Args:
        pr (torch.Tensor): Predicted tensor of shape (N, C, H, W) or (C, H, W).
        gt (torch.Tensor): Ground truth tensor of the same shape.
        beta (float): Positive constant for weighting precision vs. recall.
        eps (float): Small epsilon to avoid zero division.
        threshold (float, optional): Threshold for binarizing predictions. 
            If None, no binarization is performed.
        ignore_channels (List[int], optional): Channels to ignore.

    Returns:
        torch.Tensor: Scalar F score.
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    numerator = (1 + beta**2) * tp + eps
    denominator = (1 + beta**2) * tp + (beta**2) * fn + fp + eps

    return numerator / denominator


def f_score_w(
    pr: torch.Tensor,
    gt: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-7,
    threshold: Optional[float] = None,
    ignore_channels: Optional[List[int]] = None,
    class_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculate a weighted F-score (e.g., Weighted Dice) between ground truth and prediction tensors.
    Each class can have its own weighting factor.

    Args:
        pr (torch.Tensor): Predicted tensor of shape (N, C, H, W).
        gt (torch.Tensor): Ground truth tensor of the same shape.
        beta (float): Positive constant for weighting precision vs. recall in the F-beta score.
        eps (float): Small epsilon to avoid zero division.
        threshold (float, optional): Threshold for binarizing predictions. 
            If None, no binarization is performed.
        ignore_channels (List[int], optional): Channels to ignore.
        class_weights (torch.Tensor, optional): 1D tensor of shape (C,) containing
            the weight for each class. If None, all classes are weighted equally.

    Returns:
        torch.Tensor: Scalar weighted F score (mean across classes).
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # Default all-ones if class weights not provided
    if class_weights is None:
        class_weights = torch.ones(gt.shape[1], device=gt.device)

    # Compute intersection/union for each channel
    tp = torch.sum(gt * pr, dim=(0, 2, 3))
    fp = torch.sum(pr, dim=(0, 2, 3)) - tp
    fn = torch.sum(gt, dim=(0, 2, 3)) - tp

    # Apply per-class weights
    weighted_tp = class_weights * tp
    weighted_fp = class_weights * fp
    weighted_fn = class_weights * fn

    # Weighted F-score calculation
    numerator = (1 + beta**2) * weighted_tp + eps
    denominator = (1 + beta**2) * weighted_tp + (beta**2) * weighted_fn + weighted_fp + eps

    score_per_class = numerator / denominator
    return score_per_class.mean()  # Mean across classes


def accuracy(
    pr: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.5,
    ignore_channels: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Calculate accuracy between ground truth and prediction tensors.

    Args:
        pr (torch.Tensor): Predicted tensor of shape (N, C, H, W) or (C, H, W).
        gt (torch.Tensor): Ground truth tensor of the same shape.
        threshold (float): Threshold for binarizing predictions.
        ignore_channels (List[int], optional): Channels to ignore.

    Returns:
        torch.Tensor: Scalar accuracy score.
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # tp in this context is "true predictions," i.e., same values in pr and gt
    tp = (gt == pr).sum(dtype=pr.dtype)
    total_elements = gt.numel()

    return tp / total_elements


def precision(
    pr: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-7,
    threshold: Optional[float] = None,
    ignore_channels: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Calculate precision (positive predictive value) between ground truth and predictions.

    Args:
        pr (torch.Tensor): Predicted tensor of shape (N, C, H, W).
        gt (torch.Tensor): Ground truth tensor of the same shape.
        eps (float): Small epsilon to avoid zero division.
        threshold (float, optional): Threshold for binarizing predictions. If None, no binarization.
        ignore_channels (List[int], optional): Channels to ignore.

    Returns:
        torch.Tensor: Scalar precision score.
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    return (tp + eps) / (tp + fp + eps)


def recall(
    pr: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-7,
    threshold: Optional[float] = None,
    ignore_channels: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Calculate recall (sensitivity) between ground truth and predictions.

    Args:
        pr (torch.Tensor): Predicted tensor of shape (N, C, H, W).
        gt (torch.Tensor): Ground truth tensor of the same shape.
        eps (float): Small epsilon to avoid zero division.
        threshold (float, optional): Threshold for binarizing predictions. If None, no binarization.
        ignore_channels (List[int], optional): Channels to ignore.

    Returns:
        torch.Tensor: Scalar recall score.
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    return (tp + eps) / (tp + fn + eps)
