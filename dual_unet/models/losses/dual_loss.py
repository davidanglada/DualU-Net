import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union

from .dice_loss import DiceLoss_w
from .mse_loss import MSELoss


class DualLoss_combined(nn.Module):
    """
    A loss function combining:
      1. Cross-entropy + Dice loss for segmentation
      2. MSE loss for centroid regression
    """

    def __init__(
        self,
        ce_weights: torch.Tensor,
        weight_dice: float = 1.0,
        weight_dice_b: float = 1.0,
        weight_ce: float = 1.0,
        weight_mse: float = 1.0,
        smooth: float = 1.0
    ) -> None:
        """
        Args:
            ce_weights (torch.Tensor): Class weights for cross-entropy loss (shape: [num_classes]).
            weight_dice (float): Weight factor for Dice loss term.
            weight_dice_b (float): Additional weight/factor that could be used in dice formulations.
            weight_ce (float): Weight factor for cross-entropy term.
            weight_mse (float): Weight factor for MSE term.
            smooth (float): Smoothing factor, applicable in some Dice variations.
        """
        super(DualLoss_combined, self).__init__()
        self.ce_weights = ce_weights
        self.weight_dice = weight_dice
        self.weight_dice_b = weight_dice_b
        self.weight_mse = weight_mse
        self.weight_ce = weight_ce
        self.smooth = smooth

        # Weighted Dice loss that can handle per-class weights
        self.dice_loss = DiceLoss_w(class_weights=ce_weights)

        # Cross-entropy loss with optional class weights
        self.ce_loss = nn.CrossEntropyLoss(weight=self.ce_weights)

        # MSE loss for centroid maps
        self.mse_loss = MSELoss()

    def forward(
        self,
        pred: List[torch.Tensor],
        target: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute the combined segmentation + centroid loss.

        Args:
            pred (List[torch.Tensor]): A list with two elements:
                pred[0] = segmentation logits (N, C, H, W)
                pred[1] = centroid map (N, 1, H, W)
            target (List[Dict[str, torch.Tensor]]): A list of length N, where each element is a dictionary containing:
                "segmentation_mask" (tensor) -> (C, H, W) or (H, W) ground truth
                "centroid_gaussian" (tensor) -> (1, H, W) ground truth

        Returns:
            torch.Tensor: A scalar tensor representing the combined loss.
        """
        # Unpack predictions
        pred_segmentation = pred[0]  # shape (N, C, H, W)
        pred_centroids = pred[1]     # shape (N, 1, H, W)

        # Stack target segmentation & centroid maps along the batch dimension
        target_segmentation = torch.stack([t["segmentation_mask"] for t in target])
        target_centroids = torch.stack([t["centroid_gaussian"] for t in target])

        # Compute cross-entropy loss
        loss_ce = self.ce_loss(pred_segmentation, target_segmentation)

        # Compute weighted Dice loss (uses softmax for multi-class)
        loss_dice = self.dice_loss(torch.softmax(pred_segmentation, dim=1), target_segmentation)

        # Compute MSE loss for centroid regression
        loss_mse = self.mse_loss(pred_centroids, target_centroids)

        # Weighted sum of all components
        total_loss = loss_ce + (self.weight_dice * loss_dice) + (self.weight_mse * loss_mse)

        return total_loss
