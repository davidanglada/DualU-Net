import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import DiceLoss, DiceLoss_w
from .mse_loss import MSELoss


class DualLoss_combined(nn.Module):
    """
    A dual-purpose loss function that combines a segmentation loss (Cross-Entropy + Dice)
    with a centroid regression loss (MSE) for multi-task learning.

    Specifically:
      - Segmentation predictions are trained via a weighted combination of:
          * CrossEntropyLoss
          * DiceLoss (with optional class weights)
      - Gaussian centroid predictions are trained via MSELoss.

    Args:
        ce_weights (torch.Tensor): Class weights for the CrossEntropyLoss.
        weight_dice (float): Weight factor for the Dice loss term.
        weight_dice_b (float): [Currently unused in the code, but kept for possible expansions].
        weight_ce (float): Weight factor for the CrossEntropyLoss term.
        weight_mse (float): Weight factor for the MSELoss term.
        smooth (float): Smoothing term for the Dice loss (unused in this code snippet,
            but can be passed along if needed).
    """

    def __init__(
        self,
        ce_weights: torch.Tensor,
        weight_dice: float = 1.0,
        weight_dice_b: float = 1.0,  # Not currently used
        weight_ce: float = 1.0,
        weight_mse: float = 1.0,
        smooth: float = 1.0
    ):
        """
        Initialize the dual loss components: DiceLoss_w for segmentation, CrossEntropyLoss,
        and MSELoss for centroid regression.

        Args:
            ce_weights (torch.Tensor): Class weights tensor for the CrossEntropyLoss.
            weight_dice (float): Weight factor for the Dice loss term. Default=1.0
            weight_dice_b (float): Extra Dice weight (currently unused).
            weight_ce (float): Weight factor for the CrossEntropy loss term. Default=1.0
            weight_mse (float): Weight factor for the MSE loss term. Default=1.0
            smooth (float): Smoothing factor for Dice loss (unused in current code).
        """
        super().__init__()
        self.ce_weights = ce_weights
        self.weight_dice = weight_dice
        self.weight_dice_b = weight_dice_b
        self.weight_mse = weight_mse
        self.weight_ce = weight_ce

        # Using a weighted DiceLoss variant
        self.dice_loss = DiceLoss_w(class_weights=ce_weights)

        # CrossEntropyLoss that uses ce_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=self.ce_weights)

        # MSE for centroid regression
        self.mse_loss = MSELoss()

    def forward(self, pred, target):
        """
        Compute the combined loss from segmentation predictions and centroid predictions.

        The input `pred` is expected to be a tuple/list of:
            pred_segmentation: (N, C, H, W)
            pred_centroids: (N, 1, H, W)

        The input `target` is expected to be a list of dictionaries, each containing:
            target_segmentation: "segmentation_mask" -> (C, H, W)
            target_centroids: "centroid_gaussian" -> (1, H, W)

        Args:
            pred (Tuple[torch.Tensor, torch.Tensor]):
                (pred_segmentation, pred_centroids)
            target (List[Dict[str, torch.Tensor]]):
                Each dict with 'segmentation_mask' and 'centroid_gaussian'.

        Returns:
            torch.Tensor: A scalar tensor for the combined loss.
        """
        # Unpack predictions
        pred_segmentation = pred[0]  # shape: (N, C, H, W)
        pred_centroids = pred[1]     # shape: (N, 1, H, W)

        # Stack the segmentation masks and centroids from target
        target_segmentation = torch.stack([t["segmentation_mask"] for t in target])  # (N, C, H, W)
        target_centroids = torch.stack([t["centroid_gaussian"] for t in target])     # (N, 1, H, W)

        # Cross-entropy loss on raw logits for segmentation
        loss_ce = self.ce_loss(pred_segmentation, target_segmentation)

        # Dice loss on softmax probabilities for segmentation
        loss_dice = self.dice_loss(
            torch.softmax(pred_segmentation, dim=1),
            target_segmentation
        )

        # MSE loss for centroid predictions
        loss_mse = self.mse_loss(pred_centroids, target_centroids)

        # Weighted sum of the three losses
        total_loss = (self.weight_ce * loss_ce
                      + self.weight_dice * loss_dice
                      + self.weight_mse * loss_mse)

        return total_loss
