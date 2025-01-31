import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import DiceLoss, DiceLoss_w
from .mse_loss import MSELoss

class DualLoss_combined(nn.Module):
    def __init__(self, ce_weights, weight_dice=1.0, weight_dice_b=1.0, weight_ce=1.0, weight_mse=1.0, smooth=1.0):
        """
        Dual Loss combining Dice Loss for segmentation and MSE Loss for centroids.
        
        Args:
            weight_dice (float): Weight for the Dice loss.
            weight_mse (float): Weight for the MSE loss.
            smooth (float): Smoothing factor for Dice loss.
        """
        super(DualLoss_combined, self).__init__()
        self.ce_weights = ce_weights
        self.weight_dice = weight_dice
        self.weight_dice_b = weight_dice_b
        self.weight_mse = weight_mse
        self.weight_ce = weight_ce
        self.dice_loss = DiceLoss_w(class_weights=ce_weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.ce_weights)
        self.mse_loss = MSELoss()

    def forward(self, pred, target):
        """
        Forward pass for the Dual Loss.

        Args:
            pred_segmentation (torch.Tensor): Predicted segmentation mask (CxHxW).
            target_segmentation (torch.Tensor): Ground truth segmentation mask (CxHxW).
            pred_centroids (torch.Tensor): Predicted Gaussian centroid mask (1xHxW).
            target_centroids (torch.Tensor): Ground truth Gaussian centroid mask (1xHxW).

        Returns:
            torch.Tensor: The combined loss value.
        """
        pred_segmentation = pred[0]
        pred_centroids = pred[1]
        target_segmentation = torch.stack([t['segmentation_mask'] for t in target])
        target_centroids = torch.stack([t['centroid_gaussian'] for t in target])

        # print(pred_segmentation.shape, target_segmentation.shape)
        # print(pred_centroids.shape, target_centroids.shape)

        # Compute Dice loss for the segmentation masks
        loss_ce = self.ce_loss(pred_segmentation, target_segmentation, self.ce_weights, weight_ce=self.weight_ce, weight_dice_b=self.weight_dice_b)
        loss_dice = self.dice_loss(torch.softmax(pred_segmentation, dim=1), target_segmentation)

        # Compute MSE loss for the Gaussian centroid masks
        loss_mse = self.mse_loss(pred_centroids, target_centroids)

        # Combine the losses with the respective weights
        # total_loss = self.weight_dice*(loss_ce + loss_dice) + self.weight_mse*loss_mse
        total_loss = loss_ce + self.weight_dice*loss_dice + self.weight_mse*loss_mse

        return total_loss

