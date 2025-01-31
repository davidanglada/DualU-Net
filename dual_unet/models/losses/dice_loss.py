import torch
import torch.nn as nn
from .base import Loss
from .functional import f_score, f_score_w

class DiceLoss_w(Loss):
    def __init__(self, eps=1., beta=1., ignore_channels=None, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.ignore_channels = ignore_channels
        self.class_weights = class_weights

    def forward(self, y_pr, y_gt):
        return 1 - f_score_w(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            class_weights=self.class_weights
        )

class DiceLoss(Loss):

    def __init__(self, eps=1., beta=1., ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        # self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        # y_pr = self.activation(y_pr)
        return 1 - f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-3):
#         """
#         Dice loss for multi-class segmentation with `NxCxHxW` shape.
#         Args:
#             smooth (float): Smoothing term to avoid division by zero.
#         """
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, pred, true):
#         """
#         Forward pass to compute the Dice loss.

#         Args:
#             pred (torch.Tensor): Predictions of shape NxCxHxW.
#             true (torch.Tensor): Ground truth of shape NxCxHxW.

#         Returns:
#             torch.Tensor: Dice loss value (summed across all classes).
#         """
#         # Ensure predictions and true labels have the same shape
#         assert pred.shape == true.shape, \
#             f"Shape mismatch: pred {pred.shape} vs true {true.shape}"

#         # Compute element-wise intersection and sums over spatial dimensions (H, W)
#         inse = torch.sum(pred * true, dim=(2, 3))  # Intersection for each class, sum over H and W
#         l = torch.sum(pred, dim=(2, 3))            # Sum of prediction over H and W for each class
#         r = torch.sum(true, dim=(2, 3))            # Sum of ground truth over H and W for each class

#         # Compute Dice coefficient per class and per batch
#         dice_per_class = (2.0 * inse + self.smooth) / (l + r + self.smooth)

#         # Average over batch dimension (N) and sum over class dimension (C)
#         dice_loss_per_batch = 1.0 - dice_per_class.mean(dim=0)  # Mean over batch (N), class-wise loss
#         total_dice_loss = torch.sum(dice_loss_per_batch)  # Sum over all classes

#         return total_dice_loss



# import torch
# import torch.nn as nn

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-3):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, pred, target):
#         pred = pred.contiguous().view(-1)
#         target = target.contiguous().view(-1)
#         intersection = (pred * target).sum()
#         dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
#         return 1 - dice


