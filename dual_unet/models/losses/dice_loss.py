import torch
import torch.nn as nn
from .base import Loss
from .functional import f_score, f_score_w


class DiceLoss_w(Loss):
    """
    Weighted Dice loss. Uses class_weights to weigh each class differently when computing the Dice score.
    """

    def __init__(
        self,
        eps: float = 1.0,
        beta: float = 1.0,
        ignore_channels=None,
        class_weights=None,
        **kwargs
    ):
        """
        Args:
            eps (float): A small epsilon for numerical stability, default=1.0.
            beta (float): The beta parameter in the F-beta/Dice formula, default=1.0.
            ignore_channels (optional): Channels to exclude from the calculation.
            class_weights (optional): Weights to apply per class.
        """
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.ignore_channels = ignore_channels
        self.class_weights = class_weights

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted Dice loss.

        Args:
            y_pr (torch.Tensor): Predicted tensor of shape (N, C, H, W).
            y_gt (torch.Tensor): Ground-truth tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return 1.0 - f_score_w(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            class_weights=self.class_weights,
        )


class DiceLoss(Loss):
    """
    Standard Dice loss. Does not include per-class weighting.
    """

    def __init__(
        self,
        eps: float = 1.0,
        beta: float = 1.0,
        ignore_channels=None,
        **kwargs
    ):
        """
        Args:
            eps (float): A small epsilon for numerical stability, default=1.0.
            beta (float): The beta parameter in the F-beta/Dice formula, default=1.0.
            ignore_channels (optional): Channels to exclude from the calculation.
        """
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.ignore_channels = ignore_channels

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss.

        Args:
            y_pr (torch.Tensor): Predicted tensor of shape (N, C, H, W).
            y_gt (torch.Tensor): Ground-truth tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return 1.0 - f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
