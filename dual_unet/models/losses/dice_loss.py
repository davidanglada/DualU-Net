import torch
import torch.nn as nn
from .base import Loss
from .functional import f_score, f_score_w


class DiceLoss_w(Loss):
    """
    Weighted Dice loss for segmentation tasks.

    This class uses the f_score_w function (from .functional) that accepts
    class-specific weights.

    Args:
        eps (float): A small epsilon for numerical stability.
        beta (float): Weighting factor for the F-beta score.
        ignore_channels (Optional[List[int]]): Indices of channels to ignore during score calculation.
        class_weights (Optional[List[float]]): Per-class weights for the Dice loss.
        **kwargs: Extra keyword arguments passed to the base class (e.g., `name`).
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
        Initialize the weighted Dice loss.

        Args:
            eps (float): A smoothing constant to avoid division by zero.
            beta (float): Determines the F-beta score's weighting for precision vs. recall.
            ignore_channels (list, optional): Channels to ignore (e.g., background).
            class_weights (list, optional): A list of weights, one per class.
            **kwargs: Additional arguments for the `Loss` base class (e.g. `name`).
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
            y_gt (torch.Tensor): Ground truth tensor of the same shape.

        Returns:
            torch.Tensor: A scalar value representing 1 - weighted dice coefficient.
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
    Standard Dice loss for segmentation tasks, using the f_score function.

    Args:
        eps (float): Smoothing term.
        beta (float): Weighting factor for F-beta score.
        ignore_channels (List[int], optional): Channels to ignore.
        **kwargs: Extra keyword arguments for the base class.
    """

    def __init__(
        self,
        eps: float = 1.0,
        beta: float = 1.0,
        ignore_channels=None,
        **kwargs
    ):
        """
        Initialize the Dice loss.

        Args:
            eps (float): A small epsilon for numerical stability.
            beta (float): F-beta weighting in the F-score calculation.
            ignore_channels (List[int], optional): Channels to ignore in the calculation.
            **kwargs: Additional keyword arguments (e.g., `name`) for the `Loss` base class.
        """
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.ignore_channels = ignore_channels

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss.

        Args:
            y_pr (torch.Tensor): Predictions of shape (N, C, H, W).
            y_gt (torch.Tensor): Ground truth of the same shape.

        Returns:
            torch.Tensor: A scalar value representing 1 - dice coefficient.
        """
        return 1.0 - f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
