import torch
import torch.nn as nn
from typing import Any


class MSELoss(nn.Module):
    """
    A simple wrapper around PyTorch's built-in MSELoss for clarity and potential customization.

    Example:
        mse_loss_fn = MSELoss()
        loss = mse_loss_fn(pred_centroids, gt_centroids)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the MSELoss wrapper.

        Args:
            **kwargs: Additional keyword arguments passed to `nn.MSELoss` if needed.
        """
        super().__init__()
        self.mse_loss = nn.MSELoss(**kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Squared Error (MSE) between predictions and targets.

        Args:
            pred (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Scalar MSE loss value.
        """
        return self.mse_loss(pred, target)
