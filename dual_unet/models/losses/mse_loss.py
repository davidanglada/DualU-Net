import torch
import torch.nn as nn
from typing import Union

class MSELoss(nn.Module):
    """
    A wrapper around PyTorch's nn.MSELoss for centroid or continuous target regression.
    """

    def __init__(self) -> None:
        """
        Initialize the MSELoss module.
        """
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Mean Squared Error (MSE) between `pred` and `target`.

        Args:
            pred (torch.Tensor): Predicted tensor, shape (N, *).
            target (torch.Tensor): Ground-truth tensor, same shape as `pred`.

        Returns:
            torch.Tensor: A scalar tensor representing the MSE.
        """
        return self.mse_loss(pred, target)
