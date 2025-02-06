# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

import numpy as np
from typing import Optional, List, Tuple


def preprocess_input(
    x: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    input_space: str = "RGB",
    input_range: Optional[Tuple[float, float]] = None,
    **kwargs
) -> np.ndarray:
    """
    Adjust the input array (image) according to specified mean, std, and color space.

    Steps:
      1) If input_space is "BGR", reverse channels (RGB -> BGR or vice versa).
      2) If input_range is (0,1) and the data is in [0..255], scale the array by /255.
      3) Subtract mean if provided.
      4) Divide by std if provided.

    Args:
        x (np.ndarray): Input array, typically an image with shape (..., 3).
        mean (List[float], optional): Mean values for each channel.
        std (List[float], optional): Std values for each channel.
        input_space (str): Either "RGB" or "BGR". Defaults to "RGB".
        input_range (Tuple[float, float], optional): Expected input range, e.g. (0,1) or (0,255).

    Returns:
        np.ndarray: Processed input array.
    """
    # 1) Convert color space if needed
    if input_space == "BGR":
        x = x[..., ::-1].copy()

    # 2) Rescale if input_range is in (0,1) but x is still in [0..255]
    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    # 3) Subtract mean
    if mean is not None:
        mean_arr = np.array(mean)
        x = x - mean_arr

    # 4) Divide by std
    if std is not None:
        std_arr = np.array(std)
        x = x / std_arr

    return x
