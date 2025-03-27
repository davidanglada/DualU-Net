import numpy as np
from typing import Any, List, Optional, Tuple

def preprocess_input(
    x: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    input_space: str = "RGB",
    input_range: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> np.ndarray:
    """
    Preprocess an input image array for use in a segmentation/classification model.

    This function optionally:
      - Converts images from RGB to BGR (or vice versa) by reversing the last channel dimension.
      - Normalizes the image according to a specified input range.
      - Subtracts a mean and divides by a standard deviation for each channel.

    Args:
        x (np.ndarray): Image array of shape (..., C), where C is channels (e.g. 3 for RGB).
        mean (Optional[List[float]]): A list of per-channel mean values to subtract, e.g. [0.485, 0.456, 0.406].
            If None, no mean subtraction is applied.
        std (Optional[List[float]]): A list of per-channel standard deviation values, e.g. [0.229, 0.224, 0.225].
            If None, no division by standard deviation is applied.
        input_space (str): The color space of the input, e.g. "RGB" or "BGR". Defaults to "RGB".
        input_range (Optional[Tuple[float, float]]): The input range (min, max) to which the image array should
            be normalized (e.g. (0, 1) if your model expects values in that range).
            If provided and the max of `x` is greater than 1 while `input_range` max is 1, values are scaled by /255.
        **kwargs: Additional keyword arguments, reserved for future use or compatibility.

    Returns:
        np.ndarray: The processed image array with the same shape as the input but potentially modified values.
    """
    # Convert from RGB to BGR or vice versa by reversing the channels if needed
    if input_space == "BGR":
        x = x[..., ::-1].copy()

    # Scale if input_range suggests 0..1 but data is in 0..255
    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    # Subtract per-channel mean
    if mean is not None:
        mean_arr = np.array(mean, dtype=x.dtype)
        x = x - mean_arr

    # Divide by per-channel std
    if std is not None:
        std_arr = np.array(std, dtype=x.dtype)
        x = x / std_arr

    return x
