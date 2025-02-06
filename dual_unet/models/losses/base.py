import re
import torch.nn as nn
from typing import Union, Tuple

class BaseObject(nn.Module):
    """
    Base class providing a flexible naming convention for PyTorch modules. 
    If `_name` is unset, the class name is transformed to snake_case automatically.
    """

    def __init__(self, name: str = None):
        """
        Args:
            name (str, optional): A custom name for the object. 
                                  If None, a snake_case version of the class name is used.
        """
        super().__init__()
        self._name = name

    @property
    def __name__(self) -> str:
        """
        Returns:
            str: The object name. If `_name` is defined, returns that. 
                 Otherwise, converts the class name to snake_case.
        """
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    """
    Placeholder for metric-related functionality. 
    Inherits naming behavior from BaseObject.
    """
    pass


class Loss(BaseObject):
    """
    Base class for loss functions. Supports:
      - Summation of two Loss objects via `+`
      - Multiplying a Loss by a scalar via `*`
    """

    def __add__(self, other: "Loss") -> "Loss":
        """
        Define the addition (sum) operation between two Losses.

        Args:
            other (Loss): Another Loss object to sum.

        Returns:
            SumOfLosses: A combined Loss that sums the two individual losses.
        """
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other: "Loss") -> "Loss":
        """Enables reverse-add so that `loss1 + loss2` and `loss2 + loss1` behave consistently."""
        return self.__add__(other)

    def __mul__(self, value: Union[int, float]) -> "Loss":
        """
        Scale this Loss by a numeric factor.

        Args:
            value (int or float): The multiplier.

        Returns:
            MultipliedLoss: A new Loss object scaled by `value`.
        """
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be multiplied by an int or float")

    def __rmul__(self, other: Union[int, float]) -> "Loss":
        """Enables reverse-mul so that `2 * loss` and `loss * 2` behave consistently."""
        return self.__mul__(other)


class SumOfLosses(Loss):
    """
    A composite loss representing the sum of two child losses.
    """

    def __init__(self, l1: Loss, l2: Loss):
        """
        Args:
            l1 (Loss): The first loss in the sum.
            l2 (Loss): The second loss in the sum.
        """
        name = f"{l1.__name__} + {l2.__name__}"
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs) -> float:
        """
        Invoke the combined loss.

        Args:
            inputs: Arbitrary positional arguments forwarded to both child losses.

        Returns:
            float: The sum of `l1(inputs) + l2(inputs)`.
        """
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


class MultipliedLoss(Loss):
    """
    A composite loss representing a single child loss scaled by a numeric factor.
    """

    def __init__(self, loss: Loss, multiplier: float):
        """
        Args:
            loss (Loss): The child loss to be scaled.
            multiplier (float): Scale factor.
        """
        # Resolve a readable name
        if "+" in loss.__name__:
            name = f"{multiplier} * ({loss.__name__})"
        else:
            name = f"{multiplier} * {loss.__name__}"

        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs) -> float:
        """
        Invoke the scaled loss.

        Args:
            inputs: Arbitrary positional arguments forwarded to the child loss.

        Returns:
            float: The scaled loss value.
        """
        return self.multiplier * self.loss.forward(*inputs)
