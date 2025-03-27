import re
import torch
import torch.nn as nn
from typing import Union


class BaseObject(nn.Module):
    """
    A base class that inherits from `nn.Module` and provides a uniform way
    to manage the name of the object. If no name is specified, it derives
    one from the class name using a conversion from CamelCase to snake_case.
    """

    def __init__(self, name: str = None):
        """
        Args:
            name (str, optional): Name for the object. If None, the name is derived
                from the class name (e.g., "BaseObject" -> "base_object").
        """
        super().__init__()
        self._name = name

    @property
    def __name__(self) -> str:
        """
        Returns:
            str: The designated name of the object. If not provided in the constructor,
            a snake_case version of the class name is returned.
        """
        if self._name is None:
            class_name = self.__class__.__name__
            # Insert underscores between CamelCase words
            s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", class_name)
            return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    """
    A base class for metrics, extending `BaseObject`.
    Currently, this class does not add functionality beyond its parent.
    """
    pass


class Loss(BaseObject):
    """
    A base class for defining loss functions, extending `BaseObject`.
    This class allows for composable arithmetic operations on losses:
      - Summation of two Loss objects -> SumOfLosses
      - Multiplying a Loss by a scalar -> MultipliedLoss
    """

    def __add__(self, other: "Loss") -> "Loss":
        """
        Overload the + operator for summation of Loss objects.

        Args:
            other (Loss): Another Loss to be summed with this one.

        Returns:
            SumOfLosses: A composite Loss object representing the sum.
        """
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other: "Loss") -> "Loss":
        """
        Right-side add operator, to handle cases like 0 + loss.

        Args:
            other (Loss or numeric): The left operand.

        Returns:
            Loss: The resulting sum.
        """
        return self.__add__(other)

    def __mul__(self, value: Union[int, float]) -> "Loss":
        """
        Overload the * operator for scaling a Loss by a numeric value.

        Args:
            value (int or float): The scalar multiplier.

        Returns:
            MultipliedLoss: A composite Loss object representing the scaled loss.
        """
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss multiplier must be an integer or float.")

    def __rmul__(self, other: Union[int, float]) -> "Loss":
        """
        Right-side multiply operator, for expressions like 2 * loss.

        Args:
            other (int or float): The scalar multiplier.

        Returns:
            Loss: The resulting scaled loss.
        """
        return self.__mul__(other)

    def forward(self, *inputs, **kwargs):
        """
        Compute the loss value. Must be overridden by subclasses.

        Args:
            *inputs: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The computed loss value.
        """
        raise NotImplementedError("Subclasses must implement the forward() method.")


class SumOfLosses(Loss):
    """
    A composite Loss that represents the sum of two individual losses.
    """

    def __init__(self, l1: Loss, l2: Loss):
        """
        Args:
            l1 (Loss): The first loss function.
            l2 (Loss): The second loss function.
        """
        name = f"{l1.__name__} + {l2.__name__}"
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        """
        Forward pass: sum the two sub-losses over the same inputs.

        Args:
            *inputs: Arguments for the loss functions.
            **kwargs: Additional keyword arguments (ignored here).

        Returns:
            torch.Tensor: The sum of the two loss values.
        """
        return self.l1.forward(*inputs, **kwargs) + self.l2.forward(*inputs, **kwargs)


class MultipliedLoss(Loss):
    """
    A composite Loss that represents the original loss multiplied by a scalar coefficient.
    """

    def __init__(self, loss: Loss, multiplier: float):
        """
        Args:
            loss (Loss): The base loss function to be scaled.
            multiplier (float): The scalar multiplier.
        """
        # Generate a name for display/logging
        if "+" in loss.__name__:
            name = f"{multiplier} * ({loss.__name__})"
        else:
            name = f"{multiplier} * {loss.__name__}"
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        """
        Forward pass: multiply the base loss output by self.multiplier.

        Args:
            *inputs: Arguments for the base loss function.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The scaled loss value.
        """
        return self.multiplier * self.loss.forward(*inputs, **kwargs)


# Example for a WeightedLoss, if needed:
#
# class WeightedLoss(Loss):
#     """
#     A composite loss applying separate weights to each sub-loss.
#     """
# 
#     def __init__(self, l1: Loss, l2: Loss, w1: float, w2: float):
#         name = f"{w1}*{l1.__name__} + {w2}*{l2.__name__}"
#         super().__init__(name=name)
#         self.l1 = l1
#         self.l2 = l2
#         self.w1 = w1
#         self.w2 = w2
# 
#     def forward(self, inputs1, inputs2):
#         """
#         You could define separate forward args if needed, or pass in *inputs and parse.
#         """
#         return self.w1 * self.l1.forward(inputs1) + self.w2 * self.l2.forward(inputs2)
