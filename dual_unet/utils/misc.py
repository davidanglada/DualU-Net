import os
import subprocess
import time
import datetime
import pickle
from collections import defaultdict, deque
from typing import List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
import numpy as np
import random
import torchvision  # needed due to empty tensor bug in torchvision 0.5

from .distributed import is_dist_avail_and_initialized, get_world_size


class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a
    rolling window as well as the global average.

    Attributes:
        deque (collections.deque): Stores the most recent values (up to window_size).
        total (float): Sum of all values (for computing global average).
        count (int): Count of all values added.
        fmt (str): Format string for printing.
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        """
        Args:
            window_size (int): The size of the rolling window for smoothing.
            fmt (str, optional): Format string used in __str__.
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1):
        """
        Add a new value to the rolling window and update global stats.

        Args:
            value (float): The new measurement to add.
            n (int): The weight or number of times 'value' is repeated (default=1).
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronize the total and count across processes in distributed training.
        Warning: this doesn't synchronize the rolling window (self.deque) – only global stats.
        """
        if not is_dist_avail_and_initialized():
            return

        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        self.count, self.total = int(t[0].item()), t[1].item()

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque) if len(self.deque) > 0 else 0.0

    @property
    def value(self) -> float:
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """
    Logs (smooths) a set of metrics. Has a log_every method to iterate over data 
    and print stats periodically.

    Attributes:
        meters (defaultdict): Mapping from metric name -> SmoothedValue.
        delimiter (str): Delimiter used for printing metrics.
    """

    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Update the stored metrics. Example: logger.update(loss=0.5, lr=1e-4)

        Each kwarg is typically a float or int (or a scalar tensor).
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), "Metric values must be float or int."
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self) -> str:
        """
        Return a string with all current metrics, e.g. "loss: 0.1234  lr: 1e-4 ..."
        """
        metrics_str = []
        for name, meter in self.meters.items():
            metrics_str.append(f"{name}: {meter}")
        return self.delimiter.join(metrics_str)

    def synchronize_between_processes(self):
        """
        Synchronize the global stats (total and count) of all SmoothedValues across processes.
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue):
        """
        Manually add a SmoothedValue meter to track a specific metric.

        Args:
            name (str): Metric name.
            meter (SmoothedValue): An instance for tracking this metric.
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq: int, header: Optional[str] = None):
        """
        Iterate over `iterable`, measuring the iteration/data time, and 
        printing stats every `print_freq` steps.

        Args:
            iterable: The data or list to iterate over.
            print_freq (int): Print stats every `print_freq` iterations.
            header (str, optional): A string header for the log messages.

        Yields:
            The items from iterable one by one.
        """
        if header is None:
            header = ""

        i = 0
        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        space_fmt = ":" + str(len(str(len(iterable)))) + "d"

        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                f"[{{0{space_fmt}}}/{{1}}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}"
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                f"[{{0{space_fmt}}}/{{1}}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}"
            ])

        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                    ))
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} "
              f"({total_time / len(iterable):.4f} s / it)")


def get_sha() -> str:
    """
    Retrieve git commit SHA, diff status, and branch name.

    Returns:
        str: A string describing the current git SHA, diff status, and branch name.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff_out = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff_out else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    return f"sha: {sha}, status: {diff}, branch: {branch}"


def collate_fn(batch):
    """
    A placeholder or specialized collate function for data loading.

    If the data are standard, you can just use torch.utils.data.default_collate.
    Modify if your data structure is different.
    """
    return torch.utils.data.default_collate(batch)


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    """
    Takes a list of shapes, e.g. [[C,H,W], [C,H,W], ...],
    and returns the element-wise maximum shape [C, H, W].
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for i, val in enumerate(sublist):
            maxes[i] = max(maxes[i], val)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    Converts a list of 3D tensors (C,H,W) into a single padded batch
    with shape (B, C, H_max, W_max), plus a mask of shape (B, H_max, W_max)
    indicating which elements are padding.

    If 4D input is needed, or different shape logic, adapt accordingly.
    """
    if tensor_list[0].ndim == 3:
        # compute max size
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size  # e.g. [B, C, H, W]
        b, c, h, w = batch_shape

        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        for img_idx, (img, pad_img, m) in enumerate(zip(tensor_list, tensor, mask)):
            c_img, h_img, w_img = img.shape
            pad_img[:c_img, :h_img, :w_img].copy_(img)
            m[:h_img, :w_img] = False

    else:
        raise ValueError("nested_tensor_from_tensor_list: only 3D tensors supported currently")

    return NestedTensor(tensor, mask)


class NestedTensor:
    """
    A structure to hold a batch of images (tensors) of possibly varying size,
    plus a mask indicating the valid region in each image.

    Attributes:
        tensors (torch.Tensor): The padded image tensor of shape (B, C, H_max, W_max).
        mask (torch.Tensor): A boolean mask of shape (B, H_max, W_max), 
                             True for padding, False for actual pixels.
    """

    def __init__(self, tensors: torch.Tensor, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        """
        Moves the NestedTensor to the specified device.
        """
        cast_tensors = self.tensors.to(device, non_blocking=non_blocking)
        cast_mask = None
        if self.mask is not None:
            cast_mask = self.mask.to(device, non_blocking=non_blocking)
        return NestedTensor(cast_tensors, cast_mask)

    def record_stream(self, stream):
        """
        Records this NestedTensor in a given stream (for pinned memory, etc.).
        """
        self.tensors.record_stream(stream)
        if self.mask is not None:
            self.mask.record_stream(stream)

    def decompose(self):
        """
        Returns the underlying (tensors, mask).
        """
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    """
    Compute the precision@k for the specified values of k.

    Args:
        output (torch.Tensor): Predictions of shape (N, C), where C is #classes.
        target (torch.Tensor): Ground truth indices of shape (N,).
        topk (tuple): The list of ranks for which to compute the precision.

    Returns:
        list[torch.Tensor]: A list of length len(topk), each a scalar with the precision@k.
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]

    maxk = max(topk)
    batch_size = target.size(0)

    # topk on dimension=1 (C)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # shape: (maxk, N)

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct[:k] -> shape (k, N)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(
    input: torch.Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None
) -> torch.Tensor:
    """
    A wrapper around torchvision.ops.misc.interpolate with the same interface as nn.functional.interpolate,
    but potentially with support for empty batch sizes or other custom logic as needed.
    """
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def get_total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """
    Compute the total gradient norm for a list of parameters.

    Args:
        parameters (iterable): The parameters for which to compute the gradient norm.
        norm_type (float): The norm type to compute, e.g. 2.0 for L2 norm.

    Returns:
        float: The total norm of the gradients.
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if len(parameters) == 0:
        return 0.0

    device = parameters[0].grad.device
    norm_type = float(norm_type)

    total = torch.norm(torch.stack([
        torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters
    ]), norm_type)
    return total.item()


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute the logit (inverse of sigmoid) of x, with numeric clamping.

    Args:
        x (torch.Tensor): Input in [0,1].
        eps (float): Epsilon for numeric stability, e.g. clamp x in [eps, 1-eps].

    Returns:
        torch.Tensor: The logit of x.
    """
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def seed_everything(seed: int):
    """
    Seed all random number generators for reproducibility.

    Args:
        seed (int): The seed to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch >= 1.9
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass


def seed_worker(worker_id: int):
    """
    A worker_init_fn for PyTorch DataLoader to ensure each worker has a unique seed.

    Args:
        worker_id (int): The worker ID in DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
