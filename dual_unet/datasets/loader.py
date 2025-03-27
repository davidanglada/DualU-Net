import math
import torch
from typing import Optional

"""
Fix for DistributedSampler that doesn't add duplicates.
In validation, the last batch may be smaller than the others, and the default DistributedSampler
will add duplicates to make the batch size consistent across replicas. This class prevents that behavior.
"""

class DistributedSamplerNoDuplicate(torch.utils.data.DistributedSampler):
    """
    A distributed sampler that avoids adding duplicate samples to match batch sizes across replicas.
    Useful for validation or testing, where it’s acceptable to have smaller final batches.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ) -> None:
        """
        Initialize the distributed sampler with no duplication for smaller final batches.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from.
            num_replicas (int, optional): Number of processes participating in distributed training.
            rank (int, optional): Rank of the current process within num_replicas.
            shuffle (bool): Whether to shuffle the data at every epoch.
            seed (int): Random seed used to shuffle the sampler.
            drop_last (bool): Whether to drop the last incomplete batch if it’s smaller than batch_size.
        """
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        if not self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Adjust num_samples so we don't overshoot by duplicating samples
            self.num_samples = int(
                math.ceil((len(self.dataset) - self.rank) * 1.0 / self.num_replicas)
            )
            self.total_size = len(self.dataset)
