import os
import os.path as osp
import torch
import torchvision
from typing import Any, Dict, Optional, Union, Tuple
from PIL import Image
import torch.utils.data as data
from .base import BaseCellCOCO


class ImageOnly(data.Dataset):
    """
    A lightweight dataset class that simply returns images from a folder,
    ignoring targets (returns None instead).

    Folder structure example:
        root/fold/images/*.png (or .jpg, .jpeg, etc.)

    Args:
        root (str): The root directory of the dataset.
        fold (Union[str, int]): The dataset split or fold, e.g. "train", "val",
            or an integer for a fold number.
        transforms (Optional[Any]): Transformations to apply to the image.
                                     Typically a torchvision or custom transform.
    """

    def __init__(
        self,
        root: str,
        fold: Union[str, int],
        transforms: Optional[Any] = None
    ) -> None:
        super().__init__()
        self.root = root

        # If fold is an integer, convert it to a string like "fold1"
        fold = fold if isinstance(fold, str) else f'fold{fold}'
        self.fold = fold

        # Path to the images subdirectory
        self.img_folder = osp.join(root, fold, 'images')

        # Collect all image filenames (png, jpg, jpeg, tif, etc.)
        valid_exts = ('.png', '.jpg', '.jpeg')
        self.img_paths = [
            fname for fname in os.listdir(self.img_folder)
            if fname.lower().endswith(valid_exts)
        ]
        self.img_paths.sort()

        self.transforms = transforms

    @property
    def num_classes(self) -> int:
        """
        Number of classes (excluding background).
        Adjust as needed for your actual use case.

        Returns:
            int: e.g., 3 total classes (Positive, Negative, Stroma).
        """
        return 3

    def image_size(self, idx: int) -> torch.Tensor:
        """
        Provides the size of an image in the dataset.
        Currently returns a fixed size [1024, 1024]. Adjust as needed.

        Args:
            idx (int): Index of the sample in the dataset.

        Returns:
            torch.Tensor: A tensor containing [height, width].
        """
        return torch.tensor([1024, 1024])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Any]:
        """
        Retrieve an image at index `idx` and apply transforms if available.
        No target is returned, so the second element of the tuple is `None`.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Image.Image, None]: A tuple of (image, None).
        """
        img_name = self.img_paths[idx]
        img_path = osp.join(self.img_folder, img_name)

        # Open the image
        with Image.open(img_path) as img:
            # Convert to RGB if needed (or .convert("L") for grayscale, etc.)
            img = img.convert("RGB")

        # Apply any user-specified transforms
        # Depending on the transform pipeline, you may need to pass (img, None)
        # if transforms expect an (image, target) tuple. Adjust if so.

        if self.transforms is not None:
            img = self.transforms(img)

        return img

def build_imageonly_dataset(cfg: Dict[str, Any], split: str = 'test') -> Any:
    """
    Build a PannukeImageOnly dataset object according to the specified configuration and split.
    Returns only images (no targets).
    
    Args:
        cfg (Dict[str, Any]): Configuration dictionary (similar to your existing usage).
        split (str): Which dataset split to load (e.g., 'train', 'val', or 'test').
    
    Returns:
        PannukeImageOnly: A dataset that yields only images on indexing.
    """
    from .transforms import build_transforms  # same place you import your transforms
    root = cfg['dataset'][split]['root']
    fold = cfg['dataset'][split]['fold']
    is_train = (split == 'train')

    # Build any transforms you need; they will be applied to the image alone.
    transforms = build_transforms(cfg, split, is_train=is_train)
    
    dataset = ImageOnly(root=root, fold=fold, transforms=transforms)

    # Wrap the dataset to ensure TorchVision transforms.v2 properly handle the 'image' only
    # dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(
    #     dataset,
    #     # With no target, you can omit target_keys, or set it to ()
    #     target_keys=()
    # )
    return dataset