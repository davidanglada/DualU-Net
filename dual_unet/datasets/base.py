import abc
from typing import Any, List, Tuple, Type


class BaseCellCOCO:
    """
    An abstract base class for handling cell datasets in a COCO-like format.
    Subclasses must implement properties and methods that define the dataset's structure
    and retrieval methods.
    """

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        """
        Returns:
            int: The number of classes in the dataset (excluding background if applicable).
        """
        pass

    @property
    @abc.abstractmethod
    def class_names(self) -> List[str]:
        """
        Returns:
            List[str]: A list of class names in the dataset.
        """
        pass

    @abc.abstractmethod
    def image_size(self, image_id: int = None, idx: int = None) -> Any:
        """
        Retrieves the dimensions of an image by either a unique ID or its index in the dataset.

        Args:
            image_id (int, optional): A unique identifier for the image.
            idx (int, optional): The index of the image in the dataset.

        Returns:
            Any: The size information (e.g., (width, height)) of the requested image.
        """
        pass

    @abc.abstractmethod
    def get_raw_image(self, image_id: int = None, idx: int = None) -> Any:
        """
        Retrieves the raw image data by either a unique ID or its index in the dataset.

        Args:
            image_id (int, optional): A unique identifier for the image.
            idx (int, optional): The index of the image in the dataset.

        Returns:
            Any: The raw image (e.g., a NumPy array, PIL Image, or similar).
        """
        pass


def DetectionWrapper(base_class: Type[BaseCellCOCO]) -> Type[BaseCellCOCO]:
    """
    A wrapper that creates a Detection subclass of the provided base class.
    This Detection class sets the number of classes to 1 (representing nuclei)
    and modifies the target data accordingly.

    Args:
        base_class (Type[BaseCellCOCO]): The base class to wrap.

    Returns:
        Type[BaseCellCOCO]: A new class that inherits from the given base class.
    """

    class Detection(base_class):
        """
        Subclass of the provided base class with a fixed number of classes (one: 'nuclei').
        """

        @property
        def num_classes(self) -> int:
            """
            Returns:
                int: The number of classes, fixed to 1 for 'nuclei'.
            """
            return 1

        @property
        def class_names(self) -> List[str]:
            """
            Returns:
                List[str]: A single-element list containing 'nuclei'.
            """
            return ['nuclei']

        def __getitem__(self, idx: int) -> Tuple[Any, Any]:
            """
            Retrieves an image and target pair from the dataset and ensures that any
            positive 'category_id' values are set to 1.

            Args:
                idx (int): Index of the item in the dataset.

            Returns:
                Tuple[Any, Any]: A tuple containing (image data, target data).
            """
            img, tgt = super(Detection, self).__getitem__(idx)
            for i in range(len(tgt)):
                if tgt[i]['category_id'] > 0:
                    tgt[i]['category_id'] = 1
            return img, tgt

    Detection.__name__ = 'Detection' + base_class.__name__
    return Detection
