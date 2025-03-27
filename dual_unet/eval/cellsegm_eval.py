import datetime
import itertools
import os.path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchmetrics.functional as F
from torchmetrics.regression import MeanSquaredError
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import (
    label,
    distance_transform_edt
)
import cv2
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
from skimage.segmentation import watershed, find_boundaries
from skimage import exposure
from skimage.morphology import extrema

from ..utils.distributed import is_dist_avail_and_initialized, all_gather
from .pq import (
    compute_bPQ_and_mPQ,
    remap_label_and_class_map
)


class BaseCellMetric:
    """
    A base class for cell-level metric computation that synchronizes predictions
    and targets across distributed processes if needed.
    """

    def __init__(
        self,
        num_classes: int,
        thresholds: Union[int, List[int]],
        class_names: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize the base metric class.

        Args:
            num_classes (int): Number of classes (excluding background).
            thresholds (Union[int, List[int]]): One or more thresholds for evaluation.
            class_names (Optional[List[str]]): Optional list of class names. If None,
                will create a list of stringified integers [1..num_classes].
        """
        super().__init__()
        self.num_classes = num_classes
        self.thresholds = thresholds if isinstance(thresholds, list) else [thresholds]
        self.class_names = (
            class_names
            if class_names is not None
            else [str(i) for i in range(1, num_classes + 1)]
        )
        self.preds: List[Dict[str, torch.Tensor]] = []
        self.targets: List[Dict[str, torch.Tensor]] = []

    def synchronize_between_processes(self) -> None:
        """
        Synchronize predictions and targets across processes in a distributed environment.
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        dist.barrier()
        all_preds = all_gather(self.preds)
        all_targets = all_gather(self.targets)
        self.preds = list(itertools.chain(*all_preds))
        self.targets = list(itertools.chain(*all_targets))

    def reset(self) -> None:
        """
        Reset stored predictions and targets.
        """
        self.preds = []
        self.targets = []

    def update(
        self,
        preds: List[Dict[str, torch.Tensor]],
        target: List[Dict[str, torch.Tensor]]
    ) -> None:
        """
        Add a new batch of predictions and targets.

        Args:
            preds (List[Dict[str, torch.Tensor]]): Batch predictions.
            target (List[Dict[str, torch.Tensor]]): Corresponding batch targets.
        """
        self.preds.extend(preds)
        self.targets.extend(target)

    def compute(self) -> Any:
        """
        Synchronize data across processes and compute the metric.

        Returns:
            Any: The computed metric(s).
        """
        self.synchronize_between_processes()
        values = self._get_values()
        return self._compute(*values)

    def _get_values(self) -> Any:
        """
        Extract relevant data from predictions and targets. Must be overridden by subclass.

        Returns:
            Any: Values needed for metric computation.
        """
        raise NotImplementedError

    def _compute(self, *args: Any) -> Any:
        """
        Compute the metric from extracted values. Must be overridden by subclass.

        Returns:
            Any: The metric computation result(s).
        """
        raise NotImplementedError


class MultiTaskEvaluationMetric(BaseCellMetric):
    """
    Multi-task metric computation for cell detection, segmentation, and classification.

    Computes:
      - Dice for segmentation
      - MSE for Gaussian centroid regression
      - Cell detection scores (F1, precision, recall, accuracy)
      - Optional classification scores
      - (Optional) Additional PQ metrics if needed
    """

    def __init__(
        self,
        num_classes: int,
        thresholds: Union[int, List[int]],
        class_names: Optional[List[str]] = None,
        max_pair_distance: float = 12.0,
        train: bool = True,
        th: float = 0.1,
        output_sufix: Optional[str] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize the multi-task metric evaluator.

        Args:
            num_classes (int): Number of classes (excluding background).
            thresholds (Union[int, List[int]]): One or more thresholds for evaluation.
            class_names (Optional[List[str]]): Names of each class. If None, uses a
                default list of strings [1..num_classes].
            max_pair_distance (float): Maximum pairing distance for detection evaluation.
            train (bool): If False, allows for additional test-time computations/visualizations.
            th (float): Threshold used to identify centroids in the Gaussian centroid mask.
            output_sufix (Optional[str]): Suffix for output file naming, defaults to a timestamp.
        """
        super().__init__(num_classes, thresholds, class_names, *args, **kwargs)
        self.max_pair_distance = max_pair_distance
        self.train = train
        self.output_sufix = (
            output_sufix
            if output_sufix is not None
            else datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        self.th = th

    def _get_values(
        self
    ) -> Tuple[
        List[np.ndarray],  # true_gaussian_centroids
        List[np.ndarray],  # true_labels
        List[np.ndarray],  # true_segmentation_mask
        List[np.ndarray],  # true_boxes
        List[np.ndarray],  # pred_gaussian_centroids
        List[np.ndarray],  # pred_segmentation_mask
        List[np.ndarray],  # images
        List[np.ndarray]   # true_masks (currently empty in this code, but included for extension)
    ]:
        """
        Extract the necessary values from the predictions and targets for evaluation.

        Returns:
            Tuple of lists of NumPy arrays: 
                (true_gaussian_centroids, true_labels, true_segmentation_mask, true_boxes,
                 pred_gaussian_centroids, pred_segmentation_mask, images, true_masks)
        """
        true_gaussian_centroids = [
            t["centroid_gaussian"].detach().cpu().numpy() for t in self.targets
        ]
        true_labels = [t["labels"].detach().cpu().numpy() for t in self.targets]
        true_segmentation_mask = [
            t["segmentation_mask"].detach().cpu().numpy() for t in self.targets
        ]
        true_boxes = [t["boxes"].detach().cpu().numpy() for t in self.targets]
        # In this implementation, we assume 'masks' might not exist. 
        true_masks: List[np.ndarray] = []

        pred_gaussian_centroids = [
            p["centroid_gaussian"].detach().cpu().numpy() for p in self.preds
        ]
        pred_segmentation_mask = [
            p["segmentation_mask"].detach().cpu().numpy() for p in self.preds
        ]
        images = [p["image"].detach().cpu().numpy() for p in self.preds]

        # Clean up references to avoid duplication
        for p in self.preds:
            del p["image"]
            del p["centroid_gaussian"]
            del p["segmentation_mask"]

        for t in self.targets:
            del t["centroid_gaussian"]
            del t["segmentation_mask"]
            del t["boxes"]
            del t["labels"]

        torch.cuda.empty_cache()

        return (
            true_gaussian_centroids,
            true_labels,
            true_segmentation_mask,
            true_boxes,
            pred_gaussian_centroids,
            pred_segmentation_mask,
            images,
            true_masks
        )

    def _compute(
        self,
        true_gaussian_centroids: List[np.ndarray],
        true_labels: List[np.ndarray],
        true_segmentation_mask: List[np.ndarray],
        true_boxes: List[np.ndarray],
        pred_gaussian_centroids: List[np.ndarray],
        pred_segmentation_mask: List[np.ndarray],
        images: List[np.ndarray],
        true_masks: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Perform the main metric computation.

        Args:
            true_gaussian_centroids (List[np.ndarray]): Ground truth Gaussian centroid maps.
            true_labels (List[np.ndarray]): Ground truth class labels per instance.
            true_segmentation_mask (List[np.ndarray]): Ground truth segmentation masks.
            true_boxes (List[np.ndarray]): Ground truth bounding boxes.
            pred_gaussian_centroids (List[np.ndarray]): Predicted Gaussian centroid maps.
            pred_segmentation_mask (List[np.ndarray]): Predicted segmentation masks.
            images (List[np.ndarray]): Original image data for visualization.
            true_masks (List[np.ndarray]): (Optional) Ground truth instance masks if available.

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """
        all_metrics: Dict[str, float] = {}
        dice_scores: List[float] = []
        mse_scores: List[float] = []

        paired_all: List[np.ndarray] = []
        unpaired_true_all: List[np.ndarray] = []
        unpaired_pred_all: List[np.ndarray] = []

        true_inst_type_all: List[np.ndarray] = []
        pred_inst_type_all: List[np.ndarray] = []

        hn_dice_scores: List[float] = []

        true_idx_offset = 0
        pred_idx_offset = 0

        # Evaluate each sample
        for i in range(len(true_gaussian_centroids)):
            pred_masks_i = pred_segmentation_mask[i]
            true_masks_i = true_segmentation_mask[i]
            true_gaussian_mask_i = true_gaussian_centroids[i]
            pred_gaussian_mask_i = pred_gaussian_centroids[i]

            # Convert bounding boxes to centroid coordinates
            true_boxes_i = true_boxes[i]
            true_labels_i = true_labels[i]
            true_cents_i = []
            for box in true_boxes_i:
                centroid_x = ((box[2] - box[0]) / 2) + box[0]
                centroid_y = ((box[3] - box[1]) / 2) + box[1]
                true_cents_i.append((centroid_y, centroid_x))
            true_cents_i = np.asarray(true_cents_i)

            # Dice score
            dice_val = self._dice_coefficient(
                torch.argmax(torch.tensor(true_masks_i), dim=0),
                torch.argmax(torch.tensor(pred_masks_i), dim=0),
                self.num_classes
            )
            dice_scores.append(dice_val)

            # MSE for Gaussian centroid masks
            mse_val = self._mse_centroids(true_gaussian_mask_i, pred_gaussian_mask_i)
            mse_scores.append(mse_val)

            # Watershed to refine predicted centroids
            pred_cents_i, pred_labels_i, watershed_mask, cells_mask = self._perform_watershed(
                pred_masks_i,
                pred_gaussian_mask_i
            )

            # Compute a simple Dice (binary style) on the watershed result
            pred_binary = np.zeros_like(watershed_mask)
            pred_binary[watershed_mask > 0] = 1
            true_binary = torch.argmax(torch.tensor(true_masks_i), dim=0).numpy()
            true_binary[true_binary > 0] = 1

            cc_pred, _ = label(pred_binary)
            cc_true, _ = label(true_binary)
            hn_dice = get_dice_1(cc_true, cc_pred)
            hn_dice_scores.append(hn_dice)

            # Protect against empty centroids
            if true_cents_i.shape[0] == 0:
                true_cents_i = np.array([[0, 0]])
                true_labels_i = np.array([0])
            if pred_cents_i.shape[0] == 0:
                pred_cents_i = np.array([[0, 0]])
                pred_labels_i = np.array([0])

            # Pair centroids for detection
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_cents_i, pred_cents_i, self.max_pair_distance
            )

            true_idx_offset = (
                true_idx_offset + true_inst_type_all[-1].shape[0]
                if i != 0
                else 0
            )
            pred_idx_offset = (
                pred_idx_offset + pred_inst_type_all[-1].shape[0]
                if i != 0
                else 0
            )
            true_inst_type_all.append(true_labels_i)
            pred_inst_type_all.append(pred_labels_i)

            if paired.shape[0] != 0:
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                paired_all.append(paired)

            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            unpaired_true_all.append(unpaired_true)
            unpaired_pred_all.append(unpaired_pred)

            # If in test mode, optionally save visualizations
            if not self.train:
                image_i = self._get_raw_image(images[i])
                paired_i = paired.copy()
                unpaired_true_i = unpaired_true.copy()
                unpaired_pred_i = unpaired_pred.copy()
                f1_d, prec_d, rec_d, acc_d = cell_detection_scores(
                    paired_true=true_labels_i[paired_i[:, 0]],
                    paired_pred=pred_labels_i[paired_i[:, 1]],
                    unpaired_true=true_labels_i[unpaired_true_i],
                    unpaired_pred=pred_labels_i[unpaired_pred_i]
                )
                class_f1_scores: Dict[str, float] = {}
                if self.num_classes > 1:
                    for nuc_type in range(1, self.num_classes + 1):
                        f1_cell, _, _ = cell_type_detection_scores(
                            paired_true=true_labels_i[paired_i[:, 0]],
                            paired_pred=pred_labels_i[paired_i[:, 1]],
                            unpaired_true=true_labels_i[unpaired_true_i],
                            unpaired_pred=pred_labels_i[unpaired_pred_i],
                            type_id=nuc_type
                        )
                        class_f1_scores[self.class_names[nuc_type - 1]] = f1_cell

                # Save visualizations
                self._save_visualization(
                    image=image_i,
                    gt_mask=true_masks_i,
                    segmentation_mask=pred_masks_i,
                    true_centroids_list=true_cents_i,
                    pred_centroids_list=pred_cents_i,
                    w_centroids_list=pred_cents_i,
                    classification_mask=pred_masks_i,
                    watershed_mask=watershed_mask,
                    true_gaussian=true_gaussian_mask_i[0],
                    cells_mask=cells_mask,
                    true_labels=true_labels_i,
                    pred_labels=pred_labels_i,
                    mse=mse_val,
                    hn_dice=hn_dice,
                    detection_f1=f1_d,
                    class_f1_scores=class_f1_scores,
                    filename_prefix=f"sample_{i}",
                    output_sufix=self.output_sufix
                )

        # Aggregate detection/classification results
        if len(paired_all) == 0:
            # No matched centroids at all
            paired_all_concat = np.empty((0, 2), dtype=np.int64)
        else:
            paired_all_concat = np.concatenate(paired_all, axis=0)

        unpaired_true_all_concat = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all_concat = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all_concat = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all_concat = np.concatenate(pred_inst_type_all, axis=0)

        paired_true_type = true_inst_type_all_concat[paired_all_concat[:, 0]]
        paired_pred_type = pred_inst_type_all_concat[paired_all_concat[:, 1]]
        unpaired_true_type = true_inst_type_all_concat[unpaired_true_all_concat]
        unpaired_pred_type = pred_inst_type_all_concat[unpaired_pred_all_concat]

        f1_d, prec_d, rec_d, acc_d = cell_detection_scores(
            paired_true=paired_true_type,
            paired_pred=paired_pred_type,
            unpaired_true=unpaired_true_type,
            unpaired_pred=unpaired_pred_type
        )
        nuclei_metrics = {
            "detection": {
                "f1": f1_d,
                "prec": prec_d,
                "rec": rec_d,
                "acc": acc_d
            }
        }

        # Compute classification scores if multiple classes
        if self.num_classes > 1:
            for nuc_type in range(1, self.num_classes + 1):
                f1_cell, prec_cell, rec_cell = cell_type_detection_scores(
                    paired_true_type,
                    paired_pred_type,
                    unpaired_true_type,
                    unpaired_pred_type,
                    nuc_type
                )
                nuclei_metrics[self.class_names[nuc_type - 1]] = {
                    "f1": f1_cell,
                    "prec": prec_cell,
                    "rec": rec_cell
                }

        # Aggregate main metrics
        all_metrics.update(nuclei_metrics)
        all_metrics["dice"] = float(np.mean(dice_scores))
        all_metrics["mse"] = float(np.mean(mse_scores))
        all_metrics["hn_dice"] = float(np.mean(hn_dice_scores))

        # Additional test-only results could be added here if needed.
        # For example, if you'd compute AJI or PQ in this script.

        print(all_metrics)
        return all_metrics

    def _dice_coefficient(
        self,
        true_masks: torch.Tensor,
        pred_masks: torch.Tensor,
        num_classes: int
    ) -> float:
        """
        Compute the mean Dice coefficient between two segmentation masks.

        Args:
            true_masks (torch.Tensor): Ground truth segmentation, shape (H, W).
            pred_masks (torch.Tensor): Predicted segmentation, shape (H, W).
            num_classes (int): Number of classes.

        Returns:
            float: Mean Dice coefficient across classes.
        """
        mean_dice = F.dice(pred_masks, true_masks.int())
        return mean_dice.item()

    def _mse_centroids(
        self,
        true_gaussian_mask: np.ndarray,
        pred_gaussian_mask: np.ndarray
    ) -> float:
        """
        Compute MSE between two Gaussian centroid masks.

        Args:
            true_gaussian_mask (np.ndarray): Ground truth centroid mask.
            pred_gaussian_mask (np.ndarray): Predicted centroid mask.

        Returns:
            float: MSE.
        """
        if not isinstance(true_gaussian_mask, torch.Tensor):
            true_gaussian_mask = torch.tensor(true_gaussian_mask, dtype=torch.float32)
        if not isinstance(pred_gaussian_mask, torch.Tensor):
            pred_gaussian_mask = torch.tensor(pred_gaussian_mask, dtype=torch.float32)

        mse_metric = MeanSquaredError()
        mse_value = mse_metric(pred_gaussian_mask, true_gaussian_mask)
        return mse_value.item()

    def _perform_watershed(
        self,
        pred_mask: np.ndarray,
        pred_centroids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply watershed to refine predicted segmentation with centroid markers.

        Args:
            pred_mask (np.ndarray): Predicted segmentation (C x H x W).
            pred_centroids (np.ndarray): Predicted centroid heatmap (1 x H x W).

        Returns:
            Tuple of:
              - predicted_centroids (np.ndarray) -> shape (N, 2)
              - predicted_classes (np.ndarray) -> shape (N,)
              - predicted_mask (np.ndarray) -> label map (H x W)
              - cells_mask (np.ndarray) -> binary region mask (H x W)
        """
        centroid_mask, _ = find_local_maxima(pred_centroids[0], self.th)
        _, markers = cv2.connectedComponents(
            centroid_mask.astype(np.uint8), 4, ltype=cv2.CV_32S
        )

        # Build a binary mask of the predicted region
        pred_mask_argmax = np.argmax(pred_mask, axis=0).astype(np.uint8)
        cells_mask = np.zeros_like(pred_mask_argmax)
        cells_mask[pred_mask_argmax > 0] = 1

        dist_map = distance_transform_edt(cells_mask)
        watershed_result = watershed(-dist_map, markers, mask=cells_mask, compactness=1)

        # Remove boundary pixels to refine instance separation
        contours = np.invert(find_boundaries(watershed_result, mode="outer", background=0))
        watershed_result = watershed_result * contours

        binary_mask = np.zeros_like(watershed_result)
        binary_mask[watershed_result > 0] = 1
        predicted_mask = pred_mask_argmax * binary_mask

        labeled_mask, _ = label(watershed_result)
        predicted_centroids = []
        predicted_classes = []

        for region_id in np.unique(labeled_mask):
            if region_id == 0:
                continue
            region_mask = labeled_mask == region_id
            class_in_region = pred_mask_argmax[region_mask]
            majority_class = np.bincount(class_in_region).argmax()
            predicted_mask[region_mask] = majority_class

            coords = np.argwhere(region_mask)
            centroid_yx = coords.mean(axis=0)[::-1]
            predicted_centroids.append((centroid_yx[1], centroid_yx[0]))
            predicted_classes.append(majority_class)

        return (
            np.asarray(predicted_centroids),
            np.asarray(predicted_classes),
            predicted_mask,
            cells_mask
        )

    def _denormalize(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Denormalize an image using the specified mean and std.

        Args:
            image (Union[torch.Tensor, np.ndarray]): Image to denormalize.
            mean (List[float]): Channel means.
            std (List[float]): Channel stds.

        Returns:
            Union[torch.Tensor, np.ndarray]: Denormalized image.
        """
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:  # (C,H,W)
                mean_t = torch.tensor(mean).view(-1, 1, 1)
                std_t = torch.tensor(std).view(-1, 1, 1)
            else:  # (H,W,C)
                mean_t = torch.tensor(mean).view(1, 1, -1)
                std_t = torch.tensor(std).view(1, 1, -1)
            return (image * std_t) + mean_t
        else:
            mean_arr = np.array(mean).reshape(-1, 1, 1)
            std_arr = np.array(std).reshape(-1, 1, 1)
            return (image * std_arr) + mean_arr

    def _get_raw_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert a NumPy image to a Torch tensor with optional denormalization.

        Args:
            img (np.ndarray): The image array.

        Returns:
            torch.Tensor: The processed image tensor.
        """
        img = self._denormalize(img)
        transforms_pipeline = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        return transforms_pipeline(img)

    def _save_visualization(
        self,
        image: Union[torch.Tensor, np.ndarray],
        gt_mask: np.ndarray,
        segmentation_mask: np.ndarray,
        true_centroids_list: np.ndarray,
        pred_centroids_list: np.ndarray,
        w_centroids_list: np.ndarray,
        classification_mask: np.ndarray,
        watershed_mask: np.ndarray,
        true_gaussian: np.ndarray,
        cells_mask: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        mse: float,
        hn_dice: float,
        detection_f1: float,
        class_f1_scores: Dict[str, float],
        filename_prefix: str = "output",
        output_sufix: str = "output",
        dataset: str = "ki67"
    ) -> None:
        """
        Save visualization of segmentation, centroids, and classification on the original image.
        This method does not affect the core metrics and is purely for debugging/analysis.

        Args:
            image (Union[torch.Tensor, np.ndarray]): Original image data [C,H,W] or [H,W,C].
            gt_mask (np.ndarray): Ground truth segmentation mask [C,H,W].
            segmentation_mask (np.ndarray): Predicted segmentation mask [C,H,W].
            true_centroids_list (np.ndarray): Ground truth centroid coords [N,2].
            pred_centroids_list (np.ndarray): Predicted centroid coords from local maxima [N,2].
            w_centroids_list (np.ndarray): Refined centroids from watershed [N,2].
            classification_mask (np.ndarray): Predicted classification mask [C,H,W].
            watershed_mask (np.ndarray): Label map from watershed [H,W].
            true_gaussian (np.ndarray): Ground truth Gaussian centroid mask [H,W].
            cells_mask (np.ndarray): Binary mask of predicted cell regions [H,W].
            true_labels (np.ndarray): Ground truth labels for centroids [N].
            pred_labels (np.ndarray): Predicted labels for centroids [N].
            mse (float): MSE between centroid masks.
            hn_dice (float): Dice for instance segmentation from watershed.
            detection_f1 (float): Detection F1 for matched centroids.
            class_f1_scores (Dict[str, float]): Class-wise F1 detection scores.
            filename_prefix (str): Name prefix for the saved file.
            output_sufix (str): Suffix for the saved file.
            dataset (str): String tag to differentiate dataset color maps, etc.
        """
        # Implementation for saving debug visualizations to files.
        # This code does not affect the metric values. Feel free to customize or remove if unused.
        pass


def pair_coordinates(
    setA: np.ndarray,
    setB: np.ndarray,
    radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pair coordinates from setA to setB within a given radius using linear sum assignment.

    Args:
        setA (np.ndarray): N x 2 array of points.
        setB (np.ndarray): M x 2 array of points.
        radius (float): Maximum allowable distance to pair points.

    Returns:
        Tuple:
          (paired_indices, unpairedA, unpairedB).
    """
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric="euclidean")
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)
    pair_cost = pair_distance[indicesA, paired_indicesB]
    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]
    pairing = np.column_stack([pairedA, pairedB])
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB


def cell_detection_scores(
    paired_true: np.ndarray,
    paired_pred: np.ndarray,
    unpaired_true: np.ndarray,
    unpaired_pred: np.ndarray,
    w: List[float] = [1.0, 1.0]
) -> Tuple[float, float, float, float]:
    """
    Compute detection metrics (F1, precision, recall, accuracy).

    Args:
        paired_true (np.ndarray): Labels of matched ground-truth objects.
        paired_pred (np.ndarray): Labels of matched predicted objects.
        unpaired_true (np.ndarray): Labels of unmatched ground-truth objects.
        unpaired_pred (np.ndarray): Labels of unmatched predicted objects.
        w (List[float]): Weight factors for unpaired penalty in F1.

    Returns:
        (f1_d, prec_d, rec_d, acc_d).
    """
    tp_d = paired_pred.shape[0]
    fp_d = unpaired_pred.shape[0]
    fn_d = unpaired_true.shape[0]
    tp_tn_dt = (paired_pred == paired_true).sum()
    fp_fn_dt = (paired_pred != paired_true).sum()
    acc_d = tp_tn_dt / (tp_tn_dt + fp_fn_dt + 1e-6)
    prec_d = tp_d / (tp_d + fp_d + 1e-6)
    rec_d = tp_d / (tp_d + fn_d + 1e-6)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d + 1e-6)
    return f1_d, prec_d, rec_d, acc_d


def cell_type_detection_scores(
    paired_true: np.ndarray,
    paired_pred: np.ndarray,
    unpaired_true: np.ndarray,
    unpaired_pred: np.ndarray,
    type_id: int,
    w: List[int] = [2, 2, 1, 1],
    exhaustive: bool = True
) -> Tuple[float, float, float]:
    """
    Compute detection metrics (F1, precision, recall) for a specific type/class.

    Args:
        paired_true (np.ndarray): Matched ground-truth labels.
        paired_pred (np.ndarray): Matched predicted labels.
        unpaired_true (np.ndarray): Unmatched ground-truth labels.
        unpaired_pred (np.ndarray): Unmatched predicted labels.
        type_id (int): Class ID of interest.
        w (List[int]): Weights for false positives and false negatives in F1.
        exhaustive (bool): If False, ignore unpaired with label == -1.

    Returns:
        (f1_type, prec_type, rec_type).
    """
    type_samples = (paired_true == type_id) | (paired_pred == type_id)
    paired_true = paired_true[type_samples]
    paired_pred = paired_pred[type_samples]

    tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
    tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
    fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
    fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

    if not exhaustive:
        ignore = (paired_true == -1).sum()
        fp_dt -= ignore

    fp_d = (unpaired_pred == type_id).sum()
    fn_d = (unpaired_true == type_id).sum()

    prec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[0] * fp_dt + w[2] * fp_d + 1e-6)
    rec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[1] * fn_dt + w[3] * fn_d + 1e-6)
    f1_type = (
        2.0
        * (tp_dt + tn_dt)
        / (
            2.0 * (tp_dt + tn_dt)
            + w[0] * fp_dt
            + w[1] * fn_dt
            + w[2] * fp_d
            + w[3] * fn_d
            + 1e-6
        )
    )
    return f1_type, prec_type, rec_type


def find_local_maxima(
    pred: np.ndarray,
    h: float,
    centers: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify local maxima in a heatmap or direct centroid mask.

    Args:
        pred (np.ndarray): 2D array of shape (H, W).
        h (float): Threshold for h-maxima.
        centers (bool): If True, interpret 'pred' as a binary centroid mask directly.

    Returns:
        (centroid_map, centroids_array).
    """
    if not centers:
        pred = exposure.rescale_intensity(pred)
        h_maxima = extrema.h_maxima(pred, h)
    else:
        h_maxima = pred

    connectivity = 4
    output = cv2.connectedComponentsWithStats(
        h_maxima.astype(np.uint8),
        connectivity,
        ltype=cv2.CV_32S
    )
    num_labels = output[0]
    centroids = output[3]

    centr_list = []
    for i in range(num_labels):
        if i != 0:  # Skip background
            centr_list.append(
                np.asarray((int(centroids[i, 1]), int(centroids[i, 0])))
            )
    centroid_map = np.zeros_like(h_maxima, dtype=np.uint8)
    for (r, c) in centr_list:
        centroid_map[r, c] = 255

    return centroid_map, np.asarray(centr_list)


def get_dice_1(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute traditional Dice for binary segmentation.

    Args:
        true (np.ndarray): Ground truth mask (H, W).
        pred (np.ndarray): Predicted mask (H, W).

    Returns:
        float: Dice score.
    """
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = (true * pred).sum()
    denom = (true + pred).sum()
    return 2.0 * inter / (denom + 1e-6)
