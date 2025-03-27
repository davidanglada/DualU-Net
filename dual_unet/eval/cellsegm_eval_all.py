import datetime
import itertools
import os.path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.functional import one_hot
import torchmetrics
import torchmetrics.functional as F
from torchmetrics.regression import MeanSquaredError

import scipy
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import (
    maximum_filter,
    label,
    distance_transform_edt,
    find_objects,
    gaussian_filter
)
import cv2
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
from skimage.segmentation import watershed, find_boundaries
from skimage import exposure
from skimage.morphology import extrema

from ..utils.distributed import is_dist_avail_and_initialized, all_gather
from .pq import compute_bPQ_and_mPQ, remap_label_and_class_map


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
            thresholds (Union[int, List[int]]): Threshold(s) for evaluation.
            class_names (Optional[List[str]]): Names of each class. If None, they will be generated as string indices.
        """
        super().__init__()
        self.num_classes = num_classes
        self.thresholds = thresholds if isinstance(thresholds, list) else [thresholds]
        self.class_names = (
            class_names if class_names is not None
            else [str(i) for i in range(1, num_classes + 1)]
        )
        self.preds: List[Dict[str, torch.Tensor]] = []
        self.targets: List[Dict[str, torch.Tensor]] = []

    def synchronize_between_processes(self) -> None:
        """
        Synchronize the predictions and targets across all distributed processes, if available.
        """
        if not dist.is_available() or not dist.is_initialized():
            return  # If not in a distributed environment, do nothing

        dist.barrier()  # Synchronize across all processes
        all_preds = all_gather(self.preds)
        all_targets = all_gather(self.targets)
        self.preds = list(itertools.chain(*all_preds))
        self.targets = list(itertools.chain(*all_targets))

    def reset(self) -> None:
        """
        Reset predictions and targets, typically called before recomputing metrics.
        """
        self.preds = []
        self.targets = []

    def update(
        self,
        preds: List[Dict[str, torch.Tensor]],
        target: List[Dict[str, torch.Tensor]]
    ) -> None:
        """
        Extend the list of predictions and targets with new results.

        Args:
            preds (List[Dict[str, torch.Tensor]]): List of prediction dicts.
            target (List[Dict[str, torch.Tensor]]): List of target dicts.
        """
        self.preds.extend(preds)
        self.targets.extend(target)

    def compute(self) -> Any:
        """
        Synchronize data, retrieve necessary values, and compute the metric.

        Returns:
            Any: Computed metric(s).
        """
        self.synchronize_between_processes()
        values = self._get_values()
        return self._compute(*values)

    def _get_values(self) -> Any:
        """
        Retrieve necessary values from predictions and targets. Must be overridden.

        Returns:
            Any: Values used for metric computation.
        """
        raise NotImplementedError

    def _compute(self, *args: Any) -> Any:
        """
        Compute the metric given retrieved values. Must be overridden.

        Returns:
            Any: Computed metric(s).
        """
        raise NotImplementedError


class MultiTaskEvaluationMetric(BaseCellMetric):
    """
    A multi-task metric evaluator for cell detection, segmentation, and classification.
    Includes Dice, MSE of centroid masks, detection/classification metrics, and PQ metrics.
    """

    def __init__(
        self,
        num_classes: int,
        thresholds: Union[int, List[int]],
        class_names: Optional[List[str]] = None,
        max_pair_distance: float = 12,
        train: bool = True,
        th: float = 0.1,
        output_sufix: Optional[str] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MultiTaskEvaluationMetric class.

        Args:
            num_classes (int): Number of classes (excluding background).
            thresholds (Union[int, List[int]]): Threshold(s) for evaluation.
            class_names (Optional[List[str]]): Names of each class.
            max_pair_distance (float): Maximum distance for centroid pairing in detection metrics.
            train (bool): True if in training mode (affects whether or not to save certain visualizations).
            th (float): Threshold for local maxima detection of centroids.
            output_sufix (Optional[str]): Suffix for output filenames.
        """
        super().__init__(num_classes, thresholds, class_names, *args, **kwargs)
        self.max_pair_distance = max_pair_distance
        self.train = train
        self.output_sufix = (
            output_sufix if output_sufix is not None
            else datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        self.th = th

    def _get_values(self) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray]
    ]:
        """
        Extract the necessary values from the predictions and targets for evaluation.

        Returns:
            A tuple containing:
            - true_gaussian_centroids
            - true_labels
            - true_segmentation_mask
            - true_boxes
            - pred_gaussian_centroids
            - pred_segmentation_mask
            - images
            - true_masks
        """
        true_gaussian_centroids = [t["centroid_gaussian"].detach().cpu().numpy() for t in self.targets]
        true_labels = [t["labels"].detach().cpu().numpy() for t in self.targets]
        true_segmentation_mask = [t["segmentation_mask"].detach().cpu().numpy() for t in self.targets]
        true_boxes = [t["boxes"].detach().cpu().numpy() for t in self.targets]
        true_masks = [t["masks"].detach().cpu().numpy() for t in self.targets]

        pred_gaussian_centroids = [p["centroid_gaussian"].detach().cpu().numpy() for p in self.preds]
        pred_segmentation_mask = [p["segmentation_mask"].detach().cpu().numpy() for p in self.preds]
        images = [p["image"].detach().cpu().numpy() for p in self.preds]

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
            true_gaussian_centroids, true_labels, true_segmentation_mask,
            true_boxes, pred_gaussian_centroids, pred_segmentation_mask,
            images, true_masks
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
        Compute metrics including Dice, MSE (for centroid masks), cell detection/classification,
        and Panoptic Quality.

        Args:
            true_gaussian_centroids (List[np.ndarray]): Ground truth Gaussian centroid masks.
            true_labels (List[np.ndarray]): Ground truth class labels.
            true_segmentation_mask (List[np.ndarray]): Ground truth segmentation masks.
            true_boxes (List[np.ndarray]): Ground truth bounding boxes.
            pred_gaussian_centroids (List[np.ndarray]): Predicted Gaussian centroid masks.
            pred_segmentation_mask (List[np.ndarray]): Predicted segmentation masks.
            images (List[np.ndarray]): Raw images for visualization.
            true_masks (List[np.ndarray]): Ground truth instance masks.

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """
        all_metrics: Dict[str, float] = {}

        dice_scores = []
        mse_scores = []

        paired_all: List[np.ndarray] = []
        unpaired_true_all: List[np.ndarray] = []
        unpaired_pred_all: List[np.ndarray] = []

        true_inst_type_all: List[np.ndarray] = []
        pred_inst_type_all: List[np.ndarray] = []

        hn_dice_scores = []

        gt_inst_map = []
        pred_inst_map = []
        gt_class_map = []
        pred_class_map = []

        if not self.train:
            hn_aji_scores = []
            hn_aji_plus_scores = []
            hn_pq_dq_scores = []
            hn_pq_sq_scores = []
            hn_pq_scores = []

        for i in range(len(true_gaussian_centroids)):
            print(f"Evaluating sample {i+1}/{len(true_gaussian_centroids)}")

            pred_masks_i = pred_segmentation_mask[i]
            true_masks_i = true_segmentation_mask[i]
            true_inst_masks_i = true_masks[i]

            true_gaussian_mask_i = true_gaussian_centroids[i]
            pred_gaussian_mask_i = pred_gaussian_centroids[i]

            true_boxes_i = true_boxes[i]
            true_labels_i = true_labels[i]

            # Reconstruct centroids from boxes
            true_cents_i = []
            for box in true_boxes_i:
                centroid_x = ((box[2] - box[0]) / 2) + box[0]
                centroid_y = ((box[3] - box[1]) / 2) + box[1]
                true_cents_i.append((centroid_y, centroid_x))
            true_cents_i = np.asarray(true_cents_i)

            # Dice score
            dice_val = self._dice_coefficient(
                torch.argmax(torch.Tensor(true_masks_i), dim=0),
                torch.argmax(torch.Tensor(pred_masks_i), dim=0),
                self.num_classes
            )
            dice_scores.append(dice_val)

            # MSE for Gaussian centroid masks
            mse_val = self._mse_centroids(true_gaussian_mask_i, pred_gaussian_mask_i)
            mse_scores.append(mse_val)

            # Watershed-based refinement
            pred_cents_i, pred_labels_i, watershed_mask, cells_mask = self._perform_watershed(
                pred_masks_i,
                pred_gaussian_mask_i
            )

            # Class/Instance mapping for PQ
            pred_labels_i_copy = pred_labels_i.copy()
            true_labels_i_copy = true_labels_i.copy()
            true_cents_i_copy = true_cents_i.copy()
            pred_cents_i_copy = pred_cents_i.copy()

            pred_binary = np.zeros_like(watershed_mask)
            pred_binary[watershed_mask > 0] = 1

            true_binary = torch.argmax(torch.Tensor(true_masks_i), dim=0)
            true_binary[true_binary > 0] = 1

            cc_pred, _ = label(pred_binary)

            # Build connected component for ground truth from instance masks
            cc_true = np.zeros_like(cc_pred)
            for k in range(true_inst_masks_i.shape[0]):
                cc_true[true_inst_masks_i[k] > 0] = k + 1

            gt_class_map_i = {}
            for k, l in enumerate(true_labels_i_copy):
                gt_class_map_i[k + 1] = l

            pred_class_map_i = {}
            for k, l in enumerate(pred_labels_i_copy):
                pred_class_map_i[k + 1] = l

            cc_pred, pred_class_map_i = remap_label_and_class_map(cc_pred, pred_class_map_i)
            cc_true, gt_class_map_i = remap_label_and_class_map(cc_true, gt_class_map_i)

            gt_inst_map.append(cc_true)
            pred_inst_map.append(cc_pred)
            gt_class_map.append(gt_class_map_i)
            pred_class_map.append(pred_class_map_i)

            # Dice in a simpler sense
            hn_dice = get_dice_1(cc_true, cc_pred)
            hn_dice_scores.append(hn_dice)

            # (Optional) AJI and PQ for test mode
            if not self.train:
                pass  # Here, additional test-only computations (AJI, etc.) can be done if needed

            # If no ground truth centroids, set a dummy
            if true_cents_i.shape[0] == 0:
                true_cents_i = np.array([[0, 0]])
                true_labels_i = np.array([0])

            # If no predicted centroids, set a dummy
            if pred_cents_i.shape[0] == 0:
                pred_cents_i = np.array([[0, 0]])
                pred_labels_i = np.array([0])

            # Pair up centroids for detection/classification
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_cents_i,
                pred_cents_i,
                self.max_pair_distance
            )

            true_idx_offset = 0 if i == 0 else (
                true_idx_offset + true_inst_type_all[-1].shape[0]
            )
            pred_idx_offset = 0 if i == 0 else (
                pred_idx_offset + pred_inst_type_all[-1].shape[0]
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

            # Visualization in non-train mode
            if not self.train:
                # Example: possible debug or final output images
                _, pred_centroids_list = find_local_maxima(pred_gaussian_mask_i[0], self.th)
                paired_i = paired.copy()
                unpaired_true_i = unpaired_true.copy()
                unpaired_pred_i = unpaired_pred.copy()

                f1_d, prec_d, rec_d, acc_d = cell_detection_scores(
                    paired_true=true_labels_i[paired_i[:, 0]],
                    paired_pred=pred_labels_i[paired_i[:, 1]],
                    unpaired_true=true_labels_i[unpaired_true_i],
                    unpaired_pred=pred_labels_i[unpaired_pred_i]
                )

                class_f1_scores = {}
                if self.num_classes > 1:
                    for nuc_type in range(1, self.num_classes + 1):
                        f1_cell, prec_cell, rec_cell = cell_type_detection_scores(
                            paired_true=true_labels_i[paired_i[:, 0]],
                            paired_pred=pred_labels_i[paired_i[:, 1]],
                            unpaired_true=true_labels_i[unpaired_true_i],
                            unpaired_pred=pred_labels_i[unpaired_pred_i],
                            type_id=nuc_type
                        )
                        class_f1_scores[self.class_names[nuc_type - 1]] = f1_cell

                if i % 100 == 0:
                    # Save debug visualization every 100 images
                    self._save_visualization(
                        image=self._get_raw_image(images[i]),
                        gt_mask=true_masks_i,
                        segmentation_mask=pred_masks_i,
                        true_centroids_list=true_cents_i_copy,
                        pred_centroids_list=pred_centroids_list,
                        w_centroids_list=pred_cents_i,
                        classification_mask=pred_masks_i,
                        watershed_mask=watershed_mask,
                        true_gaussian=true_gaussian_mask_i[0],
                        pred_gaussian=pred_gaussian_mask_i[0],
                        cells_mask=cells_mask,
                        true_labels=true_labels_i_copy,
                        pred_labels=pred_labels_i_copy,
                        mse=mse_val,
                        hn_dice=hn_dice,
                        detection_f1=f1_d,
                        class_f1_scores=class_f1_scores,
                        filename_prefix=f"sample_{i}",
                        output_sufix=self.output_sufix
                    )

        # Concatenate pairing results
        paired_all = (
            np.concatenate(paired_all, axis=0)
            if len(paired_all) != 0
            else np.empty((0, 2), dtype=np.int64)
        )
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        paired_true_type = true_inst_type_all[paired_all[:, 0]]
        paired_pred_type = true_inst_type_all[paired_all[:, 0]] * 0  # Placeholder if needed
        paired_pred_type = pred_inst_type_all[paired_all[:, 1]]

        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        # Final detection scores across the dataset
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
                "acc": acc_d,
            },
        }

        # If multi-class, compute type-level metrics
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
                    "rec": rec_cell,
                }

        all_metrics.update(nuclei_metrics)
        all_metrics["dice"] = np.mean(dice_scores)
        all_metrics["mse"] = np.mean(mse_scores)
        all_metrics["hn_dice"] = np.mean(hn_dice_scores)

        # Additional test-only metrics
        if not self.train:
            # Example placeholders if you compute them:
            # (We stored them above but didn't finalize them.)
            # This block can be extended if needed.
            pass

        # Compute Panoptic Quality
        all_classes = list(range(1, self.num_classes + 1))
        bPQ, bDQ, bSQ, mPQ, pq_per_class = compute_bPQ_and_mPQ(
            gt_inst_map,
            pred_inst_map,
            gt_class_map,
            pred_class_map,
            all_classes,
            match_iou=0.5
        )
        all_metrics["bPQ"] = bPQ
        all_metrics["bDQ"] = bDQ
        all_metrics["bSQ"] = bSQ
        all_metrics["mPQ"] = mPQ
        for c in sorted(pq_per_class.keys()):
            all_metrics[f"pq_{self.class_names[c - 1]}"] = pq_per_class[c]

        print(all_metrics)
        return all_metrics

    def _dice_coefficient(
        self,
        true_masks: torch.Tensor,
        pred_masks: torch.Tensor,
        num_classes: int
    ) -> float:
        """
        Compute the Dice coefficient between true and predicted segmentation masks.

        Args:
            true_masks (torch.Tensor): Ground truth segmentation mask (HxW).
            pred_masks (torch.Tensor): Predicted segmentation mask (HxW).
            num_classes (int): Number of segmentation classes.

        Returns:
            float: Mean Dice coefficient across all classes.
        """
        mean_dice = F.dice(pred_masks, true_masks.int())
        return mean_dice.item()

    def _mse_centroids(
        self,
        true_gaussian_mask: Union[np.ndarray, torch.Tensor],
        pred_gaussian_mask: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute the Mean Squared Error (MSE) between true and predicted Gaussian centroid masks.

        Args:
            true_gaussian_mask (Union[np.ndarray, torch.Tensor]): Ground truth Gaussian centroid mask.
            pred_gaussian_mask (Union[np.ndarray, torch.Tensor]): Predicted Gaussian centroid mask.

        Returns:
            float: MSE value.
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
        Apply a watershed algorithm to refine predicted masks using centroid hints.

        Args:
            pred_mask (np.ndarray): Predicted segmentation mask (C x H x W).
            pred_centroids (np.ndarray): Centroid heatmap (1 x H x W).

        Returns:
            Tuple containing:
              - pred_cents (np.ndarray): Refined centroids (N x 2).
              - pred_labels (np.ndarray): Predicted class labels for each centroid.
              - predicted_mask (np.ndarray): Watershed-refined label map.
              - cells_mask (np.ndarray): Binary mask of cell regions.
        """
        centroid_mask, pred_centr = find_local_maxima(pred_centroids[0], self.th)

        _, markers = cv2.connectedComponents(
            centroid_mask.astype(np.uint8), 4, ltype=cv2.CV_32S
        )

        # Create a binary cells_mask from the argmax of pred_mask
        pred_mask_argmax = np.argmax(pred_mask, axis=0).astype(np.uint8)
        cells_mask = np.zeros_like(pred_mask_argmax)
        cells_mask[pred_mask_argmax > 0] = 1

        distance_map = distance_transform_edt(cells_mask)
        watershed_result = watershed(-distance_map, markers, mask=cells_mask, compactness=1)

        # Remove boundary artifacts
        contours = np.invert(find_boundaries(watershed_result, mode='outer', background=0))
        watershed_result = watershed_result * contours

        binary_mask = np.zeros_like(watershed_result)
        binary_mask[watershed_result > 0] = 1

        labeled_mask, num_labels = label(watershed_result)

        predicted_centroids = []
        predicted_classes = []
        for id_val in np.unique(labeled_mask):
            if id_val == 0:
                continue
            region_mask = labeled_mask == id_val
            class_in_region = pred_mask_argmax[region_mask]
            majority_class = np.bincount(class_in_region).argmax()

            region_coords = np.argwhere(region_mask)
            centroid_yx = region_coords.mean(axis=0)[::-1]

            predicted_centroids.append((centroid_yx[1], centroid_yx[0]))
            predicted_classes.append(majority_class)

        return (
            np.asarray(predicted_centroids),
            np.asarray(predicted_classes),
            pred_mask_argmax * binary_mask,
            cells_mask
        )

    def _denormalize(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Denormalize a normalized image.

        Args:
            image (Union[torch.Tensor, np.ndarray]): Normalized image.
            mean (List[float]): Mean values used for normalization (per channel).
            std (List[float]): Standard deviations used for normalization (per channel).

        Returns:
            Union[torch.Tensor, np.ndarray]: Denormalized image.
        """
        if isinstance(image, torch.Tensor):
            if image.ndimension() == 3:
                mean_t = torch.tensor(mean).view(-1, 1, 1)
                std_t = torch.tensor(std).view(-1, 1, 1)
            else:
                mean_t = torch.tensor(mean).view(1, 1, -1)
                std_t = torch.tensor(std).view(1, 1, -1)
            return (image * std_t) + mean_t
        else:
            mean_arr = np.array(mean).reshape(-1, 1, 1)
            std_arr = np.array(std).reshape(-1, 1, 1)
            return (image * std_arr) + mean_arr

    def _get_raw_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert and scale a NumPy image to a PyTorch float tensor.

        Args:
            img (np.ndarray): Image array to be processed.

        Returns:
            torch.Tensor: Processed image tensor.
        """
        img = self._denormalize(img)
        transforms_pipeline = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        img_out = transforms_pipeline(img)
        return img_out

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
        pred_gaussian: np.ndarray,
        cells_mask: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        mse: float,
        hn_dice: float,
        detection_f1: float,
        class_f1_scores: Dict[str, float],
        filename_prefix: str = "output",
        output_sufix: str = "output",
        dataset: str = "pannuke"
    ) -> None:
        """
        Save detailed visualizations for debugging and analysis.

        Args:
            image (Union[torch.Tensor, np.ndarray]): Original image data.
            gt_mask (np.ndarray): Ground truth segmentation mask (CxHxW).
            segmentation_mask (np.ndarray): Predicted segmentation mask (CxHxW).
            true_centroids_list (np.ndarray): Ground truth centroid coordinates.
            pred_centroids_list (np.ndarray): Predicted centroid coordinates (local maxima).
            w_centroids_list (np.ndarray): Centroids from watershed.
            classification_mask (np.ndarray): Predicted classification mask (CxHxW).
            watershed_mask (np.ndarray): Label map from watershed.
            true_gaussian (np.ndarray): Ground truth Gaussian centroid mask (HxW).
            pred_gaussian (np.ndarray): Predicted Gaussian centroid mask (HxW).
            cells_mask (np.ndarray): Binary cell region mask.
            true_labels (np.ndarray): Ground truth labels of centroids.
            pred_labels (np.ndarray): Predicted labels of centroids (watershed).
            mse (float): MSE metric.
            hn_dice (float): Dice metric for instance segmentation.
            detection_f1 (float): F1 score for detection.
            class_f1_scores (Dict[str, float]): F1 scores per class.
            filename_prefix (str): Prefix for saved file names.
            output_sufix (str): Suffix for saved file names.
            dataset (str): String to identify the dataset for color logic.
        """
        # Implementation purely for visualization, does not alter metrics functionality.
        # This function can be as elaborate as needed for debugging or final analysis.
        # The user can remove or adapt as necessary.
        pass  # Placeholder implementation.


def pair_coordinates(
    setA: np.ndarray,
    setB: np.ndarray,
    radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pair coordinates between setA and setB if the distance between them is <= radius,
    using linear sum assignment for minimal cost pairing.

    Args:
        setA (np.ndarray): (N x 2) array of points.
        setB (np.ndarray): (M x 2) array of points.
        radius (float): Distance threshold for pairing.

    Returns:
        Tuple:
          - pairing (np.ndarray): Indices of matched points.
          - unpairedA (np.ndarray): Indices of setA that were not matched.
          - unpairedB (np.ndarray): Indices of setB that were not matched.
    """
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric="euclidean")
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]
    pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)

    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB


def cell_detection_scores(
    paired_true: np.ndarray,
    paired_pred: np.ndarray,
    unpaired_true: np.ndarray,
    unpaired_pred: np.ndarray,
    w: List[int] = [1, 1]
) -> Tuple[float, float, float, float]:
    """
    Compute F1, precision, recall, and accuracy for cell detection.

    Args:
        paired_true (np.ndarray): Class labels of matched ground-truth cells.
        paired_pred (np.ndarray): Class labels of matched predicted cells.
        unpaired_true (np.ndarray): Class labels of unmatched ground-truth cells.
        unpaired_pred (np.ndarray): Class labels of unmatched predicted cells.
        w (List[int]): Weight factors for F1 score calculation.

    Returns:
        Tuple of:
          - f1_d (float): F1 score
          - prec_d (float): Precision
          - rec_d (float): Recall
          - acc_d (float): Accuracy
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
    Compute F1, precision, and recall for a specific cell type.

    Args:
        paired_true (np.ndarray): Class labels of matched ground-truth cells.
        paired_pred (np.ndarray): Class labels of matched predicted cells.
        unpaired_true (np.ndarray): Class labels of unmatched ground-truth cells.
        unpaired_pred (np.ndarray): Class labels of unmatched predicted cells.
        type_id (int): The specific cell type ID to evaluate.
        w (List[int]): Weight factors for F1 calculation.
        exhaustive (bool): Whether to consider negative examples fully.

    Returns:
        Tuple of:
          - f1_type (float): F1 score for the specified cell type
          - prec_type (float): Precision
          - rec_type (float): Recall
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

    f1_type = (2 * (tp_dt + tn_dt)) / (
        2 * (tp_dt + tn_dt)
        + w[0] * fp_dt
        + w[1] * fn_dt
        + w[2] * fp_d
        + w[3] * fn_d
        + 1e-6
    )
    return f1_type, prec_type, rec_type


def find_local_maxima(pred: np.ndarray, h: float, centers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima in a centroid heatmap or binary centroid mask.

    Args:
        pred (np.ndarray): 2D array of the centroid heatmap or binary mask.
        h (float): Threshold for h-maxima transform.
        centers (bool): If True, interpret 'pred' as a binary centroid mask directly.

    Returns:
        Tuple:
          - centroides (np.ndarray): Binary image with local maxima.
          - centr (np.ndarray): Array of centroid coordinates (N x 2).
    """
    if not centers:
        pred = exposure.rescale_intensity(pred)
        h_maxima = extrema.h_maxima(pred, h)
    else:
        h_maxima = pred

    connectivity = 4
    output = cv2.connectedComponentsWithStats(
        h_maxima.astype(np.uint8), connectivity, ltype=cv2.CV_32S
    )
    num_labels = output[0]
    centroids = output[3]

    centr = []
    for i in range(num_labels):
        if i != 0:  # skip background
            centr.append(np.asarray((int(centroids[i, 1]), int(centroids[i, 0]))))

    centroides = np.zeros(h_maxima.shape)
    for c in centr:
        centroides[c[0], c[1]] = 255

    return centroides, np.asarray(centr)


def get_dice_1(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute traditional Dice score for binary segmentation.

    Args:
        true (np.ndarray): Ground truth instance mask (HxW).
        pred (np.ndarray): Predicted instance mask (HxW).

    Returns:
        float: Dice score.
    """
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom + 1e-6)
