# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import torch
import torch.distributed as dist
import torchmetrics.functional as F
from torchmetrics.regression import MeanSquaredError
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path as osp
import os
import torchvision.transforms.v2 as v2

import scipy
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt, label
from skimage import exposure
from skimage.morphology import extrema
from skimage.segmentation import watershed, find_boundaries

from typing import Dict, List, Optional, Union
from collections import OrderedDict

from ..utils.distributed import all_gather
from .pq import compute_bPQ_and_mPQ, remap_label_and_class_map


#########################################################
# Helper Functions
#########################################################

def pair_coordinates(
    setA: np.ndarray,
    setB: np.ndarray,
    radius: float
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Pair points between two sets using the Hungarian (Munkres) assignment algorithm,
    subject to a distance threshold ('radius'). Points in setA and setB that 
    can match within the given radius are paired.

    Returns:
        pairing (np.ndarray): (K, 2) matched indices (i in setA, j in setB).
        unpairedA (np.ndarray): Indices of setA that are unmatched.
        unpairedB (np.ndarray): Indices of setB that are unmatched.
    """
    # Distance matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric="euclidean")

    # Apply Hungarian assignment
    indicesA, matchedB = linear_sum_assignment(pair_distance)
    costs = pair_distance[indicesA, matchedB]

    # Keep matches within 'radius'
    valid_matches = costs <= radius
    pairedA = indicesA[valid_matches]
    pairedB = matchedB[valid_matches]

    pairing = np.stack([pairedA, pairedB], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)

    return pairing, unpairedA, unpairedB


def cell_detection_scores(
    paired_true: np.ndarray,
    paired_pred: np.ndarray,
    unpaired_true: np.ndarray,
    unpaired_pred: np.ndarray,
    w: List[float] = [1, 1]
) -> (float, float, float, float):
    """
    Compute detection-level F1, precision, recall, and "label matching" accuracy 
    for matched/unmatched cells.

    Returns:
        f1_d, prec_d, rec_d, acc_d
    """
    tp_d = len(paired_true)
    fp_d = len(unpaired_pred)
    fn_d = len(unpaired_true)

    # Among matched pairs, how many have the same label
    tp_tn_dt = (paired_pred == paired_true).sum()
    fp_fn_dt = (paired_pred != paired_true).sum()
    acc_d = tp_tn_dt / (tp_tn_dt + fp_fn_dt) if (tp_tn_dt + fp_fn_dt) > 0 else 0.0

    prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
    rec_d  = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
    denom  = (2*tp_d + w[0]*fp_d + w[1]*fn_d)
    f1_d   = (2 * tp_d) / denom if denom > 0 else 0.0

    return f1_d, prec_d, rec_d, acc_d


def cell_type_detection_scores(
    paired_true: np.ndarray,
    paired_pred: np.ndarray,
    unpaired_true: np.ndarray,
    unpaired_pred: np.ndarray,
    type_id: int,
    w: List[float] = [2, 2, 1, 1],
    exhaustive: bool = True,
) -> (float, float, float):
    """
    Compute type-specific (class-specific) F1, precision, and recall for 
    nuclei labeled with 'type_id'.
    """
    # Only keep matched pairs where at least one is of the desired type
    type_samples = (paired_true == type_id) | (paired_pred == type_id)
    paired_true = paired_true[type_samples]
    paired_pred = paired_pred[type_samples]

    tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
    tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
    fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
    fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

    if not exhaustive:
        ignore = (paired_true == -1).sum()  # potential special-case ignoring
        fp_dt -= ignore

    fp_d = (unpaired_pred == type_id).sum()
    fn_d = (unpaired_true == type_id).sum()

    # Weighted precision / recall
    denom_prec = (tp_dt + tn_dt + w[0] * fp_dt + w[2] * fp_d)
    prec_type = (tp_dt + tn_dt) / denom_prec if denom_prec > 0 else 0.0

    denom_rec = (tp_dt + tn_dt + w[1] * fn_dt + w[3] * fn_d)
    rec_type = (tp_dt + tn_dt) / denom_rec if denom_rec > 0 else 0.0

    # Weighted F1
    denom_f1 = 2*(tp_dt + tn_dt) + w[0]*fp_dt + w[1]*fn_dt + w[2]*fp_d + w[3]*fn_d
    f1_type = (2*(tp_dt + tn_dt)) / denom_f1 if denom_f1 > 0 else 0.0

    return f1_type, prec_type, rec_type


def find_local_maxima(
    pred: np.ndarray,
    h: float,
    centers: bool = False
) -> (np.ndarray, np.ndarray):
    """
    Find local maxima in a 2D map (for centroid detection).
    If 'centers=True', assume 'pred' is already a binary map of centroids.
    Otherwise use h-maxima transform with threshold 'h'.
    """
    if not centers:
        pred = exposure.rescale_intensity(pred)
        h_maxima = extrema.h_maxima(pred, h)
    else:
        # Already a centroid map
        h_maxima = pred

    connectivity = 4
    output = cv2.connectedComponentsWithStats(h_maxima.astype(np.uint8), connectivity, ltype=cv2.CV_32S)
    num_labels = output[0]
    centroids = output[3]  # (num_labels, 2) -> [cx, cy]

    coords_list = []
    for i in range(num_labels):
        if i != 0:  # skip background
            coords_list.append((int(centroids[i, 1]), int(centroids[i, 0])))  # (y,x)

    centroid_map = np.zeros_like(h_maxima)
    for (r, c) in coords_list:
        centroid_map[r, c] = 255

    return centroid_map, np.array(coords_list, dtype=int)


def get_dice_1(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Traditional binary Dice coefficient. Both 'true' and 'pred' are
    converted to binary (1 => foreground, 0 => background).
    """
    t = (true > 0).astype(np.uint8)
    p = (pred > 0).astype(np.uint8)
    inter = (t * p).sum()
    denom = t.sum() + p.sum()
    return 2.0 * inter / denom if denom > 0 else 0.0


#########################################################
# Base Classes
#########################################################

class BaseCellMetric:
    """
    Base class for cell-based metrics in a (potentially) distributed environment.
    Stores predictions and targets across steps, can synchronize among processes.
    """
    def __init__(
        self,
        num_classes: int,
        thresholds: Union[int, List[int]],
        class_names: Optional[List[str]] = None
    ):
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
        Collect predictions/targets from all processes if distributed.
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        dist.barrier()

        all_preds = all_gather(self.preds)
        all_targets = all_gather(self.targets)
        self.preds = list(itertools.chain(*all_preds))
        self.targets = list(itertools.chain(*all_targets))

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds = []
        self.targets = []

    def update(
        self, 
        preds: List[Dict[str, torch.Tensor]], 
        targets: List[Dict[str, torch.Tensor]]
    ) -> None:
        """
        Add new predictions and targets to internal buffers.
        """
        self.preds.extend(preds)
        self.targets.extend(targets)

    def compute(self) -> Dict[str, float]:
        """
        Synchronize, gather final data, and compute the metrics.
        """
        self.synchronize_between_processes()
        values = self._get_values()
        return self._compute(*values)

    def _get_values(self):
        raise NotImplementedError

    def _compute(self, *args, **kwargs):
        raise NotImplementedError


#########################################################
# Main Multi-Task Metric
#########################################################

class MultiTaskEvaluationMetric(BaseCellMetric):
    """
    Evaluates cell segmentation, centroid detection/regression, classification, and PQ.
    
    - If train=True, we assume `t['masks']` is NOT available, skip instance-level metrics (PQ, instance-based dice, etc.).
    - If train=False, we assume `t['masks']` is available, compute additional instance-level metrics (Panoptic, etc.).
    """
    def __init__(
        self,
        num_classes: int,
        dataset: str,
        thresholds: Union[int, List[int]],
        class_names: Optional[List[str]] = None,
        max_pair_distance: float = 12,
        train: bool = True,
        th: float = 0.1,
        output_sufix: Optional[str] = None
    ):
        super().__init__(num_classes, thresholds, class_names)
        self.max_pair_distance = max_pair_distance
        self.train = train
        self.th = th
        self.output_sufix = output_sufix if output_sufix is not None \
            else datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.dataset = dataset

    def _get_values(self):
        """
        Extract necessary data from self.targets and self.preds. 
        If train=True, t['masks'] won't exist -> set them to None.
        """
        true_gaussian_centroids = [t["centroid_gaussian"].detach().cpu().numpy() for t in self.targets]
        true_labels = [t["labels"].detach().cpu().numpy() for t in self.targets]
        true_segmentation_mask = [t["segmentation_mask"].detach().cpu().numpy() for t in self.targets]
        true_boxes = [t["boxes"].detach().cpu().numpy() for t in self.targets]

        if not self.train:
            # We have instance-level masks
            true_masks = [t["masks"].detach().cpu().numpy() for t in self.targets]
        else:
            # None if not available in training
            true_masks = [None] * len(self.targets)

        # Predictions
        pred_gaussian_centroids = [p["centroid_gaussian"].detach().cpu().numpy() for p in self.preds]
        pred_segmentation_mask = [p["segmentation_mask"].detach().cpu().numpy() for p in self.preds]
        images = [p["image"].detach().cpu().numpy() for p in self.preds]

        # Clean memory
        for p in self.preds:
            del p["image"]
            del p["centroid_gaussian"]
            del p["segmentation_mask"]

        for t in self.targets:
            del t["centroid_gaussian"]
            del t["segmentation_mask"]
            del t["boxes"]
            del t["labels"]
            if not self.train:
                del t["masks"]

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
        true_masks: List[Optional[np.ndarray]]
    ) -> Dict[str, float]:
        """
        Compute overall metrics:
          - Dice (class segmentation)
          - MSE (centroid heatmap)
          - Detection F1 + classification
          - If train=False (eval mode), compute instance-based metrics (PQ, instance-based dice).
        """
        all_metrics: Dict[str, float] = {}

        # Basic aggregator metrics
        dice_scores = []
        mse_scores = []

        # Detection accumulators
        paired_all = []
        unpaired_true_all = []
        unpaired_pred_all = []
        true_inst_type_all = []
        pred_inst_type_all = []
        true_idx_offset = 0
        pred_idx_offset = 0

        # If we are not in train mode, we'll compute instance-level PQ metrics
        # that require true_masks
        if not self.train:
            hn_dice_scores = []
            gt_inst_map = []
            pred_inst_map = []
            gt_class_map = []
            pred_class_map = []
        else:
            # Just create empty placeholders so references won't break
            hn_dice_scores = []
            gt_inst_map = []
            pred_inst_map = []
            gt_class_map = []
            pred_class_map = []

        # -------------- Main Loop -------------- #
        for i in range(len(true_gaussian_centroids)):
            # Convert to numpy arrays
            t_gauss = true_gaussian_centroids[i]
            p_gauss = pred_gaussian_centroids[i]
            t_seg   = true_segmentation_mask[i]
            p_seg   = pred_segmentation_mask[i]

            # Build/compute "true" centroids from boxes
            boxes_i = true_boxes[i]
            labels_i = true_labels[i]
            true_cents_i = []
            for box in boxes_i:
                # box is (ymin, xmin, ymax, xmax)
                y_min, x_min, y_max, x_max = box
                centroid_x = ((x_max - x_min) / 2) + x_min
                centroid_y = ((y_max - y_min) / 2) + y_min
                true_cents_i.append((centroid_y, centroid_x))
            true_cents_i = np.array(true_cents_i)

            # Compute dice (class-level) and MSE (heatmaps)
            d_i = self._dice_coefficient(
                true_masks=np.argmax(t_seg, axis=0),
                pred_masks=np.argmax(p_seg, axis=0),
                num_classes=self.num_classes
            )
            dice_scores.append(d_i)

            m_i = self._mse_centroids(t_gauss, p_gauss)
            mse_scores.append(m_i)

            # Watershed-based instance parse on the predicted segmentation
            pred_cents_i, pred_labels_i, watershed_mask, cells_mask = self._perform_watershed(
                p_seg, p_gauss
            )

            # Handle centroid arrays (in case they are empty)
            if true_cents_i.shape[0] == 0:
                true_cents_i = np.array([[0, 0]])
                labels_i = np.array([0])
            if pred_cents_i.shape[0] == 0:
                pred_cents_i = np.array([[0, 0]])
                pred_labels_i = np.array([0])

            # Pair coordinates
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_cents_i, pred_cents_i, self.max_pair_distance
            )

            # Offset indexing for global pairing
            if i != 0:
                true_idx_offset += true_inst_type_all[-1].shape[0]
                pred_idx_offset += pred_inst_type_all[-1].shape[0]

            true_inst_type_all.append(labels_i)
            pred_inst_type_all.append(pred_labels_i)

            if paired.shape[0] != 0:
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                paired_all.append(paired)

            unpaired_true_all.append(unpaired_true + true_idx_offset)
            unpaired_pred_all.append(unpaired_pred + pred_idx_offset)

            # If we have instance masks (train=False), compute instance-level PQ, dice, etc.
            if not self.train and true_masks[i] is not None:
                true_inst_masks_i = true_masks[i]  # shape = (#instances, H, W)

                # Build a connected-component map for the pred
                cc_pred = label((watershed_mask > 0).astype(np.uint8))[0]
                cc_true = np.zeros_like(cc_pred, dtype=np.int32)
                for k in range(true_inst_masks_i.shape[0]):
                    cc_true[true_inst_masks_i[k] > 0] = k + 1

                # Build class maps
                gt_class_map_i = {}
                for k, lab in enumerate(labels_i):
                    gt_class_map_i[k + 1] = lab

                pred_class_map_i = {}
                for k, lab in enumerate(pred_labels_i):
                    pred_class_map_i[k + 1] = lab

                # Remap
                cc_pred, pred_class_map_i = remap_label_and_class_map(cc_pred, pred_class_map_i)
                cc_true, gt_class_map_i   = remap_label_and_class_map(cc_true, gt_class_map_i)

                # Save for PQ
                gt_inst_map.append(cc_true)
                pred_inst_map.append(cc_pred)
                gt_class_map.append(gt_class_map_i)
                pred_class_map.append(pred_class_map_i)

                # Instance-level dice
                hn_dice = get_dice_1(cc_true, cc_pred)
                hn_dice_scores.append(hn_dice)

                # Optionally save a visualization
                self._save_visualization(
                    image=self._get_raw_image(images[i]),
                    gt_mask=t_seg,
                    segmentation_mask=p_seg,
                    true_centroids_list=true_cents_i,
                    pred_centroids_list=pred_cents_i,
                    w_centroids_list=pred_cents_i,
                    classification_mask=p_seg,
                    watershed_mask=watershed_mask,
                    true_gaussian=t_gauss[0],
                    pred_gaussian=p_gauss[0],
                    cells_mask=cells_mask,
                    true_labels=labels_i,
                    pred_labels=pred_labels_i,
                    mse=m_i,
                    hn_dice=hn_dice,
                    detection_f1=0.0,   # If you want per-image detection F1
                    class_f1_scores={},
                    filename_prefix=f"sample_{i}",
                    output_sufix=self.output_sufix,
                    dataset=self.dataset
                )

        # Flatten detection
        paired_all = (
            np.concatenate(paired_all, axis=0) if len(paired_all) != 0 
            else np.empty((0, 2), dtype=np.int64)
        )
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        paired_true_type = (
            true_inst_type_all[paired_all[:, 0]] if paired_all.shape[0] > 0 
            else np.empty((0,), dtype=int)
        )
        paired_pred_type = (
            pred_inst_type_all[paired_all[:, 1]] if paired_all.shape[0] > 0 
            else np.empty((0,), dtype=int)
        )
        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        # Overall detection metrics
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

        # Class-wise detection
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

        # Aggregate basic metrics
        all_metrics = nuclei_metrics
        all_metrics["dice"] = float(np.mean(dice_scores))
        all_metrics["mse"] = float(np.mean(mse_scores))

        # If not training, compute instance-level PQ metrics
        if not self.train and len(gt_inst_map) > 0:
            hn_dice_mean = float(np.mean(hn_dice_scores)) if len(hn_dice_scores) > 0 else 0.0
            all_metrics["hn_dice"] = hn_dice_mean

            # Panoptic metrics
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
                class_idx = c - 1
                class_key = (
                    self.class_names[class_idx] 
                    if class_idx < len(self.class_names) 
                    else f"class_{c}"
                )
                all_metrics[f"pq_{class_key}"] = pq_per_class[c]

        print(all_metrics)
        return all_metrics

    ##################################################
    # Internal Utility Methods
    ##################################################

    def _dice_coefficient(
        self,
        true_masks: Union[torch.Tensor, np.ndarray],
        pred_masks: Union[torch.Tensor, np.ndarray],
        num_classes: int
    ) -> float:
        """
        Mean Dice across 'num_classes'.
        """
        if not isinstance(true_masks, torch.Tensor):
            true_masks = torch.tensor(true_masks)
        if not isinstance(pred_masks, torch.Tensor):
            pred_masks = torch.tensor(pred_masks)

        mean_dice = F.dice(pred_masks, true_masks.int())
        return float(mean_dice.item())

    def _mse_centroids(
        self,
        true_gaussian_mask: Union[torch.Tensor, np.ndarray],
        pred_gaussian_mask: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Compute MSE between ground-truth and predicted centroid heatmaps (1,H,W).
        """
        if not isinstance(true_gaussian_mask, torch.Tensor):
            true_gaussian_mask = torch.tensor(true_gaussian_mask)
        if not isinstance(pred_gaussian_mask, torch.Tensor):
            pred_gaussian_mask = torch.tensor(pred_gaussian_mask)

        mse_metric = MeanSquaredError()
        val = mse_metric(pred_gaussian_mask, true_gaussian_mask)
        return float(val.item())

    def _perform_watershed(
        self,
        pred_mask: np.ndarray,
        pred_centroids: np.ndarray
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Perform watershed with predicted centroid markers.
        Returns:
          predicted_centroids (N,2),
          predicted_classes (N,),
          predicted_mask (H,W) as majority-class map,
          cells_mask (H,W) binary foreground.
        """
        # Argmax to get single-channel predicted classes
        pred_mask_argmax = np.argmax(pred_mask, axis=0).astype(np.uint8)
        cells_mask = (pred_mask_argmax > 0).astype(np.uint8)

        # Markers from local maxima
        centroid_map, centroids_list = find_local_maxima(pred_centroids[0], self.th)
        _, markers = cv2.connectedComponents(centroid_map.astype(np.uint8), 4, ltype=cv2.CV_32S)

        # Distance transform
        dist_map = distance_transform_edt(cells_mask)
        watershed_result = watershed(-dist_map, markers, mask=cells_mask, compactness=1)

        # Remove boundary lines for clarity
        contours = np.invert(find_boundaries(watershed_result, mode='outer', background=0))
        watershed_result = watershed_result * contours

        # Relabel
        labeled_mask, num_labels = label(watershed_result)
        predicted_mask = np.zeros_like(labeled_mask, dtype=np.uint8)

        predicted_centroids = []
        predicted_classes = []

        for rid in range(1, num_labels + 1):
            region_mask = (labeled_mask == rid)
            class_in_region = pred_mask_argmax[region_mask]
            majority_class = np.bincount(class_in_region).argmax()
            predicted_mask[region_mask] = majority_class

            coords = np.argwhere(region_mask)
            cy, cx = coords.mean(axis=0)
            predicted_centroids.append((cy, cx))
            predicted_classes.append(majority_class)

        return (
            np.array(predicted_centroids, dtype=float),
            np.array(predicted_classes, dtype=int),
            predicted_mask,
            cells_mask
        )

    def _denormalize(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float]  = [0.229, 0.224, 0.225]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Denormalize image using (mean, std)."""
        if isinstance(image, torch.Tensor):
            if image.ndimension() == 3:
                mean = torch.tensor(mean).view(-1, 1, 1)
                std  = torch.tensor(std).view(-1, 1, 1)
            else:
                mean = torch.tensor(mean).view(1, 1, -1)
                std  = torch.tensor(std).view(1, 1, -1)
            return (image * std) + mean
        else:
            mean_arr = np.array(mean).reshape(-1, 1, 1)
            std_arr  = np.array(std).reshape(-1, 1, 1)
            return (image * std_arr) + mean_arr

    def _get_raw_image(self, img: np.ndarray) -> torch.Tensor:
        """Convert raw image array (C,H,W) from normalization to a float32 tensor in [0,1]."""
        img = self._denormalize(img)
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        return transforms(img)

    def _save_visualization(
        self,
        image: Union[np.ndarray, torch.Tensor],
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
        Save debug visualizations including original image, ground truth mask, watershed mask,
        Gaussian maps, and various metric info. Color mappings and legends are set based on the dataset.

        Args:
            image (np.ndarray or torch.Tensor): Input image.
            gt_mask (np.ndarray): Ground truth mask (e.g., one-hot or probability map).
            segmentation_mask (np.ndarray): Predicted segmentation mask (not used for display here).
            true_centroids_list (np.ndarray): True centroid coordinates.
            pred_centroids_list (np.ndarray): Predicted centroid coordinates.
            w_centroids_list (np.ndarray): Additional centroid data.
            classification_mask (np.ndarray): Classification mask.
            watershed_mask (np.ndarray): Watershed segmentation mask to be displayed as the prediction.
            true_gaussian (np.ndarray): Ground truth Gaussian centroid map.
            pred_gaussian (np.ndarray): Predicted Gaussian centroid map.
            cells_mask (np.ndarray): Binary mask of cell regions.
            true_labels (np.ndarray): True labels for instances.
            pred_labels (np.ndarray): Predicted labels for instances.
            mse (float): MSE metric value.
            hn_dice (float): HN Dice metric value.
            detection_f1 (float): Detection F1 metric value.
            class_f1_scores (Dict[str, float]): Dictionary of class-wise F1 scores.
            filename_prefix (str): Prefix for the output filename.
            output_sufix (str): Suffix for the output filename.
            dataset (str): Dataset flag ("consep", "ki67", "pannuke") to determine color mapping.
        """
        import matplotlib.pyplot as plt

        # Set color maps and legend elements based on dataset
        if dataset == "consep":
            class_colors = [
                [0, 0, 0],       # Background: Black
                [255, 0, 0],     # Miscellaneous: Red
                [0, 255, 0],     # Inflammatory: Green
                [0, 0, 255],     # Epithelial: Blue
                [255, 255, 0]    # Spindleshaped: Yellow
            ]
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='Miscellaneous', markerfacecolor='r', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Inflammatory', markerfacecolor='g', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Epithelial', markerfacecolor='b', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Spindleshaped', markerfacecolor='y', markersize=10)
            ]
        elif dataset == "ki67":
            class_colors = [
                [0, 0, 0],       # Background: Black
                [255, 0, 0],     # Class 1: Red
                [0, 255, 0],     # Class 2: Green
                [0, 0, 255]      # Class 3: Blue
            ]
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='Class1', markerfacecolor='r', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Class2', markerfacecolor='g', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Class3', markerfacecolor='b', markersize=10)
            ]
        elif dataset == "pannuke":
            class_colors = [
                [0, 0, 0],       # Background: Black
                [255, 0, 0],     # Neoplastic: Red
                [0, 255, 0],     # Inflammatory: Green
                [255, 255, 0],   # Connective: Yellow
                [255, 255, 255], # Necrosis: White
                [0, 0, 255]      # Epithelial: Blue
            ]
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='Neoplastic', markerfacecolor='r', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Inflammatory', markerfacecolor='g', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Connective', markerfacecolor='y', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Necrosis', markerfacecolor='w', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Epithelial', markerfacecolor='b', markersize=10)
            ]
        else:
            class_colors = None
            legend_elements = None

        # Convert image to numpy array if it's a torch.Tensor.
        if isinstance(image, torch.Tensor):
            image = image.permute(2, 0, 1).cpu().numpy()
        image = np.clip(image, 0, 1)

        # Use watershed_mask (instead of segmentation_mask) for predicted segmentation visualization.
        gt_labels = np.argmax(gt_mask, axis=0)
        pred_labels = watershed_mask

        if class_colors is not None:
            gt_color = np.zeros((gt_labels.shape[0], gt_labels.shape[1], 3), dtype=np.uint8)
            pred_color = np.zeros((pred_labels.shape[0], pred_labels.shape[1], 3), dtype=np.uint8)
            for cls in range(len(class_colors)):
                gt_color[gt_labels == cls] = class_colors[cls]
                pred_color[pred_labels == cls] = class_colors[cls]
        else:
            gt_color = gt_labels
            pred_color = pred_labels

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].imshow(image)
        axs[0, 0].set_title("Original Image")

        axs[0, 1].imshow(gt_color)
        axs[0, 1].set_title("GT Mask (Colored)")
        if legend_elements is not None:
            axs[0, 1].legend(handles=legend_elements, loc="upper right")

        # Show watershed mask as predicted segmentation
        axs[0, 2].imshow(pred_color)
        axs[0, 2].set_title("Watershed Mask (Colored)")

        axs[1, 0].imshow(true_gaussian, cmap="jet")
        axs[1, 0].set_title("True Gaussian")

        axs[1, 1].imshow(pred_gaussian, cmap="jet")
        axs[1, 1].set_title("Pred Gaussian")

        info_text = f"MSE={mse:.3f}\nHN_DICE={hn_dice:.3f}\nDetF1={detection_f1:.3f}"
        axs[1, 2].text(0.1, 0.5, info_text, fontsize=12)
        axs[1, 2].axis("off")

        plt.suptitle(f"{filename_prefix} | {output_sufix}")
        plt.tight_layout()
        save_dir = "./final_outputs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, f"{filename_prefix}_{output_sufix}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Visualization saved to {save_path}")

