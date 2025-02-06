import numpy as np
from typing import Dict, List, Tuple

def get_fast_pq(
    true: np.ndarray, 
    pred: np.ndarray, 
    match_iou: float = 0.5
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Compute single-class Panoptic Quality (PQ) metrics (DQ, SQ, PQ) given ground-truth and predicted 
    instance segmentation maps, each containing integer instance IDs.

    Args:
        true (np.ndarray): Ground-truth instance map, shape (H, W), where each pixel is an integer ID (0 for background).
        pred (np.ndarray): Predicted instance map, shape (H, W), same format as `true`.
        match_iou (float): IoU threshold for matching instances. Default is 0.5.

    Returns:
        (List[float], List[np.ndarray]):
            - A list [DQ, SQ, PQ] containing detection quality, segmentation quality, and panoptic quality.
            - A list [paired_true, paired_pred, unpaired_true, unpaired_pred] describing the matching:
                * paired_true  (np.ndarray): IDs from the ground truth that were matched.
                * paired_pred  (np.ndarray): IDs from the predictions that were matched.
                * unpaired_true (np.ndarray): Ground-truth IDs that were not matched.
                * unpaired_pred (np.ndarray): Predicted IDs that were not matched.
    """
    # Copy to avoid modifying inputs
    true = np.copy(true)
    pred = np.copy(pred)

    # Get unique IDs (including 0)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    # Build binary masks for each instance in true and pred
    true_masks = [None]  # index 0 is None for background
    for t in true_id_list[1:]:  # skip background (ID=0)
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None]
    for p in pred_id_list[1:]:  # skip background (ID=0)
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # pairwise_iou shape: (#true_instances, #pred_instances)
    pairwise_iou = np.zeros((len(true_id_list) - 1, len(pred_id_list) - 1), dtype=np.float64)

    # Calculate IoU for each pair of ground-truth and predicted instances
    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        # Overlapping predicted IDs in the region covered by the current true_id
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(np.unique(pred_true_overlap))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # skip background
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()   # union
            inter = (t_mask * p_mask).sum()   # intersection
            iou = 0.0
            if (total - inter) > 0:  # avoid zero division
                iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    # Matching step: either direct thresholding if match_iou >= 0.5 or Hungarian if < 0.5
    if match_iou >= 0.5:
        # Keep only pairwise IoUs above threshold
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        # Convert from 0-based indices to actual instance IDs
        paired_true += 1
        paired_pred += 1
    else:
        from scipy.optimize import linear_sum_assignment
        # Hungarian assignment on negative IoU to find max IoU pairing
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        valid = paired_iou > match_iou
        paired_true = paired_true[valid] + 1
        paired_pred = paired_pred[valid] + 1
        paired_iou = paired_iou[valid]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    # True positives
    tp = len(paired_true)
    # False positives
    fp = len(unpaired_pred)
    # False negatives
    fn = len(unpaired_true)

    # Detection Quality (DQ)
    denominator = (tp + 0.5 * fp + 0.5 * fn)
    dq = (tp / denominator) if denominator else 0.0

    # Segmentation Quality (SQ), average IoU among matched pairs
    sq = (paired_iou.sum() / (tp + 1.0e-6)) if tp > 0 else 0.0

    # Panoptic Quality (PQ)
    pq = dq * sq

    return [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def isolate_class(
    inst_map: np.ndarray, 
    class_map: Dict[int, int], 
    the_class: int
) -> np.ndarray:
    """
    Zero out all instance IDs in 'inst_map' that do NOT belong to 'the_class'.

    Args:
        inst_map (np.ndarray): A 2D array of instance IDs.
        class_map (Dict[int, int]): A mapping from instance_id -> class_label.
        the_class (int): The class label of interest.

    Returns:
        np.ndarray: A 2D array where only the instances belonging to `the_class` remain.
    """
    out_map = np.zeros_like(inst_map, dtype=inst_map.dtype)
    unique_ids = np.unique(inst_map)
    for inst_id in unique_ids:
        if inst_id == 0:
            continue
        if class_map[inst_id] == the_class:
            out_map[inst_map == inst_id] = inst_id
    return out_map


def compute_bPQ_and_mPQ(
    gt_inst_maps: List[np.ndarray],
    pred_inst_maps: List[np.ndarray],
    gt_class_map_dict: List[Dict[int, int]],
    pred_class_map_dict: List[Dict[int, int]],
    all_classes: List[int],
    match_iou: float = 0.5
) -> Tuple[float, float, float, float, Dict[int, float]]:
    """
    Compute binary PQ (bPQ) and multi-class PQ (mPQ) across a dataset of instance segmentations.

    1) bPQ, bDQ, bSQ: 
       - bPQ (binary PQ) is computed by treating all instances as one class. 
         We do get_fast_pq on each (gt, pred) pair and average over images.

    2) mPQ:
       - For each class c in all_classes, we isolate that class in the GT and prediction,
         then compute single-class PQ and accumulate raw TP/FP/FN/IoU.
       - Finally, we average across classes to get mPQ.

    Args:
        gt_inst_maps (List[np.ndarray]): List of ground-truth instance maps for each image.
        pred_inst_maps (List[np.ndarray]): List of predicted instance maps for each image.
        gt_class_map_dict (List[Dict[int,int]]): For each image, a dict mapping instance_id -> class_label.
        pred_class_map_dict (List[Dict[int,int]]): Same as above but for predictions.
        all_classes (List[int]): The set of class IDs to consider.
        match_iou (float): IoU threshold for instance matching, default=0.5.

    Returns:
        A tuple containing:
          - bPQ (float): Binary PQ (average over images).
          - bDQ (float): Binary DQ (average over images).
          - bSQ (float): Binary SQ (average over images).
          - mPQ (float): Multi-class PQ (averaged over classes).
          - class_to_pq (Dict[int,float]): Mapping from class ID to PQ for that class.
    """
    N = len(gt_inst_maps)

    # -----------------------------
    # 1) Compute bPQ, bDQ, bSQ
    # -----------------------------
    bPQ_list = []
    bDQ_list = []
    bSQ_list = []

    for i in range(N):
        [dq_i, sq_i, pq_i], _ = get_fast_pq(gt_inst_maps[i], pred_inst_maps[i], match_iou)
        bPQ_list.append(pq_i)
        bDQ_list.append(dq_i)
        bSQ_list.append(sq_i)

    bPQ = np.mean(bPQ_list)
    bDQ = np.mean(bDQ_list)
    bSQ = np.mean(bSQ_list)

    # -----------------------------
    # 2) Accumulate detection stats per class
    # -----------------------------
    class_TP = {c: 0 for c in all_classes}
    class_FP = {c: 0 for c in all_classes}
    class_FN = {c: 0 for c in all_classes}
    class_IoU_sum = {c: 0.0 for c in all_classes}

    for i in range(N):
        for c in all_classes:
            # Isolate class c in ground-truth
            c_gt_map = isolate_class(gt_inst_maps[i], gt_class_map_dict[i], c)
            # Remap labels so instance IDs are contiguous
            c_gt_map, _ = remap_label_and_class_map(c_gt_map, {})

            # Isolate class c in prediction
            c_pred_map = isolate_class(pred_inst_maps[i], pred_class_map_dict[i], c)
            c_pred_map, _ = remap_label_and_class_map(c_pred_map, {})

            # Single-class PQ for (image i, class c)
            [dq_ic, sq_ic, pq_ic], [paired_true, paired_pred, unpaired_true, unpaired_pred] = \
                get_fast_pq(c_gt_map, c_pred_map, match_iou)

            # Convert to raw counts
            tp_ic = len(paired_true)
            fp_ic = len(unpaired_pred)
            fn_ic = len(unpaired_true)
            sum_iou_ic = sq_ic * tp_ic  # IoU contribution for matched pairs

            # Accumulate
            class_TP[c] += tp_ic
            class_FP[c] += fp_ic
            class_FN[c] += fn_ic
            class_IoU_sum[c] += sum_iou_ic

    # -----------------------------
    # 3) Final PQ per class
    # -----------------------------
    class_to_pq = {}
    for c in all_classes:
        tp = class_TP[c]
        fp = class_FP[c]
        fn = class_FN[c]
        sum_iou = class_IoU_sum[c]

        if tp == 0:
            dq_c = 0.0
            sq_c = 0.0
            pq_c = 0.0
        else:
            dq_c = tp / (tp + 0.5 * fp + 0.5 * fn)
            sq_c = sum_iou / tp
            pq_c = dq_c * sq_c

        class_to_pq[c] = pq_c

    # Average PQ across classes
    mPQ = np.mean(list(class_to_pq.values()))

    return bPQ, bDQ, bSQ, mPQ, class_to_pq


def remap_label_and_class_map(
    inst_map: np.ndarray, 
    class_map: Dict[int, int]
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Remap instance IDs in `inst_map` to [1..K], preserving background (0) as is.
    Also update the `class_map` dict accordingly.

    Args:
        inst_map (np.ndarray): 2D array of instance IDs (0 for background).
        class_map (Dict[int,int]): A mapping from old instance_id -> class_label.

    Returns:
        (np.ndarray, Dict[int,int]):
            - new_inst_map (np.ndarray): The remapped instance map with IDs in [1..K].
            - new_class_map (Dict[int,int]): Remapped {new_inst_id -> class_label}.
    """
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids != 0]

    new_inst_map = np.zeros_like(inst_map, dtype=np.int32)
    new_class_map = {}

    for new_id, old_id in enumerate(inst_ids, start=1):
        # All pixels with old_id become new_id
        new_inst_map[inst_map == old_id] = new_id
        if old_id in class_map:
            new_class_map[new_id] = class_map[old_id]
        else:
            # If the old_id is not found in class_map, do nothing or handle as needed
            pass

    return new_inst_map, new_class_map
