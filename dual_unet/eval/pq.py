import numpy as np

def get_fast_pq(true, pred, match_iou=0.5):
    """
    Compute panoptic-quality (PQ) metrics for a single pair of (true, pred) instance segmentation maps,
    treating all instances as a single category (i.e., ignoring class labels).

    Args:
        true (np.ndarray): 2D array of instance IDs for ground truth. Shape: (H, W).
        pred (np.ndarray): 2D array of instance IDs for prediction. Shape: (H, W).
        match_iou (float): IoU threshold for matching true/pred instances. 
            If >= 0.5, we do direct thresholding on IoU. Otherwise, we solve an 
            assignment problem maximizing IoU subject to it being > match_iou.

    Returns:
        tuple: ([DQ, SQ, PQ], [paired_true, paired_pred, unpaired_true, unpaired_pred])
          - [DQ, SQ, PQ] is detection quality, segmentation quality, and panoptic quality.
          - The second element is the matching info:
            paired_true (list): True instance IDs matched to predictions.
            paired_pred (list): Prediction instance IDs matched to truth.
            unpaired_true (list): True instances with no match.
            unpaired_pred (list): Predicted instances with no match.
    """
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    # Build masks for each instance
    true_masks = [None]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    pairwise_iou = np.zeros((len(true_id_list) - 1, len(pred_id_list) - 1), dtype=np.float64)

    # Calculate pairwise IoU
    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(np.unique(pred_true_overlap))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            denom = (total - inter)
            iou = inter / denom if denom > 0 else 0.0
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    # Matching step
    if match_iou >= 0.5:
        # Direct threshold: pairwise_iou <= match_iou => set to 0
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        # matched
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        # Convert from 0-based to instance IDs
        paired_true += 1
        paired_pred += 1
    else:
        # Solve an assignment problem with a cost = -IoU 
        # to find maximal unique pairing
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[row_ind, col_ind]
        valid = paired_iou > match_iou
        paired_true = row_ind[valid] + 1
        paired_pred = col_ind[valid] + 1
        paired_iou = paired_iou[valid]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)

    # Detection Quality (DQ)
    denom = (tp + 0.5 * fp + 0.5 * fn)
    dq = tp / denom if denom else 0.0

    # Segmentation Quality (SQ)
    sq = paired_iou.sum() / (tp + 1.0e-6) if tp > 0 else 0.0

    # Panoptic Quality (PQ)
    pq = dq * sq

    return [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def isolate_class(inst_map, class_map, the_class):
    """
    Zero out all instance IDs in 'inst_map' that do NOT belong to 'the_class'.
    'class_map' is e.g. {instance_id -> class_label}.
    
    Args:
        inst_map (np.ndarray): 2D array of instance IDs.
        class_map (dict): Mapping from instance_id -> class_label.
        the_class (int): The class label to isolate.
    
    Returns:
        np.ndarray: A copy of inst_map where only the instances belonging to 'the_class' remain,
            and the others are zeroed out.
    """
    out_map = np.zeros_like(inst_map, dtype=inst_map.dtype)
    unique_ids = np.unique(inst_map)
    for inst_id in unique_ids:
        if inst_id == 0:
            continue
        if class_map.get(inst_id, None) == the_class:
            out_map[inst_map == inst_id] = inst_id
    return out_map


def remap_label_and_class_map(inst_map, class_map):
    """
    Remap instance IDs in `inst_map` to contiguous [1..K] while preserving 0 as background.
    Also remap the associated class_map accordingly.
    
    Args:
        inst_map (np.ndarray): 2D array of instance IDs.
        class_map (dict): {old_id -> class_label}.
    
    Returns:
        (np.ndarray, dict): 
          - new_inst_map: same shape as inst_map with instance IDs remapped to [1..K].
          - new_class_map: dict {new_id -> class_label}.
    """
    # Identify all unique IDs (ignoring 0)
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids != 0]

    new_inst_map = np.zeros_like(inst_map, dtype=np.int32)
    new_class_map = {}

    # Create a map from old ID to new ID
    for new_id, old_id in enumerate(inst_ids, start=1):
        new_inst_map[inst_map == old_id] = new_id
        if old_id in class_map:
            new_class_map[new_id] = class_map[old_id]
        else:
            # Possibly unknown or skip
            pass

    return new_inst_map, new_class_map


def compute_bPQ_and_mPQ(
    gt_inst_maps,
    pred_inst_maps,
    gt_class_map_dict,
    pred_class_map_dict,
    all_classes,
    match_iou=0.5
):
    """
    Compute both:
      1) bPQ (binary PQ), bDQ, bSQ: by ignoring classes and computing the average 
         panoptic quality across the entire dataset as though it's a single class.
      2) mPQ (multi-class PQ): by computing aggregated TP/FP/FN and IoU across 
         each class, then averaging over classes.

    Args:
        gt_inst_maps (list of np.ndarray): List of ground-truth instance maps, one per image.
        pred_inst_maps (list of np.ndarray): List of predicted instance maps, one per image.
        gt_class_map_dict (list of dict): For each image, a dict {instance_id -> class_label}.
        pred_class_map_dict (list of dict): For each image, a dict {instance_id -> class_label}.
        all_classes (list): The set of possible class labels (excluding background).
        match_iou (float): IoU threshold for matching instances.

    Returns:
        bPQ, bDQ, bSQ, mPQ, class_to_pq:
          - bPQ (float): The average binary panoptic quality across images.
          - bDQ (float): The average detection quality across images.
          - bSQ (float): The average segmentation quality across images.
          - mPQ (float): The multi-class panoptic quality averaged over classes.
          - class_to_pq (dict): {class_label: PQ_value} for each class.
    """
    N = len(gt_inst_maps)

    # 1) bPQ, bDQ, bSQ
    bPQ_list, bDQ_list, bSQ_list = [], [], []
    for i in range(N):
        [dq_i, sq_i, pq_i], _ = get_fast_pq(gt_inst_maps[i], pred_inst_maps[i], match_iou)
        bPQ_list.append(pq_i)
        bDQ_list.append(dq_i)
        bSQ_list.append(sq_i)

    bPQ = np.mean(bPQ_list)
    bDQ = np.mean(bDQ_list)
    bSQ = np.mean(bSQ_list)

    # 2) Accumulate raw detection stats for each class across all images
    # We'll sum up (TP, FP, FN) and sum of matched IoUs for matched pairs in each class
    class_TP = {c: 0 for c in all_classes}
    class_FP = {c: 0 for c in all_classes}
    class_FN = {c: 0 for c in all_classes}
    class_IoU_sum = {c: 0.0 for c in all_classes}

    for i in range(N):
        for c in all_classes:
            # Isolate class c in GT
            c_gt_map = isolate_class(gt_inst_maps[i], gt_class_map_dict[i], c)
            c_gt_map, _ = remap_label_and_class_map(c_gt_map, {})

            # Isolate class c in pred
            c_pred_map = isolate_class(pred_inst_maps[i], pred_class_map_dict[i], c)
            c_pred_map, _ = remap_label_and_class_map(c_pred_map, {})

            # Single-class PQ for these maps
            [dq_ic, sq_ic, _pq_ic], [paired_true, paired_pred, unpaired_true, unpaired_pred] = \
                get_fast_pq(c_gt_map, c_pred_map, match_iou)

            tp_ic = len(paired_true)
            fp_ic = len(unpaired_pred)
            fn_ic = len(unpaired_true)
            sum_iou_ic = sq_ic * tp_ic

            class_TP[c] += tp_ic
            class_FP[c] += fp_ic
            class_FN[c] += fn_ic
            class_IoU_sum[c] += sum_iou_ic

    # 3) Compute final PQ for each class from aggregated stats
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
            sq_c = sum_iou / (tp + 1e-6)
            pq_c = dq_c * sq_c
        class_to_pq[c] = pq_c

    # 4) mPQ = average PQ across classes
    mPQ = np.mean(list(class_to_pq.values())) if len(all_classes) > 0 else 0.0

    return bPQ, bDQ, bSQ, mPQ, class_to_pq


def test_compute_bPQ_and_mPQ():
    """
    A simple test using contrived data for 2 small (5x5) images
    with 2 classes to confirm that bPQ and mPQ computations 
    run successfully and produce values in [0,1].
    """
    # -----------
    # Synthetic image 1
    gt_inst_map_1 = np.array([
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 1, 1, 0, 2],
        [0, 3, 3, 3, 0],
        [0, 3, 3, 3, 0],
    ], dtype=np.int32)
    gt_class_map_1 = {1: 1, 2: 2, 3: 1}

    pred_inst_map_1 = np.array([
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 4],
        [0, 1, 1, 4, 4],
        [0, 5, 5, 5, 0],
        [0, 5, 5, 0, 0],
    ], dtype=np.int32)
    pred_class_map_1 = {1: 1, 4: 2, 5: 1}

    # -----------
    # Synthetic image 2
    gt_inst_map_2 = np.array([
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 2],
        [0, 0, 0,10, 2],
        [0, 0,10,10, 2],
    ], dtype=np.int32)
    gt_class_map_2 = {1: 1, 2: 1, 10: 3}

    pred_inst_map_2 = np.array([
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 2],
        [0, 0, 0, 2, 2],
        [0, 0, 0,10, 2],
        [0, 0,10,10,10],
    ], dtype=np.int32)
    pred_class_map_2 = {1: 1, 2: 1, 10: 2}

    # Remap to contiguous IDs
    gt_inst_map_1, gt_class_map_1 = remap_label_and_class_map(gt_inst_map_1, gt_class_map_1)
    pred_inst_map_1, pred_class_map_1 = remap_label_and_class_map(pred_inst_map_1, pred_class_map_1)
    gt_inst_map_2, gt_class_map_2 = remap_label_and_class_map(gt_inst_map_2, gt_class_map_2)
    pred_inst_map_2, pred_class_map_2 = remap_label_and_class_map(pred_inst_map_2, pred_class_map_2)

    # Combine into lists
    gt_inst_maps = [gt_inst_map_1, gt_inst_map_2]
    pred_inst_maps = [pred_inst_map_1, pred_inst_map_2]
    gt_class_map_dict = [gt_class_map_1, gt_class_map_2]
    pred_class_map_dict = [pred_class_map_1, pred_class_map_2]

    # Suppose we have 2 classes total
    all_classes = [1, 2]

    # Compute metrics
    bPQ, bDQ, bSQ, mPQ, pq_per_class = compute_bPQ_and_mPQ(
        gt_inst_maps, pred_inst_maps,
        gt_class_map_dict, pred_class_map_dict,
        all_classes,
        match_iou=0.5
    )

    print("Results on a small synthetic test:")
    print(f"  bPQ (binary PQ across images)  : {bPQ:.4f}")
    print(f"  bDQ (avg detection quality)    : {bDQ:.4f}")
    print(f"  bSQ (avg segmentation quality) : {bSQ:.4f}")
    print(f"  mPQ (multi-class PQ)           : {mPQ:.4f}")
    print("  PQ per class:")
    for c in sorted(pq_per_class.keys()):
        print(f"    Class {c}: {pq_per_class[c]:.4f}")

    # Basic sanity checks
    assert 0.0 <= bPQ <= 1.0, "bPQ should be in [0,1]"
    assert 0.0 <= bDQ <= 1.0, "bDQ should be in [0,1]"
    assert 0.0 <= bSQ <= 1.0, "bSQ should be in [0,1]"
    assert 0.0 <= mPQ <= 1.0, "mPQ should be in [0,1]"
    for c in pq_per_class:
        assert 0.0 <= pq_per_class[c] <= 1.0, f"Class {c} PQ not in [0,1]"

if __name__ == "__main__":
    test_compute_bPQ_and_mPQ()
