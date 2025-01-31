import numpy as np

def get_fast_pq(true, pred, match_iou=0.5):
    """
    Your original single-class PQ function, unchanged.
    Returns [DQ, SQ, PQ], plus pairing info.
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

    pairwise_iou = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

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
            iou = inter / (total - inter) if (total - inter) > 0 else 0
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    # Matching step
    if match_iou >= 0.5:
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        # Convert from 0-based to instance IDs
        paired_true += 1
        paired_pred += 1
    else:
        from scipy.optimize import linear_sum_assignment
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        valid = paired_iou > match_iou
        paired_true = paired_true[valid] + 1
        paired_pred = paired_pred[valid] + 1
        paired_iou = paired_iou[valid]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)

    # Detection Quality
    denominator = (tp + 0.5 * fp + 0.5 * fn)
    dq = tp / denominator if denominator else 0.0
    # Segmentation Quality
    sq = paired_iou.sum() / (tp + 1.0e-6) if tp > 0 else 0.0
    # Panoptic Quality
    pq = dq * sq

    return [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred]

def isolate_class(inst_map, class_map, the_class):
    """
    Zero out all instance IDs in 'inst_map' that do NOT belong to 'the_class'.
    'class_map' is e.g. {instance_id -> class_label}.
    """
    out_map = np.zeros_like(inst_map, dtype=inst_map.dtype)
    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        if class_map[inst_id] == the_class:
            out_map[inst_map == inst_id] = inst_id
    return out_map

def compute_bPQ_and_mPQ(
    gt_inst_maps,
    pred_inst_maps,
    gt_class_map_dict,
    pred_class_map_dict,
    all_classes,
    match_iou=0.5
):
    """
    Computes:
      - bPQ: binary PQ (averaged over images, ignoring classes),
        plus the corresponding bDQ and bSQ (averaged DQ and SQ across images).
      - mPQ: multi-class PQ. We accumulate raw TP/FP/FN/IoU across all images per class,
        then compute PQ_c for each class, and average over classes.
      - pq_per_class: dict mapping each class -> final PQ for that class.

    Returns:
      bPQ, bDQ, bSQ, mPQ, class_to_pq
    """
    N = len(gt_inst_maps)

    # ---------------------------------------------------------------------
    # 1) Compute bPQ, bDQ, bSQ by treating all instances as a single class
    #    => do get_fast_pq per image, then average over images
    # ---------------------------------------------------------------------
    bPQ_list = []
    bDQ_list = []
    bSQ_list = []
    for i in range(N):
        [dq_i, sq_i, pq_i], _ = get_fast_pq(gt_inst_maps[i], pred_inst_maps[i], match_iou)
        bPQ_list.append(pq_i)
        bDQ_list.append(dq_i)
        bSQ_list.append(sq_i)

    bPQ = np.mean(bPQ_list)  # average PQ over images
    bDQ = np.mean(bDQ_list)  # average DQ over images
    bSQ = np.mean(bSQ_list)  # average SQ over images

    # ---------------------------------------------------------------------
    # 2) Accumulate raw detection stats for each class across all images
    #    We'll sum up (TP, FP, FN) and IoU for matched pairs.
    # ---------------------------------------------------------------------
    class_TP = {c: 0 for c in all_classes}
    class_FP = {c: 0 for c in all_classes}
    class_FN = {c: 0 for c in all_classes}
    class_IoU_sum = {c: 0.0 for c in all_classes}

    for i in range(N):
        for c in all_classes:
            # Isolate class c in GT
            c_gt_map = isolate_class(gt_inst_maps[i], gt_class_map_dict[i], c)
            # Remap so instance IDs are contiguous [1..K]
            c_gt_map, _ = remap_label_and_class_map(c_gt_map, {})

            # Isolate class c in prediction
            c_pred_map = isolate_class(pred_inst_maps[i], pred_class_map_dict[i], c)
            c_pred_map, _ = remap_label_and_class_map(c_pred_map, {})

            # Single-class PQ for this (image i, class c)
            [dq_ic, sq_ic, pq_ic], [paired_true, paired_pred, unpaired_true, unpaired_pred] = \
                get_fast_pq(c_gt_map, c_pred_map, match_iou)

            # Convert that to raw counts
            tp_ic = len(paired_true)
            fp_ic = len(unpaired_pred)
            fn_ic = len(unpaired_true)
            # sum of matched IoUs => sq_ic * tp_ic
            sum_iou_ic = sq_ic * tp_ic

            # Accumulate to dataset-level
            class_TP[c]     += tp_ic
            class_FP[c]     += fp_ic
            class_FN[c]     += fn_ic
            class_IoU_sum[c] += sum_iou_ic

    # ---------------------------------------------------------------------
    # 3) Compute final PQ_c for each class c from aggregated TP,FP,FN,IoU
    # ---------------------------------------------------------------------
    class_to_pq = {}
    for c in all_classes:
        tp = class_TP[c]
        fp = class_FP[c]
        fn = class_FN[c]
        sum_iou = class_IoU_sum[c]

        if tp == 0:
            # If we have no true positives for this class,
            # PQ_c is 0 by definition (no matched pairs).
            dq_c = 0.0
            sq_c = 0.0
            pq_c = 0.0
        else:
            dq_c = tp / (tp + 0.5 * fp + 0.5 * fn)
            sq_c = sum_iou / tp  # average IoU across matched pairs
            pq_c = dq_c * sq_c

        class_to_pq[c] = pq_c

    # ---------------------------------------------------------------------
    # 4) mPQ = average PQ across classes
    # ---------------------------------------------------------------------
    mPQ = np.mean(list(class_to_pq.values()))

    return bPQ, bDQ, bSQ, mPQ, class_to_pq

def remap_label_and_class_map(inst_map, class_map):
    """
    inst_map: 2D numpy array of instance IDs.
    class_map: dict {instance_id -> class_label}.

    Returns:
      new_inst_map: same shape as inst_map, but instance IDs are remapped
                    to [1..K], with 0 = background
      new_class_map: dict {new_inst_id -> class_label}, consistent with new_inst_map
    """
    import numpy as np

    # Identify all unique IDs (ignoring 0)
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids != 0]

    new_inst_map = np.zeros_like(inst_map, dtype=np.int32)
    new_class_map = {}

    # Create a map from old ID to new ID
    for new_id, old_id in enumerate(inst_ids, start=1):
        # Remap the pixels in the mask
        new_inst_map[inst_map == old_id] = new_id
        # Remap the entry in the class_map
        if old_id in class_map:  # or handle unknown IDs if needed
            new_class_map[new_id] = class_map[old_id]
        else:
            # If the old_id isn't in class_map, handle it (skip or default)
            # e.g., new_class_map[new_id] = 0 or "unknown"
            pass

    return new_inst_map, new_class_map


def test_compute_bPQ_and_mPQ():
    """
    Simple test with 2 small 'images' and 2 classes.
    We'll contrive data so that:
      - Some instances match partially or perfectly
      - Some are mismatched
    This ensures we test both bPQ and mPQ pipelines and also get per-class PQ.
    """
    # ---------------------------
    # Synthetic 'image' 1 (5x5)
    gt_inst_map_1 = np.array([
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 1, 1, 0, 2],
        [0, 3, 3, 3, 0],
        [0, 3, 3, 3, 0],
    ], dtype=np.int32)
    # classes for each GT instance
    gt_class_map_1 = {1: 1, 2: 2, 3: 1}

    pred_inst_map_1 = np.array([
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 4],
        [0, 1, 1, 4, 4],
        [0, 5, 5, 5, 0],
        [0, 5, 5, 0, 0],
    ], dtype=np.int32)
    # classes for each Pred instance
    pred_class_map_1 = {1: 1, 4: 2, 5: 1}

    # ---------------------------
    # Synthetic 'image' 2 (5x5)
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

    gt_inst_map_1, gt_class_map_1 = remap_label_and_class_map(gt_inst_map_1, gt_class_map_1)
    pred_inst_map_1, pred_class_map_1 = remap_label_and_class_map(pred_inst_map_1, pred_class_map_1)
    gt_inst_map_2, gt_class_map_2 = remap_label_and_class_map(gt_inst_map_2, gt_class_map_2)
    pred_inst_map_2, pred_class_map_2 = remap_label_and_class_map(pred_inst_map_2, pred_class_map_2)


    # Combine into lists (our "dataset" of 2 images)
    gt_inst_maps = [gt_inst_map_1, gt_inst_map_2]
    pred_inst_maps = [pred_inst_map_1, pred_inst_map_2]
    gt_class_map_dict = [gt_class_map_1, gt_class_map_2]
    pred_class_map_dict = [pred_class_map_1, pred_class_map_2]

    # Suppose we have 2 classes in total
    all_classes = [1, 2]

    # Compute both metrics
    bPQ, dPQ, sPQ, mPQ, pq_per_class = compute_bPQ_and_mPQ(
        gt_inst_maps,
        pred_inst_maps,
        gt_class_map_dict,
        pred_class_map_dict,
        all_classes,
        match_iou=0.5
    )

    print("Results on a small synthetic test:")
    print(f"  bPQ (binary PQ across images)  : {bPQ:.4f}")
    print(f"  mPQ (multi-class PQ)           : {mPQ:.4f}")
    print("  PQ per class:")
    for c in sorted(pq_per_class.keys()):
        print(f"    Class {c}: {pq_per_class[c]:.4f}")
    
    # Simple sanity checks
    assert 0.0 <= bPQ <= 1.0, "bPQ should be between 0 and 1"
    assert 0.0 <= mPQ <= 1.0, "mPQ should be between 0 and 1"
    for c in pq_per_class:
        assert 0.0 <= pq_per_class[c] <= 1.0, f"Class {c} PQ not in [0,1]"

if __name__ == "__main__":
    test_compute_bPQ_and_mPQ()
