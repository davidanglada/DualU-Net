import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
from skimage.morphology import extrema

from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
import torchvision.transforms.v2 as v2


# Distributed, logging, and config utilities
import wandb
from dual_unet.utils.distributed import init_distributed_mode, get_rank, is_main_process
from dual_unet.utils.misc import seed_everything
from dual_unet.utils.config import load_config

# Dataset/model-building utilities
from dual_unet.datasets import build_dataset, build_loader
from dual_unet.models import build_model

# ---------------------------
# Watershed & Visualization Utilities
# ---------------------------
from typing import Tuple
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import label


def _denormalize(
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

def _get_raw_image(img: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy image to a Torch tensor with optional denormalization.

    Args:
        img (np.ndarray): The image array.

    Returns:
        torch.Tensor: The processed image tensor.
    """
    img = _denormalize(img)
    transforms_pipeline = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    img = transforms_pipeline(img)
    img = img.permute(0, 2, 1)  # (H,W,C) -> (C,H,W)
    return img  # (C,H,W) -> (H,W,C)

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

def _perform_watershed(
    pred_mask: np.ndarray,
    pred_centroids: np.ndarray,
    th: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply watershed to refine predicted segmentation with centroid markers.

    Args:
        pred_mask (np.ndarray): Predicted segmentation (C x H x W).
        pred_centroids (np.ndarray): Predicted centroid heatmap (1 x H x W).
        th (float): Threshold for local maxima detection.

    Returns:
        (predicted_centroids, predicted_classes, predicted_mask, cells_mask)
        where:
          predicted_centroids -> shape (N, 2)   (x, y) coordinates
          predicted_classes   -> shape (N,)
          predicted_mask      -> label map (H x W) with predicted classes
          cells_mask          -> binary region mask (H x W)
    """
    # 1) Find local maxima for centroids
    centroid_mask, _ = find_local_maxima(pred_centroids[0], th)

    # 2) Convert to connected markers for watershed
    _, markers = cv2.connectedComponents(
        centroid_mask.astype(np.uint8), 4, ltype=cv2.CV_32S
    )

    # 3) Create a binary mask for the predicted region
    pred_mask_argmax = np.argmax(pred_mask, axis=0).astype(np.uint8)
    cells_mask = (pred_mask_argmax > 0).astype(np.uint8)

    # 4) Distance transform + watershed
    dist_map = distance_transform_edt(cells_mask)
    watershed_result = watershed(-dist_map, markers, mask=cells_mask, compactness=1)

    # 5) Remove boundary pixels to refine instance separation
    contours = np.invert(find_boundaries(watershed_result, mode="outer", background=0))
    watershed_result = watershed_result * contours

    # 6) Build a final predicted mask over classes
    predicted_mask = np.zeros_like(pred_mask_argmax)
    labeled_mask, _ = label(watershed_result, return_num=True)
    predicted_centroids = []
    predicted_classes = []

    for region_id in np.unique(labeled_mask):
        if region_id == 0:
            continue
        region_mask = (labeled_mask == region_id)
        class_in_region = pred_mask_argmax[region_mask]
        majority_class = np.bincount(class_in_region).argmax()
        predicted_mask[region_mask] = majority_class

        coords = np.argwhere(region_mask)
        centroid_yx = coords.mean(axis=0)[::-1]  # (y, x) → reversing gives (x, y)
        predicted_centroids.append((centroid_yx[1], centroid_yx[0]))
        predicted_classes.append(majority_class)

    return (
        np.asarray(predicted_centroids),
        np.asarray(predicted_classes),
        predicted_mask,
        cells_mask
    )

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_inference(
    original_img: np.ndarray,
    pred_mask: np.ndarray,
    pred_centroids: np.ndarray,
    watershed_result: np.ndarray,
    save_path: str = "inference_result.png"
):
    """
    Creates a 2×2 figure showing:
        Top-Left: Original image
        Top-Right: Segmentation (class argmax)
        Bottom-Left: Centroid heatmap
        Bottom-Right: Watershed label map

    Classes are colored as follows:
        0 -> black
        1 -> green
        2 -> red
        3 -> yellow

    Args:
        original_img    (np.ndarray): Original image (H x W) or (H x W x 3).
        pred_mask       (np.ndarray): Segmentation mask (H x W).
        pred_centroids  (np.ndarray): Centroid heatmap (1 x H x W).
        watershed_result(np.ndarray): Label map from watershed (H x W).
        save_path       (str): Where to save the figure.
    """
    # ------------------------
    # Define a discrete color map for classes 0..3
    # ------------------------
    # Index 0 -> black, 1 -> green, 2 -> red, 3 -> yellow
    class_colors = [
        (0, 0, 0),     # black
        (0, 1, 0),     # green
        (1, 0, 0),     # red
        (1, 1, 0)      # yellow
    ]
    cmap = mcolors.ListedColormap(class_colors)
    # Boundaries: [0,1,2,3,4] so that 0..<1->class0, 1..<2->class1, etc.
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4], len(class_colors))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # ------------------------
    # Top-left: Original Image
    # ------------------------
    # If original_img has pixel values outside [0,1], normalize or clip as needed.
    if original_img.ndim == 2:
        # Grayscale: expand dims to interpret as (H,W,1) for plotting
        original_img = np.stack([original_img]*3, axis=-1)
    # If the image is already in [H,W,3], you can optionally rescale/clamp as needed
    
    display_img = _get_raw_image(original_img)
    display_img = np.clip(display_img, 0, 1)
    axes[0, 0].imshow(display_img, interpolation='none')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # ------------------------
    # Top-right: Segmentation
    # ------------------------
    axes[0, 1].imshow(pred_mask, cmap=cmap, norm=norm, interpolation='none')
    axes[0, 1].set_title('Segmentation')
    axes[0, 1].axis('off')

    # ------------------------
    # Bottom-left: Centroids (heatmap)
    # ------------------------
    axes[1, 0].imshow(pred_centroids[0], cmap='viridis', interpolation='none')
    axes[1, 0].set_title('Centroids')
    axes[1, 0].axis('off')

    # ------------------------
    # Bottom-right: Watershed
    # ------------------------
    axes[1, 1].imshow(watershed_result, cmap=cmap, norm=norm, interpolation='none')
    axes[1, 1].set_title('Watershed Result')
    axes[1, 1].axis('off')

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------
# Inference Script (Updated)
# ---------------------------
@torch.no_grad()
def inference_only(cfg: dict, save_visuals: bool = True, threshold: float = 0.15):
    """
    Pure inference entry-point for a dual U-Net (segmentation + counting) model,
    now extended with watershed + visualization.

    Steps:
        1) Distributed setup.
        2) (Optional) init wandb.
        3) Seed for reproducibility.
        4) Build test dataset + loader.
        5) Build model + load checkpoint.
        6) Run forward pass to get segmentation + centroids.
        7) Perform watershed + visualization per sample (optional).

    Args:
        cfg (dict): Configuration dictionary loaded from a config file (YAML/JSON).
        save_visuals (bool): If True, save 2×2 visualizations for each sample.
        threshold (float): Threshold for local maxima detection in `_perform_watershed`.

    Returns:
        list: All predictions from the model (seg logits & centroids) for further processing.
    """
    # 1) Distributed setup
    init_distributed_mode(cfg)
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")

    # 2) (Optional) wandb
    if 'wandb' not in cfg['experiment']:
        cfg['experiment']['wandb'] = False
    if cfg['experiment']['wandb'] and is_main_process():
        wandb.init(
            project=cfg['experiment']['project'],
            name=cfg['experiment']['name'],
            config=cfg,
            group=cfg['experiment'].get('wandb_group', None)
        )

    # 3) Seed everything
    seed = cfg['experiment']['seed'] + get_rank()
    seed_everything(seed)

    # 4) Build test dataset/loader
    test_dataset = build_dataset(cfg, split='test')
    test_loader = build_loader(cfg, test_dataset, split='test')

    # 5) Build model + load checkpoint
    model = build_model(cfg).to(device)
    checkpoint_path = osp.join(cfg['experiment']['output_dir'], cfg['experiment']['output_name'])
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in ckpt:
        ckpt = ckpt['model']
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=True)
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  # model keys: {len(model.state_dict().keys())}, # checkpoint keys: {len(ckpt.keys())}")
    print(f"  # missing keys: {len(missing_keys)}, # unexpected keys: {len(unexpected_keys)}")

    # 6) (If distributed) wrap model
    if cfg['distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg['gpu']])

    model.eval()
    all_predictions = []

    # 7) Run inference loop
    for batch_idx, (samples, _) in enumerate(test_loader):
        samples = samples.to(device)
        outputs = model(samples)

        batch_preds = []
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            # Example model output structure: [segmentation_logits, centroid_heatmaps]
            seg_logits, centroids = outputs
            for i in range(len(seg_logits)):
                # Convert logits to probabilities if needed
                seg_mask = torch.softmax(seg_logits[i], dim=0).cpu().numpy()  # shape: (C,H,W)
                cent_map = centroids[i].cpu().numpy()                         # shape: (1,H,W)

                # Save raw predictions
                pred_dict = {
                    "segmentation_logits": seg_mask,
                    "centroid_heatmap": cent_map
                }
                batch_preds.append(pred_dict)

                # 7A) Perform watershed
                (pred_centroids, pred_classes,
                    predicted_mask, cells_mask) = _perform_watershed(
                        pred_mask=seg_mask,
                        pred_centroids=cent_map,
                        th=0.15
                    )

                # 7B) Prepare original image for visualization
                #    (samples[i] is a tensor of shape [C,H,W])
                original_img = samples[i].cpu().numpy()
                
                # 7C) Argmax segmentation for display
                seg_argmax = np.argmax(seg_mask, axis=0)

                # 7D) Save a figure for each sample
                out_root = "./final_inference/"
                if not os.path.exists(out_root):
                    os.makedirs(out_root)
                out_path = f"{out_root}inference_result_batch{batch_idx}_idx{i}.png"
                visualize_inference(
                    original_img=original_img,
                    pred_mask=seg_argmax,
                    pred_centroids=cent_map,
                    watershed_result=predicted_mask,
                    save_path=out_path
                )

        else:
            # If the model returns a single output or another structure, adapt as needed
            for out in outputs:
                batch_preds.append({"output": out.cpu().numpy()})

        all_predictions.extend(batch_preds)

    # 8) (Optional) log stats
    print(f"Inference complete. Generated {len(all_predictions)} sample predictions.")
    if cfg['experiment']['wandb'] and is_main_process():
        wandb.log({"inference_samples": len(all_predictions)})

    return all_predictions

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference-only script with watershed + visualization.')
    parser.add_argument('--config-file', type=str, default=None, help='Path to config file.')
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line in key=value pairs",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    assert args.config_file is not None, "Please provide a config file via --config-file."

    # Load config
    cfg = load_config(args.config_file)

    # Command-line overrides
    if args.opts is not None:
        for opt in args.opts:
            k, v = opt.split('=')
            # Simple override logic; adapt for booleans/ints if needed
            cfg_keys = k.split('.')
            d = cfg
            for key_part in cfg_keys[:-1]:
                d = d[key_part]
            d[cfg_keys[-1]] = v

    # Run inference
    predictions = inference_only(
        cfg
    )
    # If you want, save them:
    # torch.save(predictions, "inference_results_watershed.pth")
