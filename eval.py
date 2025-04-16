import argparse
import os.path as osp
import sys

import torch
import torch.nn as nn
import numpy as np

# For logging or distributed setup
import wandb

from dual_unet.utils.distributed import init_distributed_mode, get_rank, is_main_process
from dual_unet.utils.misc import seed_everything
from dual_unet.utils.config import load_config
from dual_unet.datasets import build_dataset, build_loader, compute_class_weights_with_background
from dual_unet.models import build_model
from dual_unet.engine import evaluate_test
from dual_unet.models.losses import DualLoss_combined

def test(cfg):
    """
    Test/evaluation entry-point for a dual U-Net (segmentation + counting) model.

    Steps:
        1) Initialize distributed mode if needed.
        2) Optionally initialize Weights & Biases (wandb) for logging (if master process).
        3) Set random seed for reproducibility.
        4) Build the test dataset and loader.
        5) Build the model and a suitable criterion (loss function) – for evaluation logging.
        6) Load the model checkpoint.
        7) (If distributed) wrap the model with DistributedDataParallel.
        8) Evaluate the model on the test set with `evaluate_test`.
        9) Print/log the stats.

    Args:
        cfg (dict): Configuration dictionary loaded from a config file (YAML/JSON).
    """
    # 1) Setup for distributed
    init_distributed_mode(cfg)
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")

    # 2) Optionally initialize wandb
    if 'wandb' not in cfg['experiment']:
        cfg['experiment']['wandb'] = False
    if cfg['experiment']['wandb'] and is_main_process():
        wandb.init(
            project=cfg['experiment']['project'],
            name=cfg['experiment']['name'],
            config=cfg,
            group=cfg['experiment'].get('wandb_group', None)
        )

    # 3) Seed everything for reproducibility
    seed = cfg['experiment']['seed'] + get_rank()
    seed_everything(seed)

    # 4) Build the test dataset and data loader
    test_dataset = build_dataset(cfg, split='test')
    test_loader = build_loader(cfg, test_dataset, split='test')

    # 5) Build the model and criterion
    model = build_model(cfg)
    # Decide the loss function based on config

    ce_weights_path = cfg['training'].get('ce_weights', 'ce_weights.npy')

    if not osp.exists(ce_weights_path):
        # Weighted with background
        ce_weights = compute_class_weights_with_background(
            test_dataset,
            cfg['dataset']['test']['num_classes'],
            background_importance_factor=10
        ).to(device)
        np.save(ce_weights_path, ce_weights.cpu().numpy())
    else:
        # Load existing weights
        ce_weights = torch.tensor(np.load(ce_weights_path)).to(device)

    criterion = DualLoss_combined(
        ce_weights=ce_weights,
        weight_dice=cfg['training']['weight_dice'],
        weight_dice_b=cfg['training']['weight_dice_b'],
        weight_ce=cfg['training']['weight_ce'],
        weight_mse=cfg['training']['weight_mse']
    )

    model.to(device)
    criterion.to(device)

    # 6) Load checkpoint
    checkpoint_path = osp.join(cfg['experiment']['output_dir'], cfg['experiment']['output_name'])
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in ckpt:
        ckpt = ckpt['model']
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=True)
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"\t # model keys: {len(model.state_dict().keys())}, # checkpoint keys: {len(ckpt.keys())}")
    print(f"\t # missing keys: {len(missing_keys)}, # unexpected keys: {len(unexpected_keys)}")

    # 7) (If distributed) wrap model with DistributedDataParallel
    if cfg['distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg['gpu']])

    # 8) Evaluate the model
    test_stats = evaluate_test(
        cfg=cfg,
        model=model,
        criterion=criterion,
        data_loader=test_loader,
        device=device,
        thresholds=cfg['evaluation']['thresholds'],
        max_pair_distance=cfg['evaluation']['max_pair_distance'],
        output_sufix=cfg['experiment']['name']
    )
    # Prepare stats for printing/logging
    stats = {f"test_{k}": v for k, v in test_stats.items()}
    print(stats)

    # 9) Log to wandb if master
    if cfg['experiment']['wandb'] and is_main_process():
        wandb.log(stats)

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test a dual U-Net model.')
    parser.add_argument('--config-file', type=str, default=None, help='Path to config file.')
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line, in key=value pairs",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config_file is not None, "Please provide a config file via --config-file."

    # Load and override config
    cfg = load_config(args.config_file)
    # Additional command-line overrides: 
    # E.g. "key1=value1 key2=value2" => apply to cfg
    if args.opts is not None:
        for opt in args.opts:
            k, v = opt.split('=')
            # This is naive: for more robust approach, parse types carefully
            # E.g., convert strings like "True"/"False" to boolean, 
            # or "42" to int
            cfg_keys = k.split('.')
            # navigate in cfg dict
            d = cfg
            for key_part in cfg_keys[:-1]:
                d = d[key_part]
            d[cfg_keys[-1]] = v

    # Run test
    test(cfg)
