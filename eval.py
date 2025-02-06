import argparse
import os.path as osp
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict

import sys
sys.path.append('../dual_unet')
from dual_unet.utils.distributed import init_distributed_mode, get_rank, is_main_process
from dual_unet.utils.misc import seed_everything
from dual_unet.utils.config import load_config
from dual_unet.datasets import build_dataset, build_loader, compute_class_weights_no_background, compute_class_weights_with_background
from dual_unet.models import build_model
from dual_unet.engine import evaluate_test
from dual_unet.models.losses import DualLoss_combined

import wandb

def test(cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Test the model on the test dataset.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Dictionary of test metrics.
    """
    # Initialize distributed mode
    init_distributed_mode(cfg)
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")

    if 'wandb' not in cfg['experiment']:
        cfg['experiment']['wandb'] = False
    if cfg['experiment']['wandb'] and is_main_process():
        wandb.init(project=cfg['experiment']['project'],
                   name=cfg['experiment']['name'],
                   config=cfg,
                   group=cfg['experiment']['wandb_group'])
    
    # Set seed
    seed = cfg['experiment']['seed'] + get_rank()
    seed_everything(seed)

    # Build test dataset and loader
    test_dataset = build_dataset(cfg, split='test')
    test_loader = build_loader(cfg, test_dataset, split='test')

    #check if the file exists
    if not osp.exists(cfg['training']['ce_weights']):
        print("Computing class weights...")
        ce_weights = compute_class_weights_with_background(train_dataset, cfg['dataset']['train']['num_classes'], background_importance_factor=10).to(device)
        # Save weights
        np.save(cfg['training']['ce_weights'], ce_weights.cpu().numpy())
        print(f"Class weights saved to {cfg['training']['ce_weights']}")
    else:
        print("Loading class weights from file...")
        ce_weights = torch.tensor(np.load(cfg['training']['ce_weights'])).to(device)

    # Build model and criterion
    print("Building model and criterion...")
    model = build_model(cfg)
    criterion = DualLoss_combined(
        ce_weights=ce_weights,
        weight_dice=cfg['training']['weight_dice'],
        weight_dice_b=cfg['training']['weight_dice_b'],
        weight_ce=cfg['training']['weight_ce'],
        weight_mse=cfg['training']['weight_mse']
    )

    model.to(device)
    criterion.to(device)

    # Load checkpoint
    path = osp.join(cfg['experiment']['output_dir'], cfg['experiment']['output_name'])
    ckpt = torch.load(path, map_location='cpu')
    ckpt = ckpt['model'] if 'model' in ckpt else ckpt
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=True)
    print(f"\t # model keys: {len(model.state_dict().keys())}, # checkpoint keys: {len(ckpt.keys())}")
    print(f"\t # missing keys: {len(missing_keys)}, # unexpected keys: {len(unexpected_keys)}")

    # Distributed model
    if cfg['distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg['gpu']])

    # Evaluate
    test_stats = evaluate_test(
        cfg, model, criterion, test_loader, device,
        thresholds=cfg['evaluation']['thresholds'],
        max_pair_distance=cfg['evaluation']['max_pair_distance'],
        output_sufix=cfg['experiment']['name']
    )
    stats = {f'test_{k}': v for k, v in test_stats.items()}
    print(stats)
    if cfg['experiment']['wandb'] and is_main_process():
        wandb.log(stats)
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DualU-Net Evaluation')

    # Config file
    parser.add_argument('--config-file', type=str, required=True, help='Path to the config file')
    # Options to override config
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    test(cfg)
