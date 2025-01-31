import argparse
import os.path as osp
from typing import Dict, Any

import torch
import torch.nn as nn

from collections import OrderedDict

import sys
sys.path.append('../dual_unet')
from dual_unet.utils.distributed import init_distributed_mode, get_rank, is_main_process
from dual_unet.utils.misc import seed_everything
from dual_unet.utils.config import load_config
from dual_unet.datasets import build_dataset, build_loader
from dual_unet.models import build_model
from dual_unet.engine import train_one_epoch, evaluate
from dual_unet.models.losses import DualLoss_combined

import wandb

def train(cfg: Dict[str, Any]) -> None:
    """
    Train the model.

    Args:
        cfg: Configuration dictionary.
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

    # Build datasets and loaders
    train_dataset = build_dataset(cfg, split='train')
    val_dataset = build_dataset(cfg, split='val')
    train_loader = build_loader(cfg, train_dataset, split='train')
    val_loader = build_loader(cfg, val_dataset, split='val')

    # Build model and criterion
    model = build_model(cfg)
    criterion = DualLoss_combined(
        ce_weights=compute_class_weights_with_background(train_dataset),
        weight_dice=cfg['training']['weight_dice'],
        weight_dice_b=cfg['training']['weight_dice_b'],
        weight_ce=cfg['training']['weight_ce'],
        weight_mse=cfg['training']['weight_mse']
    )

    model.to(device)
    criterion.to(device)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'])

    # Load checkpoint if available
    if cfg['experiment']['resume']:
        path = osp.join(cfg['experiment']['output_dir'], cfg['experiment']['output_name'])
        if osp.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"Resumed from checkpoint: {path}")

    # Distributed model
    if cfg['distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg['gpu']])

    # Training loop
    for epoch in range(cfg['optimizer']['epochs']):
        train_stats = train_one_epoch(
            cfg, model, criterion, train_loader, optimizer, device, epoch,
            thresholds=cfg['evaluation']['thresholds'],
            max_pair_distance=cfg['evaluation']['max_pair_distance'],
            max_norm=cfg['optimizer']['max_norm']
        )

        val_stats = evaluate(
            model, criterion, val_loader, device,
            thresholds=cfg['evaluation']['thresholds'],
            max_pair_distance=cfg['evaluation']['max_pair_distance']
        )

        if is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch}
            if cfg['experiment']['wandb']:
                wandb.log(log_stats)

            # Save checkpoint
            save_path = osp.join(cfg['experiment']['output_dir'], f"checkpoint_{epoch}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, save_path)
            print(f"Checkpoint saved: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DualU-Net Training')

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
    train(cfg)