import argparse
import os.path as osp
import sys
import os
import numpy as np

import torch
import torch.nn as nn

from collections import OrderedDict

# Add project path if needed
sys.path.append('../dual_unet')

# Distributed / logging
from dual_unet.utils.distributed import init_distributed_mode, save_on_master, is_main_process, get_rank
from dual_unet.utils.misc import seed_everything
from dual_unet.utils.config import load_config

# Data, Model, Engine
from dual_unet.datasets import (
    build_dataset,
    build_loader,
    compute_class_weights_with_background,
    compute_class_weights_no_background
)
from dual_unet.models import build_model, load_state_dict
from dual_unet.engine import train_one_epoch, evaluate

# Losses
from dual_unet.models.losses import DualLoss_combined

import wandb


def train(cfg: dict):
    """
    Main training loop for the Dual U-Net (segmentation + counting).

    Steps:
      1) Initialize distributed mode (if applicable).
      2) Set up device and (optionally) Weights & Biases.
      3) Seed everything for reproducibility.
      4) Build datasets and loaders for train and val splits.
      5) Build the model and the chosen loss function.
      6) Possibly compute or load class weights for the segmentation portion.
      7) Load any existing checkpoint if needed (resume).
      8) Run the training loop for the configured number of epochs:
         - Train one epoch
         - Step LR scheduler
         - Evaluate on val set at specified intervals
         - Save checkpoints
         - Log results (train & val) to wandb if main process
    """

    # Step 1: Initialize distributed
    torch.backends.cudnn.benchmark = False
    init_distributed_mode(cfg)
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")

    # Step 2: Initialize wandb if needed
    if not cfg['experiment'].get('wandb', False):
        cfg['experiment']['wandb'] = False

    if cfg['experiment']['wandb'] and is_main_process():
        wandb.init(
            project=cfg['experiment']['project'],
            name=cfg['experiment']['name'],
            config=cfg
        )

    # Step 3: Seed
    seed = cfg['experiment']['seed'] + get_rank()
    seed_everything(seed)

    # Step 4: Build train & val datasets/loaders
    train_dataset = build_dataset(cfg, split='train')
    val_dataset = build_dataset(cfg, split='val')
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = build_loader(cfg, train_dataset, split='train')
    val_loader = build_loader(cfg, val_dataset, split='val')
    print("Data loaders created.")

    # Step 5: Build model
    model = build_model(cfg)
    print("Model built.")

    # Step 6: Compute or load class weights
    # We'll store them in `ce_weights`
    ce_weights_path = cfg['training'].get('ce_weights', 'ce_weights.npy')

    if not osp.exists(ce_weights_path):
        # Weighted with background
        ce_weights = compute_class_weights_with_background(
            train_dataset,
            cfg['dataset']['train']['num_classes'],
            background_importance_factor=10
        ).to(device)
        np.save(ce_weights_path, ce_weights.cpu().numpy())
    else:
        # Load existing weights
        ce_weights = torch.tensor(np.load(ce_weights_path)).to(device)

    # Step 7: Build the loss criterion
    # Here we show an example using DualLoss_combined; adjust as needed
    criterion = DualLoss_combined(
        ce_weights=ce_weights,
        weight_dice=cfg['training']['weight_dice'],
        weight_dice_b=cfg['training']['weight_dice_b'],
        weight_ce=cfg['training']['weight_ce'],
        weight_mse=cfg['training']['weight_mse']
    )

    # Move model & criterion to device
    model.to(device)
    criterion.to(device)
    print("Model and criterion prepared.")

    # Step 8: Possibly load an existing checkpoint
    load_state_dict(cfg, model)

    # Step 9: Setup optimizer
    base_lr = cfg['optimizer']['lr_base']
    lr_auto_scale = cfg['optimizer'].get('lr_auto_scale', False)
    lr_scale = 1.0

    if lr_auto_scale:
        # Example scaling rule
        base_batch_size = 8
        actual_batch_size = cfg['loader']['train']['batch_size'] * cfg['world_size']
        lr_scale = actual_batch_size / base_batch_size
        print(f"Auto-scaling LR by factor {lr_scale} (batch_size scaling)")

    param_dicts = [
        {
            "params": model.parameters(),
            "lr": base_lr * lr_scale
        }
    ]
    optimizer = torch.optim.Adam(
        param_dicts,
        lr=base_lr * lr_scale,
        weight_decay=cfg['optimizer']['weight_decay']
    )

    # Step 10: Setup LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg['optimizer']['lr_drop_steps'],
        gamma=cfg['optimizer']['lr_drop_factor']
    )

    # If distributed
    if cfg['distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg['gpu']])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Possibly resume from checkpoint
    curr_epoch = 1
    if cfg['experiment'].get('resume', False):
        output_dir = cfg['experiment']['output_dir']
        output_name = cfg['experiment']['output_name']
        ckpt_path = osp.join(output_dir, output_name)
        if osp.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model_without_ddp.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            curr_epoch = ckpt.get('epoch', 1)
            print(f"Resumed from checkpoint {ckpt_path} at epoch {curr_epoch}.")

    max_epochs = cfg['optimizer']['epochs']
    eval_interval = cfg['evaluation'].get('interval', 10)  # e.g., evaluate every 10 epochs or so

    for epoch in range(curr_epoch, max_epochs + 1):
        print(f"Starting epoch {epoch} / {max_epochs}...")
        if cfg['distributed']:
            # set epoch in sampler for correct shuffling
            train_loader.sampler.set_epoch(epoch)

        # Training
        train_stats = train_one_epoch(
            cfg, model, criterion, train_loader, optimizer, device, epoch
        )
        lr_scheduler.step()

        # Evaluate if needed
        val_stats = {}
        if epoch == 1 or epoch == max_epochs or (epoch % eval_interval == 0):
            print("Evaluating on validation set...")
            val_stats = evaluate(
                model, criterion, val_loader, device,
                thresholds=cfg['evaluation']['thresholds'],
                max_pair_distance=cfg['evaluation']['max_pair_distance']
            )
            print(f"Epoch {epoch} Validation Stats: {val_stats}")

        # Save checkpoint
        output_dir = cfg['experiment'].get('output_dir', None)
        output_name = cfg['experiment'].get('output_name', None)
        if output_dir and output_name and (epoch >= 50) and (epoch % eval_interval == 0):
            # We can choose a metric to track improvements, but here we just save every interval
            ckpt = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }
            ckpt_path = osp.join(output_dir, f"{output_name}_epoch_{epoch}.pth")
            save_on_master(ckpt, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

        # Log to wandb if main process
        if cfg['experiment']['wandb'] and is_main_process():
            log_dict = {}
            log_dict.update({f"train_{k}": v for k, v in train_stats.items()})
            log_dict.update({f"val_{k}": v for k, v in val_stats.items()})
            wandb.log(log_dict)

    # End training
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DualUNet")

    # config-file
    parser.add_argument('--config-file', type=str, default=None, help='Path to config file')
    parser.add_argument(
        "--opts",
        help="Override config options in key=value format",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    assert args.config_file is not None, "Please provide a --config-file path."

    cfg = load_config(args.config_file)

    # Possibly override cfg options from cmd line
    if args.opts is not None:
        for opt in args.opts:
            k, v = opt.split('=')
            # Simple type inference
            if v.isdigit():
                v = int(v)
            elif v.replace('.', '', 1).isdigit():
                v = float(v)
            elif v.lower() in ['true', 'false']:
                v = (v.lower() == 'true')
            # Nested keys
            keys = k.split('.')
            d = cfg
            for key_part in keys[:-1]:
                d = d[key_part]
            d[keys[-1]] = v

    train(cfg)
