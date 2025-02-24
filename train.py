import argparse
import os.path as osp
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict

import sys
sys.path.append('../dual_unet')
from dual_unet.utils.distributed import init_distributed_mode, get_rank, is_main_process, save_on_master
from dual_unet.utils.misc import seed_everything
from dual_unet.utils.config import load_config
from dual_unet.datasets import build_dataset, build_loader, compute_class_weights_no_background, compute_class_weights_with_background
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
    print("Initializing distributed mode...")
    init_distributed_mode(cfg)
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if 'wandb' not in cfg['experiment']:
        cfg['experiment']['wandb'] = False
    if cfg['experiment']['wandb'] and is_main_process():
        print("Initializing Weights and Biases...")
        wandb.init(project=cfg['experiment']['project'],
                   name=cfg['experiment']['name'],
                   config=cfg,
                   group=cfg['experiment']['wandb_group'])
    
    # Set seed
    print("Setting seed...")
    seed = cfg['experiment']['seed'] + get_rank()
    seed_everything(seed)

    # Build datasets and loaders
    print("Building datasets and loaders...")
    train_dataset = build_dataset(cfg, split='train')
    val_dataset = build_dataset(cfg, split='val')
    train_loader = build_loader(cfg, train_dataset, split='train')
    val_loader = build_loader(cfg, val_dataset, split='val')

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

    # Build optimizer
    print("Building optimizer...")

    lr_scale = 1.0
    if cfg['optimizer']['lr_auto_scale']:
        base_batch_size   = 8
        actual_batch_size = cfg['loader']['train']['batch_size'] * cfg['world_size']
        lr_scale = actual_batch_size / base_batch_size

    param_dicts = [ 
        {"params": model.parameters(),
         "lr": cfg['optimizer']['lr_base'] * lr_scale}
    ]
 
    optimizer = torch.optim.Adam(param_dicts, 
                                  lr=cfg['optimizer']['lr_base'] * lr_scale,
                                  weight_decay=cfg['optimizer']['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        cfg['optimizer']['lr_drop_steps'],
                                                        cfg['optimizer']['lr_drop_factor'])

    # Load checkpoint if available
    if cfg['experiment']['resume']:
        path = osp.join(cfg['experiment']['output_dir'], cfg['experiment']['output_name'])
        if osp.exists(path):
            print(f"Loading checkpoint from {path}...")
            ckpt = torch.load(path, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"Resumed from checkpoint: {path}")

    # Distributed model
    if cfg['distributed']:
        print("Using DistributedDataParallel...")
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg['gpu']])

        model_without_ddp = model.module

    # Training loop
    for epoch in range(cfg['optimizer']['epochs']):
        print(f"Starting epoch {epoch}...")
        # set epoch in sampler
        if cfg['distributed']:
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            cfg, model, criterion, train_loader, optimizer, device, epoch,
            max_pair_distance=cfg['evaluation']['max_pair_distance']
        )
        lr_scheduler.step()
        
        val_stats = dict()
        if epoch in [1, cfg['optimizer']['epochs']] or epoch % cfg['evaluation']['interval']==0:
            print("Evaluating")
            val_stats = evaluate(
                cfg, model, criterion, val_loader, device,
                  thresholds=cfg['evaluation']['thresholds'],
                  max_pair_distance=cfg['evaluation']['max_pair_distance'])
            print(f"Epoch {epoch}: TRAIN {train_stats}")
            print(f"Epoch {epoch}: VAL {val_stats}")

        

        if 'output_dir' in cfg['experiment'] and\
              'output_name' in cfg['experiment'] and epoch % cfg['evaluation']['interval']==0: # and val_stats['f']['dice'] > max_dice:
            
            ckpt = {
                    'model' : model_without_ddp.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler.state_dict(),
                    'epoch' : epoch
                }    
            ckpt_path = osp.join(cfg['experiment']['output_dir'], 
                                f"{cfg['experiment']['output_name']}_epoch_{epoch}.pth")
            save_on_master(ckpt, ckpt_path)
                        
        if 'wandb' in cfg['experiment'] and is_main_process():
            train_stats = {'train_'+k:v for k,v in train_stats.items()}
            val_stats = {'val_'+k:v for k,v in val_stats.items()}
            wandb.log({**train_stats, **val_stats})
    
    if 'wandb' in cfg['experiment'] and is_main_process():
        wandb.finish()
    
    print("Training completed.")

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

    print("Loading configuration...")
    cfg = load_config(args.config_file)
    print("Starting training...")
    train(cfg)