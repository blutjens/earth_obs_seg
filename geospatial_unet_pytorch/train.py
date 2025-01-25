'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.
'''
# Best practice is to import all packages at the beginning of the code.
import os
import argparse
import logging
import yaml
import copy
import wandb
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from osgeo import gdal
import segmentation_models_pytorch as smp

# Import helpful functions and classes from pytorch
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from geospatial_unet_pytorch.eval.metrics import MaskedLoss
from geospatial_unet_pytorch.eval.online_eval import online_eval
from geospatial_unet_pytorch.utils.utils import set_all_seeds
from geospatial_unet_pytorch.utils.utils import lookup_torch_dtype
# from geospatial_unet_pytorch.dataset.dataset_kelp import KelpDataset
from geospatial_unet_pytorch.dataset.dataset_hrmelt import HRMeltDataset
from geospatial_unet_pytorch.predict import GeoDatasetConvolution
from geospatial_unet_pytorch.dataset.dataset_kelp import create_dataset_splits_with_reserved_site
from geospatial_unet_pytorch.eval.metrics import MaskedDice   
from geospatial_unet_pytorch.predict import predict

def get_args():
    parser = argparse.ArgumentParser(description='Trains the UNet using the config in cfg_path')
    parser.add_argument('--cfg_path', type=str, default='runs/unet_smp/demo_run/config/config.yaml',
                        help='Path to config yaml')
    parser.add_argument('--no_wandb', action='store_true', default=False, help='Disable wandb logs')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Enable gdal error messages (otherwise each CPU raises a warning message at the start of each epoch)
    gdal.UseExceptions()

    # Import cfg
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    dtype = lookup_torch_dtype(cfg['dtype'])

    # Init cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Initialize random number generator, s.t., stochastics parts of this program
    #  are the same for every run.
    set_all_seeds(cfg['seed'], device=device.type, 
                  use_deterministic_algorithms=cfg['use_deterministic_algorithms'],
                  warn_only=True)

    print('Default model configuration:')
    pprint(cfg)

    # Optional: Insert code to overwrite config during hyperparameter sweep here

    # Load model
    if cfg['model_key']=='unet_smp':
        # See definition of model_args in the config.yaml
        cfg['model_args'] = {
            'encoder_name': cfg['encoder_name'],
            'encoder_weights': cfg['encoder_weights'],
            'in_channels': cfg['in_channels'],
            'classes': cfg['out_channels'],
        }
        model = smp.Unet(**cfg['model_args'])
    else:
        raise NotImplementedError(f'model_key, {cfg["model_key"]}, from config.yaml not implemented')

    model = model.to(memory_format=torch.channels_last)

    # Optional: Insert code to load model in case training is interrupted

    # Load model weights into GPU memory
    model.to(device=device)

    # Optionally, insert code here to create split if cfg['split_cfg'] != 'csv'

    # Instantiate train and validation dataset
    train_set = HRMeltDataset(cfg=cfg, split='train', verbose=False)
    val_set = HRMeltDataset(cfg=cfg, split='val', verbose=False)
    
    
    loader_args = dict(batch_size=cfg['batch_size'],
                       num_workers=cfg['num_workers'],
                       pin_memory=True,
                       drop_last=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    if cfg['periodical_eval']:
        logging.info("Periodical evaluation active.")

        loader_args['batch_size'] = cfg['prediction_batch_size']
        loader_args['drop_last'] = False
        predict_set = GeoDatasetConvolution(cfg=cfg,
                                               split='periodical_eval',
                                               verbose=False,
                                               stride=cfg['prediction_stride'])
        predict_loader = DataLoader(predict_set, shuffle=False, **loader_args)

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if cfg['optimizer'] == 'adamW':
        optimizer = optim.AdamW(model.parameters(), 
                            lr=cfg['learning_rate'], betas=(0.9, 0.999), 
                            weight_decay=cfg['weight_decay'], foreach=True)
    else:
        raise NotImplementedError(f'optimizer, {cfg["optimizer"]}, from config.yaml not implemented')

    # Initialize a scheduler that will dynamically adjust the learning rate 
    if cfg['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=cfg['T_0'],
            T_mult=cfg['T_mult'],
            eta_min=cfg['eta_min'],
            verbose=False
        )
    elif cfg['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     patience=cfg['lr_patience'])
    else:
        raise NotImplementedError(f'lr_scheduler, {cfg["lr_scheduler"]}, from config.yaml not implemented')

    # Define loss function
    if cfg['loss_function'] == 'dice':
        criterion = MaskedDice(mode=smp.losses.BINARY_MODE, from_logits=True)
    else:
        criterion = MaskedLoss(cfg['loss_function'])
        raise NotImplementedError(f'loss_function, {cfg["loss_function"]}, from config.yaml not implemented')
    
    if cfg['periodical_eval']:
        metrics_fn = {
            'MaskedL1': MaskedLoss('l1'),
            'MaskedDice': MaskedDice(mode=smp.losses.BINARY_MODE, from_logits=True)
        }

    # (Initialize logging)
    if not args.no_wandb:
        wandb_run = wandb.init(project=cfg['wandb_project_name'],
                               resume='allow', anonymous='must',
                               dir=cfg['path_wandb'])
        wandb_cfg_log = dict()
        for key in cfg:
            if isinstance(cfg[key], list):
                # If hyperparameter is a list enter each entry separately
                for i, entry in enumerate(cfg[key]):
                    wandb_cfg_log[f'{key}_{i}'] = entry
            else:
                wandb_cfg_log[key] = cfg[key]
        wandb_run.config.update(wandb_cfg_log)
        # Log full config file to wandb -> artifacts -> config-file -> Files
        artifact = wandb.Artifact(name="config-file", type="config")
        artifact.add_file(local_path=args.cfg_path, name="config.yaml")
        wandb_run.log_artifact(artifact)  # Logs the cfg to "config.yaml:v0"
    else:
        wandb_run = None

    # Begin training
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_set),
                  desc=f'Epoch {epoch}/{cfg["epochs"]}', unit='img',
                  # disable=(sweep==False) # disable tqdm if printing to log instead of console
                  ) as pbar:
            for i, batch in enumerate(train_loader):
                inputs, targets, targets_mask, meta = batch

                # Send batch to GPU
                inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
                targets = targets.to(device=device, dtype=dtype)
                targets_mask = targets_mask.to(device=device, dtype=dtype)

                # Create model predictions
                pred = model(inputs)

                # Compute loss
                loss = criterion(pred, targets, targets_mask)
                
                # Reset and scale gradients
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

                if cfg['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
                    # Set step to a real value between 0 and cfg['epochs']
                    scheduler.step((epoch-1) + (i / len(train_loader)))

                if not args.no_wandb:
                    # Commit is True, which will increase the global wandb step for every batch.
                    wandb_run.log({
                        'train loss': loss.item(),
                        'learning rate during train': optimizer.param_groups[0]['lr'],
                    }, commit=True)

                pbar.update(inputs.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'avg loss/img': epoch_loss / float(i+1)})

        # Do online evaluation after every epoch
        val_score = online_eval(model, dataloader=val_loader, criterion=criterion, 
                                     device=device, dtype=dtype, cfg=cfg, wandb_run=wandb_run)
        if cfg['lr_scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_score)

        # Optional: insert code to evaluate the model on predictions over the full-scale tif using multiple metrics and log it to wandb
        if (cfg['periodical_eval'] and 
            (epoch-cfg['starting_epoch']) % cfg['epochs_between_periodical_eval'] == 0 and
                epoch >= cfg['starting_epoch']):
            logging.info("Starting extensive evaluation.")

            # Save model; move outside of periodical eval if wanting to save every epoch
            Path(cfg['path_checkpoints']).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            checkpoint_filename = f'checkpoint_epoch{epoch}.pth'
            torch.save(state_dict, str(Path(cfg['path_checkpoints']) / checkpoint_filename))
            logging.info(f'Checkpoint {epoch} saved!')

            add_full_scale_im_to_wandb = ((epoch - cfg['starting_epoch']) // cfg['epochs_between_periodical_eval']) % cfg['period_wandb_full_scale_im'] == 0
            periodical_eval_paths = predict(
                model=model,
                dataloader=predict_loader,
                device=device,
                cfg=cfg,
                compress=True,
                specific_pred_path=cfg['path_periodical_eval'],
                verbose=(epoch==1),
                metrics_fn=metrics_fn, 
                wandb_run=wandb_run,
                add_full_scale_im_to_wandb=add_full_scale_im_to_wandb
            )

            logging.info("Done")

        # Log epoch at the end of the epoch to make sure evalution is logged into the correct epoch
        if not args.no_wandb:
            wandb_run.log({'epoch': epoch}, commit=False)

    print("Finished train.py")