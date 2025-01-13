'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.
'''
# Best practice is to import all packages at the beginning of the code.
import os
import argparse
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from osgeo import gdal
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

# Import helpful functions and classes from pytorch
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from earth_obs_seg.eval.online_eval import online_eval
from earth_obs_seg.utils.utils import set_all_seeds
from earth_obs_seg.utils.utils import lookup_torch_dtype

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--cfg_path', type=str, default='runs/unet/default/config/config.yaml',
                        help='Path to config yaml')
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
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    device = torch.device(torch.device(device_type))
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

    # Create dataset and split it into train and val dataset.
    if not os.path.exists(cfg['path_data']):
        SimpleOxfordPetDataset.download(cfg['path_data'])
    train_set = SimpleOxfordPetDataset(cfg['path_data'], "train")
    val_set = SimpleOxfordPetDataset(cfg['path_data'], "valid")

    loader_args = dict(batch_size=cfg['batch_size'],
                       num_workers=cfg['num_workers'],
                       pin_memory=True,
                       drop_last=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    sample = train_set[0]

    # Optional: Insert code for wandb logging here

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
    if cfg['loss_function'] == 'bcewithlogits':
        assert cfg['out_channels'] == 1, f'bcewithlogitsloss requires out_channels==1, not {cfg["out_channels"]}'
        # Use reduction='mean' to compute loss per img in batch
        criterion = nn.BCEWithLogitsLoss(reduction='mean') 
    elif cfg['loss_function'] == 'dice':
        assert cfg['out_channels'] == 1, f'binary dice loss requires out_channels==1, not {cfg["out_channels"]}'
        criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    else:
        raise NotImplementedError(f'loss_function, {cfg["loss_function"]}, from config.yaml not implemented')

    # Begin training
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_set),
                  desc=f'Epoch {epoch}/{cfg["epochs"]}', unit='img',
                  #disable=(sweep==False) # disable tqdm if printing to log instead of console
                  ) as pbar:
            for i, batch in enumerate(train_loader):
                inputs = batch['image']
                targets = batch['mask']

                # Send batch to GPU
                inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
                targets = targets.to(device=device, dtype=dtype)
                
                # Todo: define a transform that includes normalization, augmentations, etc.
                inputs = inputs / 255.
                
                # Create model predictions
                pred = model(inputs)

                # Compute loss
                loss = criterion(pred, targets)
                
                # Reset and scale gradients
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

                if cfg['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
                    # Set step to a real value between 0 and cfg['epochs']
                    scheduler.step((epoch-1) + (i / len(train_loader)))

                pbar.update(inputs.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'avg loss/img': epoch_loss / float(i+1)})

        # Do online evaluation after every epoch
        val_score = online_eval(model, dataloader=val_loader, criterion=criterion, 
                                     device=device, dtype=dtype, cfg=cfg)
        if cfg['lr_scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_score)

        # Save model
        Path(cfg['path_checkpoints']).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        checkpoint_filename = f'checkpoint_epoch{epoch}.pth'
        torch.save(state_dict, str(Path(cfg['path_checkpoints']) / checkpoint_filename))
        logging.info(f'Checkpoint {epoch} saved!')

        # Optional: insert code to evaluate the model on predictions over the full-scale tif using multiple metrics and log it to wandb

    print("Finished train.py")