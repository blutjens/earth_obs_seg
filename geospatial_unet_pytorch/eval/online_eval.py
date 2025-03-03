import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

@torch.inference_mode()
def online_eval(model, dataloader, 
        criterion, device, 
        cfg=None, dtype=torch.float32,
        wandb_run=None):
    '''
    Compute scores on validation dataset during training

    Args:
        model torch.nn.Module : neural network object
        dataloader torch.utils.data.dataloader.DataLoader
        criterion torch.nn.Module: loss function
        device torch.device: device, e.g., cpu or gpu
        cfg dict: Config file with all hyperparameters.
        wandb_run: optional wandb logging object created with wandb.init()
    '''
    model.eval()
    n_val = len(dataloader.dataset)
    total_loss = 0.
    batch_sizes = [] # placeholder. Batch size varies with dataloader.drop_last = False 
    if len(dataloader) == 0.: # Set batch size if val_size == 0
        batch_sizes = [1]
    # Random index of image that will be plotted
    # generates int between 0 (inclusive) and n_val (exclusive)
    plot_img_idx = torch.randint(0, n_val, (1,)) 

    # iterate over the validation set
    with tqdm(total=n_val, desc='validation', unit='tile', leave=False) as pbar2:
        for i, batch in enumerate(dataloader):
            inputs, targets, nan_mask, meta = batch

            batch_sizes.append(inputs.shape[0]) 
            inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
            targets = targets.to(device=device, dtype=dtype)
            nan_mask = nan_mask.to(device=device, dtype=dtype)

            # predict the output
            pred = model(inputs)

            loss = criterion(pred, targets, nan_mask) # average loss per img

            total_loss += loss * batch_sizes[i] # compute total loss by multiplying with number of images in batch
        
            pbar2.update(batch_sizes[i])
            pbar2.set_postfix(**{'val loss/valid_px': loss.cpu().numpy()})

            # Plot one image per epoch to wandb
            if i == np.floor(plot_img_idx / batch_sizes[0]): # if current batch contains the image we'd like to plot
                if wandb_run is not None and cfg['plot_val_img_to_wandb']:
                    import wandb
                    idx_in_batch = (plot_img_idx - i * batch_sizes[0]).cpu().numpy().item() # get the index of the image within the batch
                    # Add all in- and output images as a list of images, s.t., epoch slider moves all the same.
                    log_ims_wandb = [wandb.Image(input_img) for input_img in inputs[idx_in_batch].cpu().numpy()] # add every channel within the image we're plotting individually as grayscale
                    log_ims_wandb.append(wandb.Image(nan_mask[idx_in_batch,0].cpu().numpy()))
                    log_ims_wandb.append(wandb.Image(targets[idx_in_batch].cpu().numpy())) # add target
                    log_ims_wandb.append(wandb.Image(pred[idx_in_batch].cpu().numpy())) # add prediction
                    if 'in_keys' in cfg:
                        in_keys = cfg['in_keys'] + cfg['in_keys_static']
                    else:
                        in_keys = ''
                    wandb_run.log({'inputs('+','.join(in_keys)+'), nan-mask, target, pred': 
                                log_ims_wandb}, commit=False)

    val_score = total_loss / n_val
    logging.info('val loss/valid_px: {}'.format(val_score))

    # Log plots to wandb
    if wandb_run is not None:
        import wandb

        # Optionally log distribution of weights and gradients per layer. This takes ~1sec per log. 
        histograms = {}
        if cfg['plot_weight_histograms_to_wandb']:
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
        
        wandb_run.log({
            'validation loss': val_score,
            **histograms
        }, commit=False)

    model.train()
    return val_score