"""
Uses the trained model to apply it to across all tifs
 in the dataset and save predictions in a results folder.
"""
import yaml
import argparse
import logging
import os
import random
import sys
import numpy as np
import wandb
import torch
from osgeo import gdal # rasterio in dataloader uses
    # gdal. Need to import gdal to suppress warning msg. 
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from typing import Callable, Dict, List, Any
import torchvision.transforms.functional as F
import segmentation_models_pytorch as smp

from geospatial_unet_pytorch.dataset.dataset_hrmelt import HRMeltDataset
from geospatial_unet_pytorch.eval.online_eval import online_eval
from geospatial_unet_pytorch.utils.utils import lookup_torch_dtype
from geospatial_unet_pytorch.utils.utils import set_all_seeds
from geospatial_unet_pytorch.utils.utils import get_size_of_tif
from geospatial_unet_pytorch.utils.utils import save_tensor_as_tif

class Prediction(object):
    def __init__(self, tif_size, tile_size, offsets_list, device, dtype, erode_size=0):
        """
        SRC: copied from hrmelt; we're trying not to change this.

        Class that holds a running sum of all predictions on the current tif.
        This can used, e.g., to convolve the trained model across the image and to
        create a new full-scale tif.
        """
        height_tif, width_tif = tif_size # size of predicted tif
        self.shape = (1, height_tif, width_tif) # shape of prediction
        self.height_tile = min(tile_size[0], height_tif) # size of each tile
        self.width_tile = min(tile_size[1], width_tif)
        self.device = device # cpu or gpu
        self.dtype = dtype # datatype of prediction
        self.offsets_list = offsets_list # list with offsets of each tile. see Dataset class

        # Declare full-scale .tif prediction
        self.pred_sum = None
        self.counts = None
        self.tile_idx_in_tif_counter = 0

        self.erode_size = erode_size
        # Instantiate full-scale tif predictions
        self.reset_counters()

    def reset_counters(self):
        """
        resets counters for convolution of model across full-scale tif
        """
        # Running sum of predictions in each pixel of the full-scale tif
        self.pred_sum = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        # Running sum of how many predictions were made per pixel
        self.counts = torch.zeros(self.shape, device=self.device, dtype=int)
        # Running sum of how many tiles have been computed in current tif.
        #  indexes next tile in tif; resets after every tif
        self.tile_idx_in_tif_counter = 0 

        return 1
    
    def compute_pred_sum(self, pred):
        """
        This function takes in predicted tiles and adds them to the full sized image array.
        It also keeps track of how many predictions there are for any given pixel to later average them out.
        Args:
            pred torch.Tensor(batch_size, n_ch, h, w) Batch of predicted tiles with different offsets wrt. full-size tif
        """
        # Add each predicted tile onto pred_sum
        # todo: parallelize this
        for j, pred_tile in enumerate(pred):  # (n_ch, h, w)
            # get the offsets to know where tiles should be added
            y_offset, x_offset = self.offsets_list[self.tile_idx_in_tif_counter]

            # get the min and max offset values which are at the beginning and end of the offset list
            y_max, x_max = self.offsets_list[-1]
            y_min, x_min = self.offsets_list[0]

            # define erosion if the tile is not on the edge of the full sized img
            y_erosion_top = self.erode_size
            y_erosion_bottom = -self.erode_size
            x_erosion_left = self.erode_size
            x_erosion_right = -self.erode_size

            # remove erosion if the tile is on a edge of the full sized tif. This will make sure the size
            # of the full image stays the same
            if y_offset == y_min:
                y_erosion_top = 0

            if y_offset == y_max:
                y_erosion_bottom = 0

            if x_offset == x_min:
                x_erosion_left = 0

            if x_offset == x_max:
                x_erosion_right = 0

            # if there is no erosion the added tile should not be eroded on at the end of the axis
            tile_y_bottom = y_erosion_bottom if y_erosion_bottom != 0 else None
            tile_x_right = x_erosion_right if x_erosion_right != 0 else None

            # here all predictions get added to a array that will later make up the full sized prediction
            # 1. access where the tile should be added with erosion decreasing accessed tile size
            # 2. adding pixel values of the predicted tile while removing eroded pixels from the prediction
            self.pred_sum[:,
            y_offset + y_erosion_top:y_offset + self.height_tile + y_erosion_bottom,
            x_offset + x_erosion_left:x_offset + self.width_tile + x_erosion_right] \
                += pred_tile[:, y_erosion_top:tile_y_bottom, x_erosion_left:tile_x_right]

            # this array keeps count of where pixel values have been added
            # to later average out where more than one pixel value has been added
            # accesses the same tile size and offset as above and adds 1 to the selected values
            self.counts[:,
            y_offset + y_erosion_top:y_offset + self.height_tile + y_erosion_bottom,
            x_offset + x_erosion_left:x_offset + self.width_tile + x_erosion_right] += 1

            # keeps count of tiles added to know when a full sized tif has been created
            self.tile_idx_in_tif_counter += 1
        return 1

    def compute_pred_avg(self):
        """
        Calculates pred_sum / counts to get the average prediction
        Returns:
            pred_avg torch.Tensor((self.shape)): average prediction of full-scale tif
        """
        # Current tif ends with tiles in current batch. Only occurs if n_tiles_in_tif is truly divisible by batch_size
        assert not torch.any(self.counts==0), 'There was no predictions for some area of prediction.'

        # Calculate average prediction
        pred_avg = self.pred_sum / self.counts

        return pred_avg

@torch.inference_mode()
def predict(model,
        dataloader,
        device,
        cfg=None,
        compress=False,
        verbose=False,
        specific_pred_path=None,
        metrics_fn=None,
        wandb_run=None,
        add_full_scale_im_to_wandb=True
        ):
    '''
    Uses the model to create and store
    predictions of large-scale tifs. The large-scale tif
    is loaded img-by-img via the dataloader.

    If the dataloader returns targets and metrics_fn is not None
     this function will also compute a set of evaluation metrics.

    Args:
        model torch.nn.Module
        dataloader torch.utils.data.dataloader.DataLoader
        device torch.device: device, e.g., cpu or gpu
        cfg dict: Config file with all hyperparameters.
        compress: this will also save the predictions as compressed png.
        specific_pred_path path: path used to save prediction. If not specified the cfg pred_path is used.
                metrics_fn=None,
        metrics_fn {metric_key1: metric_fn1,
                    ...}: Dictionary with all desired metrics and the 
                    function to compute them
        wandb_run: optional wandb logging object created with wandb.init()
        add_full_scale_im_to_wandb: If True and wandb_run is not None, then we plot the full-scale image to wandb
    Returns:
        a array of paths to the predicted images
    '''
    model.eval()
    dtype = lookup_torch_dtype(cfg['dtype'])

    n_tifs = len(dataloader.dataset.data) # number of tifs in dataset
    n_tiles = len(dataloader.dataset) # number of tiles in dataset
    n_batches = len(dataloader) # number of batches in dataset
    if verbose:
        print('# of tifs: ', len(dataloader.dataset.data))
        print('# of tiles in full dataset: ', n_tiles)

    # keeps track of the metrics if metrics_fn is passed
    if metrics_fn:
        metric_values = {}
        for metrics_key in metrics_fn:
            metric_values[metrics_key] = 0. 
    
    tile_counter = 0 # keep track of the total # of tiles that have been processed
    pred_paths = []
    prediction = None
    with tqdm(total=n_tiles, desc='prediction', unit='tile') as pbar:
        for idx_batch, batch in enumerate(dataloader):
            inputs, _, nan_mask, meta = batch
            batch_size = inputs.shape[0] # batch_size can vary with dataloader.drop_last = False

            # Retrieve the index of the tif to which the tiles belong. Take the idx
            #  of the first tifs in case the batch contains tiles from more than one tif
            tif_idcs_in_batch = dataloader.dataset.idx_tifs[tile_counter:tile_counter+batch_size]
            idx_tif = tif_idcs_in_batch[0]
            tile_idcs_that_belong_to_tif = np.argwhere(dataloader.dataset.idx_tifs==idx_tif).flatten()
            n_tiles_in_tif = len(tile_idcs_that_belong_to_tif)

            assert len(np.unique(tif_idcs_in_batch)) <= 2, f'Batch size of {batch_size}'\
                'too large. The function geospatial_unet_pytorch.predict.predict() assumes that one'\
                'batch should contain tiles from not more than two tifs. Fix the '\
                'issue by reducing the batch_size or passing larger tifs.'

            if prediction is None:
                # Initialize the full-scale .tif prediction
                offsets_of_each_tile_in_tif = dataloader.dataset.offsets_list[tile_idcs_that_belong_to_tif]
                prediction = Prediction(tif_size=dataloader.dataset.tif_sizes[idx_tif],
                                        tile_size=dataloader.dataset.cfg['tile_size'],
                                        offsets_list=offsets_of_each_tile_in_tif, 
                                        device=device, 
                                        dtype=dtype,
                                        erode_size=cfg['erode_size']
                                        )

            inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
            nan_mask = nan_mask.to(device=device, dtype=dtype)
            
            pred = model(inputs)
            
            # Number of tiles in current batch that belong to current prediction tif.
            #  The last batch of the current tif might contain tiles of two tifs. In that case,
            #  n_tiles_in_current_tif will be the number of leftover tiles
            n_tiles_in_batch_of_current_tif = np.count_nonzero(tif_idcs_in_batch==idx_tif)

            # Add the current batch of predictions to the current prediction tif
            prediction.compute_pred_sum(pred[:n_tiles_in_batch_of_current_tif,...])
            
            # If all tiles in tif have successfully been predicted:
            if prediction.tile_idx_in_tif_counter == n_tiles_in_tif:
                # Compute the average prediction for every pixel in the tif.
                pred_avg = prediction.compute_pred_avg()

                # optionally, postprocess here
                # pred_avg = postprocess(pred_avg, cfg)

                # Get the path to a reference tif that contains the metadata for 
                #  our prediction. In this case, we're using the first 
                #  modality in the inputs
                filename = meta['filename'][0]
                reference_tif_path = (Path(cfg['path_data']) /
                    Path(cfg[f'path_{cfg["in_keys"][0]}']) /
                    Path(filename))
                                
                # Define the path where to store the prediction
                if specific_pred_path is not None:
                    new_tif_path = str(Path(specific_pred_path) / Path(filename))
                elif dataloader.dataset.split == 'deploy':
                    new_tif_path = str(Path(cfg['path_deploy']) / Path(filename))
                else:
                    new_tif_path = str(Path(cfg['path_predictions']) / Path(filename))

                if compress:
                    # Save compressed image
                    path = Path(new_tif_path)
                    new_png_path = path.with_suffix('.' + 'png')
                    Path(new_png_path).parent.mkdir(parents=True,exist_ok=True)

                    save_image(pred_avg, str(new_png_path))
                    if verbose:
                        logging.info(f"\nSaved: {new_png_path}")

                # Uncompressed images are stored as .tifs, s.t., evaluation metrics can be computed
                save_tensor_as_tif(pred_avg, tif_path=str(reference_tif_path), 
                                   new_tif_path=str(new_tif_path),
                                   dtype=cfg['dtype'], verbose=verbose)

                # Compute metrics on the full-scale tif
                if metrics_fn is not None:
                    # Load full-scale targets and mask into memory
                    path_targets = str(Path(cfg['path_data']) / Path(cfg['path_targets']) / Path(filename))
                    targets_full_im, nan_mask_full_im = dataloader.dataset.load_targets_and_nan_mask(
                        path_targets=path_targets, tile_size=None, offsets=None)
                    targets_full_im = targets_full_im.to(device=device, dtype=dtype)
                    nan_mask_full_im = nan_mask_full_im.to(device=device, dtype=dtype)
                    for metric_key in metrics_fn.keys():
                        # Add computed metric value to the running sum. This is the loss per valid pixel.
                        metric_values[metric_key] += metrics_fn[metric_key](pred_avg, targets_full_im, nan_mask_full_im)

                # Free memory of the done prediction
                del prediction
                prediction = None

                # If the current batch contains tiles from next tif, we add them to the new prediction:
                if n_tiles_in_batch_of_current_tif < batch_size:
                    idx_tif = tif_idcs_in_batch[-1] # Grabbing the index of the next tif here by passing -1
                    tile_idcs_that_belong_to_tif = np.argwhere(dataloader.dataset.idx_tifs==idx_tif).flatten()
                    offsets_of_each_tile_in_tif = dataloader.dataset.offsets_list[tile_idcs_that_belong_to_tif]
                    prediction = Prediction(tif_size=dataloader.dataset.tif_sizes[idx_tif],
                                        tile_size=dataloader.dataset.cfg['tile_size'],
                                        offsets_list=offsets_of_each_tile_in_tif, 
                                        device=device, 
                                        dtype=dtype,
                                        erode_size=cfg['erode_size']
                                        )
                    prediction.compute_pred_sum(pred[n_tiles_in_batch_of_current_tif:,...])

                pred_paths.append(new_tif_path)
            pbar.update(batch_size)

            tile_counter += batch_size

        # Compute average loss per valid pixel across all tiles and log to wandb 
        if metrics_fn is not None:
            for metric_key in metrics_fn.keys():
                metric_values[metric_key] = metric_values[metric_key] / n_tifs
                logging.info(f'periodical eval {metric_key} / valid px: {metric_values[metric_key]}')
                if wandb_run:
                    wandb_run.log(metric_values, commit=False)

        # Log full-scale images to wandb during selected evaluations
        if (wandb_run and add_full_scale_im_to_wandb):
            pred_paths_png = [path.replace('.tif', '.png') for path in pred_paths]
            wandb_run.log(
                {'periodical_eval': [wandb.Image(path, caption=Path(path).stem)
                                        for path in pred_paths_png]}, commit=False)
        
    return pred_paths

def create_list_of_yx_offsets_in_tif(
    tif_size: [int,int], 
    tile_size: [int,int], 
    stride: int,
    ) -> np.ndarray:
    """
    # Creates a list of y- and x- offsets of each tile in a full-scale
    #  image. This is done to allow loading tiles in batches. y
    Args:
        tif_size: size of the full tif [height,width] in px
        tile_size: size of each crop in the full tif [height,width] in px
        stride: Stride is the number of pixels between every tile's top
            left corner +1. If the combination of tile_size and stride
            would have skipped over the pixels at the boundary, this algorithm
            will add an extra tile at the boundary.
    Returns:
        offsets_list: List of y-, x-offsets of shape (n_tiles_in_tif, 2).
            Iterating over offsets_list will iterate over rows first; 
            then columns. offsets_list[:,0] is y-dim and offsets_list[:,1] is 
            x-dim.
    """
    height_tif, width_tif = tif_size

    # Specify the size of the tile
    height_tile = min(tile_size[0], height_tif) 
    width_tile = min(tile_size[1], width_tif)

    # Calculate the maximum offset to prevent going out of bounds
    max_y_offset = height_tif - height_tile
    max_x_offset = width_tif - width_tile

    # Compute x-, and y-offsets for every tile in the full-scale tif
    #  (we add +1. s.t. e.g., if the tif and tile are size 1, the list of 
    #   offsets is [0])
    y_offsets = np.arange(0, max_y_offset+1, stride)
    if y_offsets[-1] != max_y_offset:
        # Add last row, if the combination of tile_height and stride
        #  would have skipped over the pixels at the boundary
        y_offsets = np.concatenate((y_offsets, np.array([max_y_offset])))

    x_offsets = np.arange(0, max_x_offset+1, stride)
    if x_offsets[-1] != max_x_offset:
        x_offsets = np.concatenate((x_offsets, np.array([max_x_offset])))

    # Create a 2D grid of offsets
    y_grid, x_grid = np.meshgrid(y_offsets, x_offsets)
    # Stack the grid into a 2D array of shape (n, 2). 
    offsets_list = np.block([y_grid.reshape(-1, 1), x_grid.reshape(-1, 1)])

    return offsets_list

class GeoDatasetConvolution(HRMeltDataset):
    def __init__(self, cfg, split='periodical_eval', verbose=False, stride=None):
        '''
            Child class of HRMeltDataset that is used to create pre-
            dictions across a list of many full-scale tifs. These
            tifs can be of varying size. To create predictions across
            all tifs, this class will create a long list that contains
            the top-left corner offsets of every tile within every tif.
            The dataloader will then iterate over that long list. 
        Args:
            cfg, split, verbose: see parent class
            stride int: Stride is the number of pixels+1 between every tile's top-left
             corner that is loaded into memory for prediction. If the stride
             is smaller than the tile_size, each pixel in the predicted 
             image will be a weighted average of all predictions at that
             pixel.
        '''
        # Call the parent class constructor
        super().__init__(cfg=cfg, split=split, verbose=verbose)

        if stride is None:
            if 'erode_size' in cfg:
                stride = cfg['tile_size'][0] - cfg['erode_size']
            else:
                stride = cfg['tile_size'][0]
        else:
            if stride > cfg['tile_size'][0] or stride > cfg['tile_size'][1]:
                raise ValueError(f'Configuration stride of {stride} is larger than tile size of {cfg["tile_size"]}.')
            if 'erode_size' in cfg:
                if stride > (cfg['tile_size'][0] - cfg['erode_size']) or stride > (cfg['tile_size'][1] - cfg['erode_size']):
                    raise ValueError(f'Configuration stride of {stride} is larger than img size, {cfg["tile_size"]}, minus erode_size, {cfg["erode_size"]}.')
            self.cfg['prediction_stride'] = stride

        # Create a list of the position of every tile within every tif
        #  E.g., the 34th entry may contain the position of the 34th tile in the first tif and then 60th entry contains the 20th tile of the 2nd tif.
        self.offsets_list = np.array([], dtype=int).reshape(0,2) # dims: (number of total tiles, 2)
        self.idx_tifs = np.array([], dtype=int) # Indices of the tif that belongs to each tile. dims: (number of total tiles)
        self.tif_sizes = np.zeros((len(self.data),2), dtype=int) # Store size of the tifs to initialize predictions. len: number of tifs
        for idx_tif, data_stack in enumerate(self.data):
            sample_key = cfg['in_keys'][0]
            tifpath = data_stack[sample_key] # Get the path of an input channel
            self.tif_sizes[idx_tif] = get_size_of_tif(str(tifpath))
            offsets_in_tif = create_list_of_yx_offsets_in_tif(
                    self.tif_sizes[idx_tif],
                    tile_size=cfg['tile_size'],
                    stride=stride
                ) # dims: (tiles in tif, 2)
            self.offsets_list = np.concatenate((self.offsets_list, offsets_in_tif), axis=0)
            self.idx_tifs = np.concatenate((self.idx_tifs, idx_tif * np.ones(len(offsets_in_tif), dtype=int)))

    def __len__(self):
        '''
            Returns the length of the dataset, which is the number of tiles
        '''
        n_tiles_in_dataset = len(self.offsets_list)
        return n_tiles_in_dataset
    
    def __getitem__(self, idx):
        """
        Args:
            idx Index into all tiles in dataset
        """
        idx_tif = self.idx_tifs[idx] # Index of the full-scale tif to which the tile belongs

        # Get offset position of one tile from list
        offsets = self.offsets_list[idx]
        offsets = [offsets[0].item(), offsets[1].item()] # convert numpy int to python int

        return super().__getitem__(idx_tif, offsets)

def get_args():
    parser = argparse.ArgumentParser(description='Use a trained model to create full-scale predictions across one data split')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--cfg_path', type=str, default='runs/unet_smp/demo_run/config/config.yaml', help='Path to config yaml')
    parser.add_argument('--data_split', type=str, default='periodical_eval', help='Split [train, val, or test] for which the'\
                         'predictions will be calculated')
    parser.add_argument('--verbose', action='store_true', default=False, help='Set true to print verbose logs')
    parser.add_argument('--prediction_stride', type=int, default=None, help='Overwrite cfg[prediction_stride] argument')
    parser.add_argument('--erode_size', type=int, default=None, help='Overwrite cfg[erode_size] argument')
    parser.add_argument('--prediction_batch_size', type=int, default=None, help='Overwrite cfg[prediction_batch_size] argument')
    parser.add_argument('--num_workers', type=int, default=None, help='Overwrite cfg[num_workers] argument')
    parser.add_argument('--apply_landmask_to_predictions', type=bool, default=None, help='Overwrite cfg[apply_landmask_to_predictions] argument')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Enable gdal error messages (otherwise each CPU raises a warning message at the start of each epoch)
    gdal.UseExceptions()

    # Import cfg and overwrite arguments with command line parameters
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    dtype = lookup_torch_dtype(cfg['dtype'])
    if args.prediction_stride is not None:
        cfg['prediction_stride'] = args.prediction_stride
    if args.erode_size is not None:
        cfg['erode_size'] = args.erode_size
    if args.prediction_batch_size is not None:
        cfg['prediction_batch_size'] = args.prediction_batch_size
    if args.apply_landmask_to_predictions is not None:
        cfg['apply_landmask_to_predictions'] = args.apply_landmask_to_predictions
    if args.num_workers is not None:
        cfg['num_workers'] = args.num_workers

    # Init cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    set_all_seeds(cfg['seed'], device=device.type, 
                  use_deterministic_algorithms=cfg['use_deterministic_algorithms'],
                  warn_only=True)

    if args.verbose:
        print('Default model configuration:')
        pprint(cfg)

    # Initialize dataset from .csv file with filenames
    predict_set = GeoDatasetConvolution(cfg=cfg, 
                                       split=args.data_split,
                                       verbose=False,
                                       stride=cfg['prediction_stride'])

    # Create data loader
    loader_args = dict(batch_size=cfg['prediction_batch_size'], num_workers=cfg['num_workers'], 
                       pin_memory=True)
    predict_loader = DataLoader(predict_set, shuffle=False, drop_last=False, **loader_args)

    # Load model
    if cfg['model_key']=='unet_smp':
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

    if args.load:
        state_dict = torch.load(args.load, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {args.load}')
        except:
            raise ValueError('Error loading model. Verify that hyperparameters in config.yaml '\
                             'match the hyperparameters of the loaded model.')
    model.to(device=device)

    predict_args = {   
        'model' : model,
        'dataloader': predict_loader,
        'device' : device,
        'compress' : True,
        'cfg' : cfg,
        'verbose' : True
    }
    predict(**predict_args)