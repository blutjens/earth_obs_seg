import os
import csv
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T # Compose, Resize, ToTensor, Normalize

from osgeo import gdal
from geospatial_unet_pytorch.utils.utils import lookup_torch_dtype
from geospatial_unet_pytorch.utils.geospatial import open_cropped_tif
from geospatial_unet_pytorch.utils.geospatial import get_random_crop

def get_crop_with_min_targets(
    path_targets: str, tile_size: [int, int], 
    min_targets_percentage: float = 0., max_n_resampling: int = 10
) -> ([int, int], [int, int]):
    """
    Samples a random crop offset within a large tif (defined by the boundaries of path_targets),
    loads the cropped targets mask into memory, checks if the percentage of targets is at least
    min_targets_percentage, and continues sampling until it finds a valid crop or exceeds
    max_n_resampling attempts.

    Args:
        path_targets: Path to the targets GeoTIFF file.
        tile_size: [height, width] of the desired crop size.
        min_targets_percentage: Minimum percentage of targets required in the crop (0-1). Pass 0. for sampling every tile.
        max_n_resampling: Maximum number of resampling attempts.
    Returns:
        offsets: [height, width] offsets for the crop.
        tile_size: [height, width] of the actual crop.
    """
    for _ in range(max_n_resampling):
        # Get random crop offsets and size
        offsets, crop_size = get_random_crop(path_targets, tile_size)

        # Load the crop from the target mask
        targets = open_cropped_tif(path=path_targets,
                                   tile_size=tile_size,
                                   offsets=offsets,
                                   band_idx=1)
        
        # Check if the crop contains sufficient positive target pixels
        total_pixels = crop_size[0] * crop_size[1]
        targets_pixels = np.sum(targets == 1)  # Assuming the positive targets class is represented by non-zero values
        targets_percentage = float(targets_pixels) / float(total_pixels)
        if targets_percentage >= min_targets_percentage:
            break
    # Return the crop if it meets the targets requirement OR is the final attempt
    return offsets, crop_size

def get_filepaths_from_csv(path_csv,
    split='train',
    cfg={
        'path_data': './data/hrmelt_sample/',
        'in_keys': ['pmw', 'mar_wa1'],
        'path_pmw': 'PMW/',
        'path_mar_wa1': 'MAR/MARv3.12/WA1/',
        'path_targets': 'SAR/'
    },
    sort=False):
    """
    Reads in a csv file of image filenames and converts it to a data 
    stack of filepaths. In this case, the image filename is assumed to be a unique
    identifier across all input channels and targets. The csv file is assumed to be
    a list of filenames with the format YYYY_MM_DD.tif, e.g.: 
    2018_05_01.tif
    2018_05_04.tif
    ...
    2020_02_09.tif
        
    Args:
        path_csv str: Path to csv file of train, val, or test split
        split: see HRMeltDataset()
        cfg: see HRMeltDataset()
        sort bool: If True, the returned data will be sorted by timestamp
    Returns:
        data: 
            List[n_samples * Dict(  'in_key1': path,
                        'in_key2': path,
                        ... 
                        'targets': path),...,Dict()]
    """
    assert 'in_keys' in cfg, 'config.yaml is missing "in_keys" argument'

    # Concatenate keys of all relevant channels
    data_keys = cfg['in_keys'] + ['targets']
    if split == 'deploy':
        data_keys.remove('targets')

    # Read the filename from the .csv using pandas
    csv_file = Path(path_csv)
    df = pd.read_csv(csv_file, header=None)
    filenames = df.squeeze('columns')

    # Sort the filenames by timestamp
    if sort:
        filenames = filenames.sort_values(ignore_index=True)

    # Add the filepaths of every variable as 
    data = []
    for filename in filenames:
        filepaths = dict()
        for key in data_keys:
            assert f'path_{key}' in cfg, f'config.yaml is missing {cfg["path_{key}"]} argument'
            dir_key = Path(cfg['path_data']) / Path(cfg[f'path_{key}'])
            filepaths[key] = dir_key/Path(filename)
        data.append(filepaths)

    return data, filenames

class HRMeltDataset(Dataset):
    def __init__(self, cfg, split='train', verbose=False):
        '''
            Sample constructor for a dataset inspired by the HRMelt dataset for
            superresolution of surface meltwater observations. The constructor 
            indexes the filepaths to all input .tifs and targets.

        Args:
            cfg dict(): Contains the loaded content from config.yaml
            split str: Specifies which split should be loaded. This argument should
             match the filename, e.g., train for 'geospatial_unet_pytorch/runs/unet_smp/demo_run/
             config/train.csv'
            verbose bool: If True, will print some verbose outputs

        '''
        self.cfg = cfg
        self.split = split
        self.verbose = verbose

        # Load filenames from <split>.csv, assuming that data split has already been 
        #  created and saved, e.g., in .csv files. This excludes the static channels.
        self.data, self.filenames = get_filepaths_from_csv(
            path_csv=os.path.join(self.cfg['path_repo'],self.cfg[f'path_{split}_split_csv']),
            split=self.split,
            cfg=self.cfg)

        # Dtype of every returned item.
        self.dtype = lookup_torch_dtype(self.cfg['dtype'])
        assert self.dtype == torch.float32, f'Dataset currently only accept float32, but got {self.dtype}'
        
        # Create list of keys to all relevant channels
        self.keys = self.cfg['in_keys'] + self.cfg['in_keys_static'] + ['nan_mask', 'targets']

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    def create_data_splits(self):
        '''
        This function is intended to create datasplits, without duplicating any data. The
        recommended way for this is to create a train.csv, val.csv, and test.csv that
        contain a list of image filepaths.
        '''
        pass

    def load_targets_and_nan_mask(self, path_targets, tile_size=None, offsets=None):
        '''
        Load one tile of the targets and the associated nan_mask
        '''

        targets = open_cropped_tif(path_targets, tile_size=tile_size, offsets=offsets)
        invalid_px_in_targets = np.ma.masked_invalid(targets).mask # NaNs in the targets, e.g., due to overexposure
        targets = np.ma.array(targets, mask=invalid_px_in_targets).filled(fill_value=0) # Fill masked values with zero.
        targets = targets.astype(np.float32) # note: src is float64, so converting to float32 will remove precision
        targets = torch.from_numpy(targets.transpose((2, 0, 1))).contiguous() # convert to tensor

        # Create a mask over which the targets are invalid. Here, we want to mask out all pixels
        #  that are NaNs in the target geotiff and all pixels over open ocean
        landmask_filepath = Path(self.cfg['path_data'])/Path(self.cfg['path_landmask'])
        landmask = open_cropped_tif(str(landmask_filepath), tile_size=tile_size, offsets=offsets)
        nan_mask = landmask == -1 # Ocean has label -1. Land has 1. We want to mask out the ocean. so we convert -1 -> 1 and 1 -> 0.
        nan_mask = np.ma.mask_or(invalid_px_in_targets, nan_mask, shrink=False) # Create union mask of both masks        
        nan_mask = nan_mask.astype(np.float32) # creates memory overhead but that's okay for now
        nan_mask = torch.from_numpy(nan_mask.transpose((2,0,1))).contiguous() # to torch

        return targets, nan_mask

    def __getitem__(self, idx, offsets=None):
        '''
            Returns a single datastack of tiles. The returned tiles come from a large tif 
            where only a crop of size, cfg.tile_size, at a random location is loaded. All
            data is converted to float32.

            Args:
                idx int: index into list of tifs in self.data
                offsets [int, int]: offsets of top-left corner of tile of tif
            Returns:
                inputs torch.Tensor(5, h, w): Inputs with stacked channel dimension
            targets torch.Tensor(1, h, w, ): Prediction targets, in this case
                surface meltwater fraction
            nan_mask torch.Tensor(1, h, w): Nan mask in outputs
                with 1. for invalid values and 0. for valid values.
            meta dict(): meta information on return datastack
        '''
        # Get datastack
        data_stack = self.data[idx]

        # Randomly the position of the top-left corner of a tile within the large tif
        if offsets is None:
            if 'targets' in data_stack.keys():
                offsets, self.cfg['tile_size'] = get_crop_with_min_targets(
                    path_targets=str(data_stack['targets']), 
                    tile_size=self.cfg['tile_size'], 
                    min_targets_percentage=self.cfg['min_targets_percentage'], 
                    max_n_resampling=self.cfg['max_n_resampling_of_cropped_tif'])
            else:
                sample_key = list(data_stack.keys())[0]
                offsets, self.cfg['tile_size'] = get_random_crop(str(data_stack[sample_key]), self.cfg['tile_size'])

        # Initialize inputs
        n_ch = len(self.cfg['in_keys']) + len(self.cfg['in_keys_static'])
        assert n_ch >= 1, 'Error. config["in_keys*"] need to have at least one key'
        height = self.cfg['tile_size'][0]
        width = self.cfg['tile_size'][1]
        inputs = torch.empty((n_ch, height, width), dtype=self.dtype)
        ch_idx = 0
        meta = {}
        if 'targets' in data_stack:
            meta['path_targets'] = str(data_stack['targets'])
        meta['channels'] = dict()
        meta['filename'] = self.filenames[idx]

        ###
        # All code below would likely need to be edited for a custom dataset
        ###
        
        ## Add each channel to the inputs
        # Adding PMW channel
        if 'pmw' in data_stack:
            # Load one tile into memory
            pmw = open_cropped_tif(str(data_stack['pmw']), self.cfg['tile_size'], offsets)
            assert pmw.dtype == np.float32, f'Expected dtype float32, but got {pmw.dtype} for pmw'
            
            # Convert from numpy array to torch Tensor
            pmw = torch.from_numpy(pmw.transpose((2, 0, 1))).contiguous() # Pytorch uses channels-first: (c, h, w)
            
            ## Crop, scale, and normalize inputs. 
            # In this case, we apply no scaling, because PMW does not have a defined lower/upper limit.
            if self.cfg['normalize_inputs']:
                pmw = T.Normalize(mean=self.cfg['mean_pmw'], std=self.cfg['std_pmw'])(pmw)
            
            ## Apply data augmentations
            # In this case, we add GaussianBlur to smooth out the sharp edges in the incoming image.
            #  If they're not smoothed out, I have seem them introduce edge artifacts.
            pmw = T.GaussianBlur(
                kernel_size=self.cfg['pmw_GaussianBlur_kernel_size'],
                sigma=self.cfg['pmw_GaussianBlur_sigma'])(pmw)

            inputs[ch_idx:ch_idx+1,...] = pmw
            meta['channels'][ch_idx] = 'pmw' 
            ch_idx += 1
        
        # Adding MAR channel
        if 'mar_wa1' in data_stack:
            mar_wa1 = open_cropped_tif(str(data_stack['mar_wa1']), self.cfg['tile_size'], offsets)
            assert mar_wa1.dtype == np.float32, f'Expected dtype float32, but got {mar_wa1.dtype} for mar_wa1'
            mar_wa1 = torch.from_numpy(mar_wa1.transpose((2, 0, 1))).contiguous()
            mar_wa1 = mar_wa1 / float(self.cfg['max_mar_wa1']) # scale
            if self.cfg['normalize_inputs']:
                mar_wa1 = T.Normalize(mean=self.cfg['mean_mar_wa1'], std=self.cfg['std_mar_wa1'])(mar_wa1) # normalize
            inputs[ch_idx:ch_idx+1,...] = mar_wa1
            meta['channels'][ch_idx] = 'mar_wa1' 
            ch_idx += 1

        # Adding static input channel DEM
        if 'dem' in data_stack:
            dem_filepath = Path(self.cfg['path_data'])/Path(self.cfg['path_dem'])
            dem = open_cropped_tif(str(dem_filepath), self.cfg['tile_size'], offsets)
            dem = dem.astype(np.float32)
            dem = torch.from_numpy(dem.transpose((2, 0, 1))).contiguous() # Pytorch uses channels-first: (c, h, w)
            if self.cfg['normalize_inputs']:
                dem = T.Normalize(mean=self.cfg['mean_dem'], std=self.cfg['std_dem'])(dem)
            inputs[ch_idx:ch_idx+1,...] = dem
            meta['channels'][ch_idx] = 'dem' 
            ch_idx += 1

        # Add targets
        if 'targets' in data_stack:
            targets, nan_mask = self.load_targets_and_nan_mask(
                path_targets=str(data_stack['targets']), 
                tile_size=self.cfg['tile_size'], offsets=offsets)
        else:
            # Targets are empty, e.g., during deployment, because no ground-truth wetland segmentations are available
            targets = torch.zeros((1, height, width), dtype=self.dtype).contiguous()
            nan_mask = torch.zeros((1, height, width), dtype=self.dtype).contiguous() # All pixels valid.
        
        return inputs, targets, nan_mask, meta