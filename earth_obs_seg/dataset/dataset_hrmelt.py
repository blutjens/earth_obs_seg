from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T # Compose, Resize, ToTensor, Normalize

from earth_obs_seg.utils.utils import lookup_torch_dtype
from earth_obs_seg.utils.geospatial import open_cropped_tif
from earth_obs_seg.utils.geospatial import get_random_crop

def get_filepaths_from_csv(path_csv,
    split='train',
    cfg={
        'path_data': './data/hrmelt_sample/',
        'in_keys': ['pmw', 'mar_wa1'],
        'path_pmw': 'PMW/',
        'path_mar_wa1': 'MAR/MARv3.12/WA1/',
        'path_melt': 'SAR/'
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
                        'melt': path),...,Dict()]
    """
    assert 'in_keys' in cfg, 'config.yaml is missing "in_keys" argument'

    # Concatenate keys of all relevant channels
    data_keys = cfg['in_keys'] + ['melt']
    if split == 'deploy':
        data_keys.remove('melt')

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
            dir_key = Path(cfg[f'path_{key}'])
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
             match the filename, e.g., train for 'earth_obs_seg/runs/unet_smp/demo_run/
             config/train.csv'
            verbose bool: If True, will print some verbose outputs

        '''
        self.cfg = cfg
        self.split = split
        self.verbose = verbose

        # Load filenames from <split>.csv, assuming that data split has already been 
        #  created and saved, e.g., in .csv files. This excludes the static channels.
        self.data, self.filenames = get_filepaths_from_csv(
            path_csv=self.cfg[f'path_{split}_split_csv'],
            split=self.split,
            cfg=self.cfg)

        # Dtype of every returned item.
        self.dtype = lookup_torch_dtype(self.cfg['dtype'])
        assert self.dtype == torch.float32, f'Dataset currently only accept float32, but got {self.dtype}'
        
        # Create list of keys to all relevant channels
        self.keys = self.cfg['in_keys'] + self.cfg['in_keys_static'] + ['melt_mask', 'melt']

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

    def __getitem__(self, idx, offsets=None):
        '''
            Returns a single datastack of tiles. The returned tiles come from a large tif 
            where only a crop of size, cfg.img_size, at a random location is loaded. All
            data is converted to float32.

            Args:
                idx int: index into list of tifs in self.data
                offsets [int, int]: offsets of top-left corner of tile of tif
            Returns:
                inputs torch.Tensor(5, h, w): Inputs with stacked channel dimension
            targets torch.Tensor(1, h, w, ): Prediction targets, in this case
                surface meltwater fraction
            targets_mask torch.Tensor(1, h, w): Nan mask in outputs
                with 1. for invalid values and 0. for valid values.
            meta dict(): meta information on return datastack
        '''
        # Get datastack
        data_stack = self.data[idx]

        # Initialize inputs
        n_ch = len(self.cfg['in_keys']) + len(self.cfg['in_keys_static'])
        assert n_ch >= 1, 'Error. config["in_keys*"] need to have at least one key'
        height = self.cfg['img_size'][0]
        width = self.cfg['img_size'][1]
        inputs = torch.empty((n_ch, height, width), dtype=self.dtype)
        ch_idx = 0
        meta = {}
        if 'melt' in data_stack:
            meta['path_melt'] = str(data_stack['melt'])
        meta['channels'] = dict()
        meta['filename'] = self.filenames[idx]

        # Randomly the position of the top-left corner of a tile within the large tif
        if offsets is None:
            sample_key = list(data_stack.keys())[0]
            offsets, self.cfg['img_size'] = get_random_crop(str(data_stack[sample_key]), self.cfg['img_size'])

        ## Add each channel to the inputs
        # Adding PMW channel
        if 'pmw' in self.cfg['in_keys']:
            # Load one tile into memory
            pmw = open_cropped_tif(str(data_stack['pmw']), self.cfg['img_size'], offsets)
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
        if 'mar_wa1' in self.cfg['in_keys']:
            mar_wa1 = open_cropped_tif(str(data_stack['mar_wa1']), self.cfg['img_size'], offsets)
            assert mar_wa1.dtype == np.float32, f'Expected dtype float32, but got {mar_wa1.dtype} for mar_wa1'
            mar_wa1 = torch.from_numpy(mar_wa1.transpose((2, 0, 1))).contiguous()
            mar_wa1 = mar_wa1 / float(self.cfg['max_mar_wa1']) # scale
            if self.cfg['normalize_inputs']:
                mar_wa1 = T.Normalize(mean=self.cfg['mean_mar_wa1'], std=self.cfg['std_mar_wa1'])(mar_wa1) # normalize
            inputs[ch_idx:ch_idx+1,...] = mar_wa1
            meta['channels'][ch_idx] = 'mar_wa1' 
            ch_idx += 1

        # Optionally, add static input channels here

        # Add targets
        melt = open_cropped_tif(str(data_stack['melt']), self.cfg['img_size'], offsets)
        mask_nans = np.ma.masked_invalid(melt).mask # Mask nans from, e.g., overexposure
        melt = np.ma.array(melt, mask=mask_nans).filled(fill_value=0) # Fill masked values with zero.
        melt = melt.astype(np.float32) # note: src is float64, so converting to float32 will remove precision
        melt = torch.from_numpy(melt.transpose((2, 0, 1))).contiguous() # convert to tensor

        # Optionally, scale and normalize targets here

        # Create a mask over which the targets are invalid. Here, we want to mask out all nans pixels
        #  and pixels over open ocean.
        landmask_filepath = Path(self.cfg['path_landmask'])
        landmask = open_cropped_tif(str(landmask_filepath), self.cfg['img_size'], offsets)
        melt_mask = landmask == -1 # Ocean has label -1. Land has 1. We want to mask out the ocean. so we convert -1 -> 1 and 1 -> 0.
        land_mask = melt_mask.copy()
        melt_mask = np.ma.mask_or(mask_nans, melt_mask, shrink=False) # Create union mask of both masks        
        melt_mask = melt_mask.astype(np.float32) # creates memory overhead but that's okay for now
        melt_mask = torch.from_numpy(melt_mask.transpose((2,0,1))).contiguous() # to torch

        # Add landmask as input to ML model. Other parts of the mask cannot be added as they'd use data from the ground-truth target.
        if 'landmask' in self.cfg['in_keys_static']:
            land_mask = land_mask.astype(np.float32)
            land_mask = torch.from_numpy(land_mask.transpose((2,0,1))).contiguous() # to torch
            inputs[ch_idx:ch_idx+1,...] = land_mask
            meta['channels'][ch_idx] = 'landmask' 
            ch_idx += 1

        targets = melt
        targets_mask = melt_mask
        return inputs, targets, targets_mask, meta