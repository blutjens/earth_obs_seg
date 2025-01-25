import os
import random
import torch
import numpy as np
from typing import Callable, Dict, List, Any, Union, Tuple, Sequence, Optional
from osgeo import gdal
import rasterio
from pathlib import Path

def get_size_of_tif(path):
    """
    # Get the size of the tif
    path: str path to full-scale tif. Only used to get dimensions
    """
    ds = gdal.Open(path)
    width_tif = ds.RasterXSize
    height_tif = ds.RasterYSize
    del ds
    return height_tif, width_tif

def set_all_seeds(seed, device='cpu',
                  use_deterministic_algorithms=False,
                  warn_only=False):
    """
    sets all seeds. 
    See src: https://github.com/pytorch/pytorch/issues/7068
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        print('in utils.py -> set_all_seeds cuda')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic=True
    # sets, e.g., nn.ConvTranspose2d to deterministic
    torch.use_deterministic_algorithms(mode=use_deterministic_algorithms, warn_only=warn_only)

def lookup_torch_dtype(dtype_name: str) -> Any:
    """
    Returns torch dtype given a string
    """
    if dtype_name == 'float16' or dtype_name == 'half':
        return torch.float16
    elif dtype_name == 'float32' or dtype_name == 'float':
        return torch.float32
    elif dtype_name == 'float64' or dtype_name == 'double':
        return torch.float64
    else:
        raise NotImplementedError('only float32 implemented')

def save_tensor_as_tif(tensor, tif_path, new_tif_path, 
                       dtype=None,
                       verbose=True):
    """
    Saves a torch tensor as tif with the same metadata from 
    an existing tif.
    Args:
        tensor torch.Tensor: tested with shape (1, height, width)
        tif_path str: Path to .tif file that contains desired metadata
        new_tif_path: Path to new .tif file that will be created
        dtype str: If passed, data type on how to store target
    """
    # Extract the metadata from the existing tif file
    with rasterio.open(tif_path) as src: 
        meta = src.meta

    # overwrite dtype
    if dtype:
        meta['dtype'] = dtype

    # overwrite information on number of output bands
    meta['count'] = tensor.shape[0]

    # Create directory
    Path(new_tif_path).parent.mkdir(parents=True,exist_ok=True)

    # Write the array to a new tif file with the same metadata
    with rasterio.open(new_tif_path, 'w', **meta) as dst: 
        dst.write(tensor.cpu().numpy())
    
    # Save the new tif file
    if verbose:
        print(f"Saved {new_tif_path}")