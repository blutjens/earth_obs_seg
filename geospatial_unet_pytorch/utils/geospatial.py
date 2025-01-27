import numpy as np
from pathlib import Path
from osgeo import gdal
import torch

def open_cropped_tif(path: str, 
    tile_size: [int,int] = None, 
    offsets: [int, int] = None,
    band_idx: int=1) -> np.ndarray:
    """
    Returns a cropped image from a given .tif. Only loads the
    cropped image into memory; not the full .tif. 

    Input:
        tile_size [height, width]
        offsets [y_offset, x_offset]
        band_idx
    Returns:
        array (height, width, n_channels)
    """
    assert Path(path).is_file(), f'Path to full-scale tif, {path}, not found.'

    ds = gdal.Open(path)
    if tile_size is None:
        array = np.array(ds.GetRasterBand(band_idx).ReadAsArray())
    else:
        if offsets is None:
            offsets, tile_size = get_random_crop(path)

        # Read the cropped area using GDAL
        array = ds.GetRasterBand(band_idx).ReadAsArray(
            offsets[1], offsets[0], tile_size[1], tile_size[0])

    del ds
    return array[...,np.newaxis]

def get_random_crop(path: str, tile_size: [int, int]) -> [int,int]:
    """
    Calculates random offsets within the .tif that's stored
    in path assuming that the tif is rectangular. If tile_size < actual tif 
    size, the full tif is returned.

    Inputs:
        path
        tile_size  [height, width]
    Returns:
        offsets [height, width]
        tile_size  [height, width]
    """
    ds = gdal.Open(path)
    assert ds is not None, f'filepath not found: {path}'

    # Get the size of the large tif
    width = ds.RasterXSize
    height = ds.RasterYSize
    del ds

    # Specify the size of the random crop
    if tile_size is not None:
        crop_height = min(tile_size[0], height)
        crop_width = min(tile_size[1], width)
    else:
        crop_height = height
        crop_width = width

    # Calculate the maximum offset to prevent going out of bounds
    max_y_offset = height - crop_height
    max_x_offset = width - crop_width

    # Generate random offsets for the crop. 
    # Note: using torch.randint instead of np.random for torch
    #  to handle the random seeds.
    if tile_size is None:
        # Set zero offset is the full image should be loaded into memory
        y_offset = 0
        x_offset = 0
    else:
        y_offset = torch.randint(0, max_y_offset, (1,)).cpu().numpy().astype(int)[0].item()
        x_offset = torch.randint(0, max_x_offset, (1,)).cpu().numpy().astype(int)[0].item()
    return [y_offset, x_offset], [crop_height, crop_width]