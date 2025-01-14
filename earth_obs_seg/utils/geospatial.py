import numpy as np
from osgeo import gdal
import torch

def open_cropped_tif(path: str, 
    img_size: [int,int] = None, 
    offsets: [int, int] = None) -> np.ndarray:
    """
    Returns a cropped image from a given .tif. Only loads the
    cropped image into memory; not the full .tif. 

    Input:
        img_size [height, width]
        offsets [y_offset, x_offset]
    Returns:
        array (height, width, n_channels)
    """
    ds = gdal.Open(path)
    if img_size is None:
        array = np.array(ds.GetRasterBand(1).ReadAsArray())
    else:
        if offsets is None:
            offsets, img_size = get_random_crop(path)

        # Read the cropped area using GDAL
        array = ds.GetRasterBand(1).ReadAsArray(
            offsets[1], offsets[0], img_size[1], img_size[0])

    del ds
    return array[...,np.newaxis]

def get_random_crop(path: str, img_size: [int, int]) -> [int,int]:
    """
    Calculates random offsets to open image. If img_size < actual image 
    size, the full image is returned.
    Source: edited from ChatGPT

    Inputs:
        path
        img_size  [height, width]
    Returns:
        offsets [height, width]
        img_size  [height, width]
    """
    ds = gdal.Open(path)
    assert ds is not None, f'filepath not found: {path}'

    # Get the size of the image
    width = ds.RasterXSize
    height = ds.RasterYSize
    del ds

    # Specify the size of the random crop
    crop_height = min(img_size[0], height) 
    crop_width = min(img_size[1], width)

    # Calculate the maximum offset to prevent going out of bounds
    max_y_offset = height - crop_height
    max_x_offset = width - crop_width

    # Generate random offsets for the crop. 
    # Note: using torch.randint instead of np.random for torch
    #  to handle the random seeds.
    y_offset = torch.randint(0, max_y_offset+1, (1,)).cpu().numpy().astype(int)[0].item()
    x_offset = torch.randint(0, max_x_offset+1, (1,)).cpu().numpy().astype(int)[0].item()
    return [y_offset, x_offset], [crop_height, crop_width]
