# geospatial_unet_pytorch
Barebone code for training and evaluating a UNet for segmentation or downscaling of geospatial data including remote sensing, satellite, aerial imagery or weather/climate data. This code works with datasets that are a collection of large-scale tifs. These tifs can contain nans, have different extent, and be quite large (e.g., 10,000px x 6,000px x 8-channels, with ~500MB each). The dataloader dynamically loads small tiles from the full-scale tifs into memory for training. After training, prediction across a new set of large-scale tifs can be created with predict.py that sweeps the model across the full area of the tif.

## Installation
We recommend installing [conda](https://docs.conda.io/en/latest/) and then setting-up the project with the following lines. Installing pytorch, cuda, and gdal is a bit tricky, but the lines below worked on our machines:
```
# click 'use this template' -> set <your_repo_name> -> click 'private'
git clone git@github.com:<username>/<your_repo_name>.git
cd <your_repo_name>
conda create -n <your_repo_name>
conda activate <your_repo_name>
conda install python==3.10
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy>1.0.0 wheel setuptools>=67
pip install jupyter
pip install --find-links=https://girder.github.io/large_image_wheels --no-cache GDAL
pip install -r requirements.txt
pip install -e . # installs geospatial_unet_pytorch as python module
```

### Test pytorch-cuda and gdal installation:
```
# If the following line returns 'True', the torch -- GPU connection seems to work.
python -c 'import torch; print(torch.cuda.is_available())'
# This line should not throw an error
python -c 'from osgeo import gdal'
```

## Train a model
```
python geospatial_unet_pytorch/train.py --cfg_path runs/unet_smp/demo_run/config/config.yaml --no_wandb
# Remove --no_wandb to monitoring training on weights and biases curve
```

## Use the trained model to create predictions 
```
python geospatial_unet_pytorch/predict.py --load 'runs/unet_smp/demo_run/checkpoints/checkpoint_epoch10.pth'
```

## How to add a custom dataset
```
- We recommend to get this repository running using demo_run and dataset_hrmelt before adding a custom dataset
- Then, create a new file, geospatial_unet_pytorch/dataset/dataset_yourdataset.py, similar to the hrmelt dataset
- Edit train.py to use the new dataset class
- Compute mean and standard deviation of each input channel across the dataset in 
   a language of your choosing and insert the values in config.yaml.
- Write a function creates data splits and saves by creating one filepath entry per full-scale tif into as train.csv, val.csv, test.csv, periodical_eval.csv
```

## Optional: Rename the src folder into <your_repo_name>
```
# Rename geospatial_unet_pytorch folder into <your_repo_name>
# Replace geospatial_unet_pytorch in pyproject.toml with <your_repo_name>
# Replace geospatial_unet_pytorch in all .py and .ipynb files with <your_repo_name>
```

## Features and functionality
```
Implemented:
- UNet from segmentation_models.pytorch (smp) library
- Pretrained model weights via integration with timm via smp library
- Logging and monitoring runs via weights and biases
- Tested reproducibility and random seeds
- Training on single GPU
- Parallel batches during train and prediction
- L1 and dice loss on nan-masked targets
- Periodically evaluate the model during training on predictions across the full-scale tif

Not implemented:
- Evaluate predictions of multiple models using one script
- Python typing in all functions
- Multi-GPU
- Integration with pretrained models for geospatial data, e.g., SatCLIP
- Mixed precision
- Other loss functions on nan-masked targets
```