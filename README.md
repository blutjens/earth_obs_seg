# geospatial_unet_pytorch
Barebone template code for training UNet for segmentation of remote sensing imagery 

Includes functionality for monitoring runs vai weights and biases. 

## Installation
We recommend installing [conda](https://docs.conda.io/en/latest/) and then setup the project using the following lines:
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

### Test pytorch and gdal installation:
```
# If the following line returns 'True', the torch -- GPU connection seems to work.
python -c 'import torch; print(torch.cuda.is_available())'
# This line should not throw an error
python -c 'from osgeo import gdal'
```

## Train a model
```
python geospatial_unet_pytorch/train.py --cfg_path runs/unet_smp/demo_run/config/config.yaml
```

## How to add a custom dataset
```
- First, we recommend to make this repository run with the demo_run and sample dataset
- Then, create a new python file geospatial_unet_pytorch/dataset/dataset_yourdataset.py
- Edit train.py to use the new dataset class
- Compute mean and standard deviation of each input channel across the dataset in 
   a language of your choosing and insert the values in config.yaml.
- Write a function creates data splits and saves by creating one filepath entry per full-scale tif into as train.csv, val.csv, test.csv, periodical_eval.csv
```

## optional: Rename the src folder into <your_repo_name>
```
# rename geospatial_unet_pytorch folder into <your_repo_name>
# replace geospatial_unet_pytorch in pyproject.toml with <your_repo_name>
# replace geospatial_unet_pytorch in all .py and .ipynb files with <your_repo_name>
```
