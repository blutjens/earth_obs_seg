# geospatial_unet_pytorch
Barebone template code for training UNet for segmentation of remote sensing imagery 

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
pip install matplotlib
pip install rasterio
# rename geospatial_unet_pytorch folder into <your_repo_name>
# replace geospatial_unet_pytorch in pyproject.toml with <your_repo_name>
pip install -e . # install <your_repo_name> as python module
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