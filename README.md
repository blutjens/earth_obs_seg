# earth_obs_seg
Barebone template code for training UNet for segmentation of remote sensing imagery 

## Installation
We recommend installing the project via [conda](https://docs.conda.io/en/latest/).
```
# click 'use this template' -> 'create a new repository' -> use your_repo_name
# replace 'earth_obs_seg' with 'your_repo_name'
git clone git@github.com:<username>/earth_obs_seg.git
cd earth_obs_seg
conda create -n earth_obs_seg python=3.12.8
conda activate earth_obs_seg
conda install -c conda-forge --file requirements.txt
pip install -e . # install 'earth_obs_seg' as python module
```

### Test torch and gdal installation:
```
# If the following line returns 'True', the torch -- GPU connection seems to work.
python -c 'import torch; print(torch.cuda.is_available())'
# If the following line throws an error, try reinstalling gdal with the commands below
python -c 'from osgeo import gdal'
pip uninstall gdal
pip install --find-links=https://girder.github.io/large_image_wheels --no-cache GDAL
```

## Train a model
```
python earth_obs_seg/train.py --cfg_path runs/unet_smp/demo_run/config/config.yaml
```
