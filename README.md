# earth_obs_seg
Barebone template code for training UNet for segmentation of remote sensing imagery 

## Installation
We recommend installing the project via [conda](https://docs.conda.io/en/latest/).
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
# rename earth_obs_seg folder into <your_repo_name>
# replace earth_obs_seg in pyproject.toml with <your_repo_name>
pip install -e . # install <your_repo_name> as python module
```

### Test torch and gdal installation:
```
# If the following line returns 'True', the torch -- GPU connection seems to work.
python -c 'import torch; print(torch.cuda.is_available())'
# This line should not throw an error
python -c 'from osgeo import gdal'
```

## Add original repository as upstream
```
# Register the original repo
git remote set-url upstream git@github.com:blutjens/earth_obs_seg.git
# Download changes in the original repository into your hidden .git folder
git fetch upstream main
# Apply the changes from the original repository into your files
git merge upstream/main
# Upload changes to your repository
git push
```

## Train a model
```
python earth_obs_seg/train.py --cfg_path runs/unet_smp/demo_run/config/config.yaml
```