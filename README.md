# earth_obs_seg
Barebone template code for training UNet for segmentation of remote sensing imagery 

## Installation
We recommend installing the project via [conda](https://docs.conda.io/en/latest/).
```
# click 'Fork' -> rename into <your_repo_name>
git clone git@github.com:<username>/<your_repo_name>.git
cd <your_repo_name>
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

## Add original repository as upstream
```
# Register the original repo
git remote set-url upstream git@github.com:blutjens/earth_obs_seg.git
# Download changes in the original repository into your hidden .git folder
git fetch upstream main
# Apply the changes from the original repository into your files
git merge upstream/main
```

## Train a model
```
python earth_obs_seg/train.py --cfg_path runs/unet_smp/demo_run/config/config.yaml
```
