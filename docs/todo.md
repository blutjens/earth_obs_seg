### personally, I like to keep track of my todo's in a .md file like this.
[x] test installation -> earth_obs_seg seems to work. Can delete earth_obs_seg2
[x] add dtypes into code
[x] normalize data
[] add comments to code
[] make GPU installation work ->> do later
# did not work
conda create -n earth_obs_seg python=3.12
conda activate earth_obs_seg
conda install pytorch-gpu torchvision
conda install rasterio
conda install gdal
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install segmentation-models-pytorch
python; import torch ->> libcusparse.so.12: undefined symbol
# next try
conda create -n earth_obs_seg python=3.12
conda activate earth_obs_seg
pip install --find-links=https://girder.github.io/large_image_wheels --no-cache GDAL
pip install rasterio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
->> undefined symbol: __nvJitLinkComplete_12_4
# restart the laptop
python; import torch ->> libcusparse.so.12: undefined symbol
# next try -> seems to work 
git clone git@github.com:blutjens/earth_obs_seg.git
cd earth_obs_seg
conda create -n earth_obs_seg python=3.12.8
conda activate earth_obs_seg
conda install -c conda-forge --file requirements.txt
pip install -e . # install 'earth_obs_seg' as python module

