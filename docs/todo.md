### personally, I like to keep track of my todo's in a .md file like this.
[] find a public dataset that i can use to demonstrate this repo :/
    -> big labeled tif from eie flood? it's just 3-channel
    -> hrmelt -> not public
    -> wetseg -> is it public yet
[] create one notebook that illustrates hrmelt
[x] insert evidence that repo works into README.md
[x] test adding 'dem' to input channels
[x] check in open_cropped_tif if filepath exists
[x] rename targets_mask to nan_mask
[x] rename img_size to tile_size
[x] create pull request in kelpseg
    L51 in online_eval.py should be
        pbar2.set_postfix(**{'val loss/img': loss.cpu().numpy()})
    Typo in L254 of dataset_kelp: 'iin_keys_planet_superdove': ['b1','b2','b3','b4','b5','b6','b7','b8']
[x] copy code from kelpseg
    [x] update installation 
    [] go file by file and call git diff.. testing with hrmelt
        x train.py
        x config.yaml
        predict.py
            implement load_targets_mask and load_targets and integrate into predict.py
        x geospatial.py
        x utils.py
        x online_eval
    [x] dataset
[x] rename earth_obs_seg into geospatial_unet_pytorch
    rename pyproject toml
    rename imports
    rename config
[x] add dtypes into code
[x] normalize data
[x] make GPU installation work ->> I can test with hrmelt environment ->> successfully tested with kevin on jetstream2f
[x] setup git fetch
[x] val loss is not improving. verify that model predictions are improving
    [x] use train_loader instead of val_loader in online_eval() to check that code works
    -> train loss decreases, but val loss does not!
    -> maybe model weights don't improve actually
[x] decide whether to incorporate hrmelt or a torchgeo dataset. vhr10 is detection. which torchgeo dataset is seg. Dies torchgeo have a default dataset that comes with large tifs? torchgeo's datasets inherit from a relatively complicated class. Their RandomGeoSampler assumes that we have a single large tifs that we want to randomly sample from.
    -> could use benin cashew, kenya crop type, inria aerial image labeling, 
    -> but unclear if there's any value using torchgeo dataset structure
    ->> use hrmelt.
[x] incorporate an hrmelt sample
    [x] 3 sample images
    [x] copy dataset step by step
        [x] get_filepaths_from_csv
        [x] write getitem
    [x] test get_filepaths_from_csv using explore_data.ipynb
    [x] test getitem
    [] replace the 'melt' key with 'targets'
    [] change train.py and online_eval.py expected _getitem_
    [] Binarize melt in HRMeltDataset and test with 'dice' loss. 
    [] change OxfordPetDataset _getitem_ return structure
[x] include code on periodical large evaluation on predictions
[x] connect logging to wandb
[x] add comments to code
[x] install with # todo: install with conda install -c conda-forge --file requirements.txt
[x] test installation -> earth_obs_seg seems to work. Can delete earth_obs_seg2
    1st issue: conda installation of gdal doesn't work. It only works with these commands.
    # If the following line throws an error, try reinstalling gdal with the commands below
    python -c 'from osgeo import gdal'
    pip uninstall gdal
    pip install --find-links=https://girder.github.io/large_image_wheels --no-cache GDAL

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
[x] edit pyproject.toml s.t., it only installs the package without requiremnents
# next try -> dont define python version; all w conda
conda env create -f environment.yml # >> installed python 3.13.1; torch 2.5.1; no CUDA; gdal works
# next try -> maybe i can only make python 3.9 work?