from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset

def get_filepaths_from_csv(path_csv,
    split='train',
    cfg={
        'path_data': './data/hrmelt_sample/',
        'in_keys': ['pmw', 'mar_wa1'],
        'path_pmw': 'PMW/',
        'path_mar_wa1': 'MAR/MARv3.12/WA1/',
        'path_melt': 'SAR/'
    },
    sort=False):
    """
    Reads in a csv file of image filenames and converts it to a data 
    stack of filepaths. In this case, the image filename is assumed to be a unique
    identifier across all input channels and targets. The csv file is assumed to be
    a list of filenames with the format YYYY_MM_DD.tif, e.g.: 
    2018_05_01.tif
    2018_05_04.tif
    ...
    2020_02_09.tif
        
    Args:
        path_csv str: Path to csv file of train, val, or test split
        split: see HRMeltDataset()
        cfg: see HRMeltDataset()
        sort bool: If True, the returned data will be sorted by timestamp
    Returns:
        data: 
            List[n_samples * Dict(  'in_key1': path,
                        'in_key2': path,
                        ... 
                        'melt': path),...,Dict()]
    """
    assert 'in_keys' in cfg, 'config.yaml is missing "in_keys" argument'

    # Concatenate keys of all relevant channels
    data_keys = cfg['in_keys'] + ['melt']
    if split == 'deploy':
        data_keys.remove('melt')

    # Read the filename from the .csv using pandas
    csv_file = Path(path_csv)
    df = pd.read_csv(csv_file, header=None)
    filenames = df.squeeze('columns')

    # Sort the filenames by timestamp
    if sort:
        filenames = filenames.sort_values(ignore_index=True)

    # Add the filepaths of every variable as 
    data = []
    for filename in filenames:
        filepaths = dict()
        for key in data_keys:
            assert f'path_{key}' in cfg, f'config.yaml is missing {cfg["path_{key}"]} argument'
            dir_key = Path(cfg[f'path_{key}'])
            filepaths[key] = dir_key/Path(filename)
        data.append(filepaths)

    return data, filenames

class HRMeltDataset(Dataset):
    def __init__(self, cfg, split='train', verbose=False):
        '''
            Sample constructor for a dataset inspired by the HRMelt dataset for
            superresolution of surface meltwater observations. The constructor 
            indexes the filepaths to all input .tifs and targets.

        Args:
            cfg dict(): Contains the loaded content from config.yaml
            split str: Specifies which split should be loaded. This argument should
             match the filename, e.g., train for 'earth_obs_seg/runs/unet_smp/demo_run/
             config/train.csv'
            verbose bool: If True, will print some verbose outputs

        '''
        self.cfg = cfg
        self.split = split
        self.verbose = verbose

        # Load filenames from <split>.csv, assuming that data split has already been 
        #  created and saved, e.g., in .csv files. This excludes the static channels.
        self.data, self.filenames = get_filepaths_from_csv(
            path_csv=self.cfg[f'path_{split}_split_csv'],
            split=self.split,
            cfg=self.cfg)

    def create_data_splits(self):
        '''
        This function is intended to create datasplits, without duplicating any data. The
        recommended way for this is to create a train.csv, val.csv, and test.csv that
        contain a list of image filepaths.
        '''
        pass