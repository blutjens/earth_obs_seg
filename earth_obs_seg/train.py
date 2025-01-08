'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.
'''
# Best practice is to import all packages at the beginning of the code.
import argparse
import logging
import yaml
import torch # this imports pytorch

from pprint import pprint
import segmentation_models_pytorch as smp
from earth_obs_seg.utils.utils import set_all_seeds

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--cfg_path', type=str, default='runs/unet/default/config/config.yaml',
                        help='Path to config yaml')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Import cfg
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))

    # Init cpu or gpu
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    device = torch.device(torch.device(device_type))
    logging.info(f'Using device {device}')

    # Initialize random number generator, s.t., stochastics parts of this program
    #  are the same for every run.
    set_all_seeds(cfg['seed'], device=device.type, 
                  use_deterministic_algorithms=cfg['use_deterministic_algorithms'],
                  warn_only=True)

    print('Default model configuration:')
    pprint(cfg)

    # Optional: Insert code to overwrite config during hyperparameter sweep here

    # Load model
    if cfg['model_key']=='unet_smp':
        # See definition of model_args in the config.yaml
        cfg['model_args'] = {
            'encoder_name': cfg['encoder_name'],
            'encoder_weights': cfg['encoder_weights'],
            'in_channels': cfg['in_channels'],
            'classes': cfg['out_channels'],
        }
        model = smp.Unet(**cfg['model_args'])
    else:
        raise KeyError(f'model key, {cfg["model_key"]}, not found. Double-check config.')    
    
    model = model.to(memory_format=torch.channels_last)

    # Optional: Insert code to load model in case training is interrupted

    model.to(device=device)
