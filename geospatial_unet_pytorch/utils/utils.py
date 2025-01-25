import os
import random
import torch
import numpy as np
from typing import Callable, Dict, List, Any, Union, Tuple, Sequence, Optional

def set_all_seeds(seed, device='cpu',
                  use_deterministic_algorithms=False,
                  warn_only=False):
    """
    sets all seeds. 
    See src: https://github.com/pytorch/pytorch/issues/7068
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        print('in utils.py -> set_all_seeds cuda')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic=True
    # sets, e.g., nn.ConvTranspose2d to deterministic
    torch.use_deterministic_algorithms(mode=use_deterministic_algorithms, warn_only=warn_only)

def lookup_torch_dtype(dtype_name: str) -> Any:
    """
    Returns torch dtype given a string
    """
    if dtype_name == 'float16' or dtype_name == 'half':
        return torch.float16
    elif dtype_name == 'float32' or dtype_name == 'float':
        return torch.float32
    elif dtype_name == 'float64' or dtype_name == 'double':
        return torch.float64
    else:
        raise NotImplementedError('only float32 implemented')
