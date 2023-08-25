import random 
import numpy as np 
import os 
import torch 


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True