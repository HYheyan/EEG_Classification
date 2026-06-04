import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(use_cpu: bool = False) -> torch.device:
    if torch.cuda.is_available() and not use_cpu:
        return torch.device("cuda")
    return torch.device("cpu")
