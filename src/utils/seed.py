"""
Random seed utilities for reproducibility.

Paper requires deterministic results across runs.
"""

import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    
    Fixes seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value (paper uses 42 as default)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"All random seeds set to {seed}")

