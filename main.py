import torch
import random
import numpy as np
from train import train
import traceback

def set_seed(seed=12):
    """
    Configure all random number generators to produce reproducible results.

    Parameters
    ----------
    seed : int, optional
        The seed value used to initialize the RNG (default is 12).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Execution of the SAR reconstruction pipeline."""
    set_seed()

    try:
        train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()