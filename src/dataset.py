import os
import random
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from safetensors import safe_open


with open("config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)


def tfm_dihedral(x, k):
    """
    Apply dihedral augmentation to all tensors.
    """
    if k in [1, 3, 4, 7]: x = [_x.flip(-2) for _x in x]
    if k in [2, 4, 5, 7]: x = [_x.flip(-1) for _x in x]
    if k in [3, 5, 6, 7]: x = [_x.transpose(-2, -1) for _x in x]
    return x


def normalize_sar(x, m, M):
    """
    Percentile-based normalization.
    """
    m = torch.Tensor(m)
    M = torch.Tensor(M)
    x = ((x - m) / (M - m))
    return x


class Dataset(torch.utils.data.Dataset):
    """
    SAR dataset containing intensity values.

    Parameters
    ----------
    csv_path : str
        Path to csv file containing file names.
    tensor_dir : str
        Path to tensor directory.
    norm_stats : str
        Path to normalization stats file.
    da : bool, optional
        Whether to apply dihedral augmentation or not (default is False).
        """
    def __init__(self, csv_path, tensor_dir, norm_stats, da=False):
        self.df = pd.read_csv(csv_path)
        self.tensor_dir = tensor_dir
        self.da = da

        pvv, pvh = torch.Tensor(np.load(norm_stats))
        self.m = torch.stack([pvv[0], pvh[0]]).reshape(-1, 1, 1)
        self.M = torch.stack([pvv[1], pvh[1]]).reshape(-1, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns
        -------
        x : torch.Tensor
            Input tensor, shape (2, H, W)
        y : torch.Tensor
            Target tensor, shape (2, H, W)
        meta : dict
            Metadata
        """

        row = self.df.iloc[idx]

        if "file_path" in row:
            file_path = row.file_path
        else:
            file_path = os.path.join(self.tensor_dir, row.file_name)

        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(f.keys()[0])

        tensor = torch.nan_to_num(tensor, 0.0)

        n_subaps = (tensor.shape[0] // 2) - 1
        subap_idx = random.randint(1, n_subaps)

        x = tensor[2 * subap_idx:2 * subap_idx + 2]
        y = tensor[0:2]


        x = torch.clamp(normalize_sar(x, self.m, self.M), 0.0, 1.0)
        y = torch.clamp(normalize_sar(y, self.m, self.M), 0.0, 1.0)

        if self.da:
            x, y = tfm_dihedral([x, y], random.randint(0,7))

        meta = {
            "file_path": file_path,
            "index": idx,
        }

        return x, y, meta

def build_datasets_from_config(config_path: str):
    """
    Convenience helper to build train, test, and validation datasets
    directly from config.yaml.
    """

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root = cfg["data"]["root_dir"]

    train_ds = Dataset(
        csv_path=os.path.join(root, cfg["data"]["train"]["csv"]),
        tensor_dir=os.path.join(root, cfg["data"]["train"]["tensor_dir"]),
        norm_stats=os.path.join(root, cfg["data"]["normalization"]["stats_file"]),
        data_augmentation=True,
    )

    val_ds = Dataset(
        csv_path=os.path.join(root, cfg["data"]["valid"]["csv"]),
        tensor_dir=os.path.join(root, cfg["data"]["valid"]["tensor_dir"]),
        norm_stats=os.path.join(root, cfg["data"]["normalization"]["stats_file"]),
        data_augmentation=True,
    )
    return train_ds, val_ds