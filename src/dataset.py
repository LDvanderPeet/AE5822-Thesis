import os
import random
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from safetensors import safe_open


def tfm_dihedral(x, k):
    """
    Apply dihedral augmentation to all tensors.
    """
    if k in [1, 3, 4, 7]: x = [_x.flip(-2) for _x in x]
    if k in [2, 4, 5, 7]: x = [_x.flip(-1) for _x in x]
    if k in [3, 5, 6, 7]: x = [_x.transpose(-2, -1) for _x in x]
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
    def __init__(self, csv_path, tensor_dir, norm_stats, cfg, da=False):
        self.df = pd.read_csv(csv_path)
        self.tensor_dir = tensor_dir
        self.da = da

        self.input_indices = cfg["input_indices"]
        self.target_index = cfg["target_index"]

        pvv, pvh = torch.Tensor(np.load(norm_stats))
        self.m_pair = torch.stack([pvv[0], pvh[0]]).reshape(2, 1, 1)
        self.M_pair = torch.stack([pvv[1], pvh[1]]).reshape(2, 1, 1)

    def __len__(self):
        """
        Returns
        -------
        len(df) : int
            length of dataframe.
        """
        return len(self.df)

    def _normalize(self, tensor):
        """
        Dynamically normalizes a SAR tensor, no matter the channels
        """
        n_channels = tensor.shape[0]
        repeats = n_channels // 2

        m = self.m_pair.repeat(repeats, 1, 1)
        M = self.M_pair.repeat(repeats, 1, 1)

        return torch.clamp((tensor - m) / (M - m), 0.0, 1.0)

    def _get_subap_slice(self, tensor, index):
        """
        Helper to grab the two-channel VV/VH slice for a given subap index.
        """
        start = index * 2
        return tensor[start:start + 2, :, :]

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
            fname = os.path.basename(row.file_path)
        else:
            fname = row.file_name

        file_path = os.path.join(self.tensor_dir, fname)

        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(f.keys()[0])

        tensor = torch.nan_to_num(tensor, 0.0)

        x_slices = [self._get_subap_slice(tensor, i) for i in self.input_indices]
        x = torch.cat(x_slices, dim=0)

        y = self._get_subap_slice(tensor, self.target_index)

        x = self._normalize(x)
        y = self._normalize(y)

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
        cfg=cfg["data"]["subaperture_config"],
        da=True,
    )

    val_ds = Dataset(
        csv_path=os.path.join(root, cfg["data"]["valid"]["csv"]),
        tensor_dir=os.path.join(root, cfg["data"]["valid"]["tensor_dir"]),
        norm_stats=os.path.join(root, cfg["data"]["normalization"]["stats_file"]),
        cfg=cfg["data"]["subaperture_config"],
        da=True,
    )
    return train_ds, val_ds

train_ds, val_ds = build_datasets_from_config("/shared/home/lvanderpeet/AE5822-Thesis/config.yaml")
x, y, _ = train_ds[0]
print(x.shape, y.shape)
