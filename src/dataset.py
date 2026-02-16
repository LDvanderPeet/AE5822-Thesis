import os
import random
import yaml
import numpy as np
import json
import zarr
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def tfm_dihedral(x, k):
    """
    Apply dihedral augmentation to a list of tensors.
    """
    if k in [1, 3, 4, 7]: x = [_x.flip(-2) for _x in x]
    if k in [2, 4, 5, 7]: x = [_x.flip(-1) for _x in x]
    if k in [3, 5, 6, 7]: x = [_x.transpose(-2, -1) for _x in x]
    return x


class ComplexSARDataset(Dataset):
    """
    SAR dataset containing single look complex (SLC) data divided into two polarizations (VV and VH)
    and various subapertures (e.g., 3, 5, 7, 9, and 11).
    Parameters
    ----------
    cfg : str
        path to config file.
    split : str
        type of dataset (default is 'train').
    da : bool
       Whether to apply dihedral augmentation or not (default is False).
    """
    def __init__(self, cfg, split='train', da=False):
        self.cfg = cfg
        self.da = da

        self.root_dir = cfg["data"]["root_dir"]
        self.input_indices = cfg["data"]["subaperture_config"]["input_indices"]
        self.output_indices = cfg["data"]["subaperture_config"]["output_indices"]

        pols_available = ["vh", "vv"]
        self.active_pols = pols_available[:cfg["data"]["subaperture_config"]["polarizations"]]

        total_looks = len(self.input_indices) + len(self.output_indices)
        self.sub_folder_name = f"sub_{total_looks}"

        patch_ids = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d.isdigit()
        ])

        n = len(patch_ids)

        if split == 'train':
            self.patch_ids = patch_ids[:int(0.8 *  n)]
        elif split == 'valid':
            self.patch_ids = patch_ids[int(0.8 * n):int(0.9 * n)]
        elif split == 'test':
            self.patch_ids = patch_ids[int(0.9 * n):]
        else:
            raise ValueError(f"Invalid split name: '{split}'. Expected 'train', 'valid', or 'test'.")

    def __len__(self):
        return len(self.patch_ids)

    def _complex_to_channels(self, arr):
        """
        Input: (N, H, W)
        Output: (2N, H, W) float32 interleaved
        """
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        real, imag = arr.real.astype(np.float32), arr.imag.astype(np.float32)
        combined = np.stack([real, imag], axis=1).reshape(-1, arr.shape[-2], arr.shape[-1])

        return torch.from_numpy(combined)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        patch_path = os.path.join(self.root_dir, patch_id)

        with open(os.path.join(patch_path, "vh", "metadata.json"), "r") as f:
            meta = json.load(f)

        x_list, y_list = [], []

        for pol in self.active_pols:
            zarr_path = os.path.join(patch_path, pol, self.sub_folder_name)

            data_zarr = zarr.open_array(zarr_path, mode='r')
            data = np.array(data_zarr[:]).squeeze().astype(np.complex64)

            x_c = data[[i-1 for i in self.input_indices]]
            y_c = data[[i-1 for i in self.output_indices]]

            x_list.append(self._complex_to_channels(x_c))
            y_list.append(self._complex_to_channels(y_c))

        x, y = torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)

        if x.shape[-1] > x.shape[-2]:
            x = x.transpose(-1, -2)
            y = y.transpose(-1, -2)

        if self.da:
            x, y = tfm_dihedral([x, y], random.randint(0, 7))

        meta_out = {
            "idx": idx,
            "incidence_angle": torch.tensor(float(meta["attributes"]["asset"]["imageAnnotation"]["imageInformation"]["incidenceAngleMidSwath"])),
            "patch_id": patch_id
        }

        return x, y, meta_out


def build_complex_datasets_from_config(config_path: str):
    """
    Convenience helper to build train, test, and validation datasets directly from config.yaml"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_ds = ComplexSARDataset(cfg=cfg, split='train', da=True)
    val_ds = ComplexSARDataset(cfg=cfg, split='valid', da=False)
    test_ds = ComplexSARDataset(cfg=cfg, split='test', da=False)
    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = build_complex_datasets_from_config("/shared/home/lvanderpeet/AE5822-Thesis/config.yaml")
x, y, meta = train_ds[0]
print(x.shape, y.shape, meta["incidence_angle"])
