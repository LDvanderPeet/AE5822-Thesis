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
    Apply one of eight possible dihedral transformations to a list of tensors.

    Parameters
    ----------
    x : list of torch.Tensor
        A list containing the data tensors.
    k : int
        An integer from 0 to 7 defining the specfic transformation.
        0: Identity (original)
        1: Flip top-bottom
        2: Flip left-right
        3: Rotate 180 deg + flip top-bottom
        4: Flip both
        5: Transpose + flip left-right
        6: Transpose
        7: Full dihedral
    """
    if k in [1, 3, 4, 7]: x = [_x.flip(-2) for _x in x]
    if k in [2, 4, 5, 7]: x = [_x.flip(-1) for _x in x]
    if k in [3, 5, 6, 7]: x = [_x.transpose(-2, -1) for _x in x]
    return x


class ComplexSARDataset(Dataset):
    """
    SAR dataset containing single look complex (SLC) data divided into two polarizations (VV and VH)
    and various subapertures (e.g., 3, 5, 7, 9, and 11).

    Handles loading of Zarr-backed subaperture data, splitting by patch IDs, and interleaving complex components into
    real channels.


    """
    def __init__(self, cfg, split='train', da=False):
        """
        Initializes the dataset, splits IDs, and determines polarization subfolders.

        Parameters
        ----------
        cfg : dict
            Path to config file.
        split : str
            Dataset phase: 'train', 'valid', or 'test'.
        da : bool
           If True, applies random dihedral augmentation during __getitem__.
        """
        self.cfg = cfg
        self.da = da

        self.root_dir = cfg["data"]["root_dir"]
        self.input_indices = cfg["data"]["subaperture_config"]["input_indices"]
        self.output_indices = cfg["data"]["subaperture_config"]["output_indices"]
        self.requested_pols = cfg["data"]["subaperture_config"]["active_polarizations"]

        total_looks = len(self.input_indices) + len(self.output_indices)
        self.sub_folder_name = f"sub_{total_looks}"

        potential_ids = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d.isdigit()
        ])

        self.patch_ids = []
        for pid in potential_ids:
            patch_path = os.path.join(self.root_dir, pid)
            pols_exist = all(os.path.exists(os.path.join(patch_path, p)) for p in self.requested_pols)

            if pols_exist:
                self.patch_ids.append(pid)

        n = len(self.patch_ids)

        if split == 'train':
            self.patch_ids = self.patch_ids[:int(0.8 *  n)]
        elif split == 'valid':
            self.patch_ids = self.patch_ids[int(0.8 * n):int(0.9 * n)]
        elif split == 'test':
            self.patch_ids = self.patch_ids[int(0.9 * n):]
        else:
            raise ValueError(f"Invalid split name: '{split}'. Expected 'train', 'valid', or 'test'.")
        print(f"Split {split}: Found {len(self.patch_ids)} valid patched containing {self.requested_pols}")

    def _normalize(self, x, y):
        """
        Applies unit sphere normalization based on the 99th percentile of magnitude.

        Both input and target are scaled by the same factor derived from the input (x) to preserve relative intensity
        relationship.

        Parameters
        ----------
        x : torch.Tensor
            Input subapertures tensor (channels, H, W).
        y : torch.Tensor
            Output subapertures tensor (channels, H, W).
        """
        real = x[0::2]
        imag = x[1::2]
        mag = torch.sqrt(real**2 + imag**2)

        v_max = torch.quantile(mag, 0.99)

        x_norm = x / v_max
        y_norm = y / v_max

        return torch.clamp(x_norm, -1.0, 1.0), torch.clamp(y_norm, -1.0, 1.0)

    def __len__(self):
        """Returns the total number of patches in the current split."""
        return len(self.patch_ids)

    def _complex_to_channels(self, arr):
        """
        Interleaves complex-valued NumPy arrays into real-valued PyTorch tensors.

        Converts a complex array of shape (N, H, W) into a float32 tensor of shape (2N, H, W).

        Parameters
        ----------
        arr : np.ndarray (complex64)
            Complex data array.
        """
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        real, imag = arr.real.astype(np.float32), arr.imag.astype(np.float32)
        combined = np.stack([real, imag], axis=1).reshape(-1, arr.shape[-2], arr.shape[-1])

        return torch.from_numpy(combined)

    def __getitem__(self, idx):
        """
        Loads a SAR patch, processes polarizations, and applies augmentations.

        Parameters
        ----------
        idx : int
            Index of the patch

        Returns
        -------
        x : torch.Tensor
            Input subapertures.
        y : torch.Tensor
            Output subapertures.
        meta_out : dict
            Metadata including patch_id and incidence_angle.
        """
        patch_id = self.patch_ids[idx]
        patch_path = os.path.join(self.root_dir, patch_id)
        ref_pol = self.requested_pols[0]

        with open(os.path.join(patch_path, ref_pol, "metadata.json"), "r") as f:
            meta = json.load(f)

        x_list, y_list = [], []

        for pol in self.requested_pols:
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

        x, y = self._normalize(x, y)

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
    Helper function to instantiate all three dataset splits from a YAML config.

    Parameters
    ----------
    config_path : str
        Path to the config.yaml file.

    Returns
    -------
    tuple
        (train_ds, val_ds, test_ds) instances of ComplexSARDataset.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_ds = ComplexSARDataset(cfg=cfg, split='train', da=True)
    val_ds = ComplexSARDataset(cfg=cfg, split='valid', da=False)
    test_ds = ComplexSARDataset(cfg=cfg, split='test', da=False)
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    train_ds, val_ds, test_ds = build_complex_datasets_from_config("/shared/home/lvanderpeet/AE5822-Thesis/config.yaml")
    x, y, meta = train_ds[0]
    print(x.shape, y.shape, meta["incidence_angle"], meta["idx"])
