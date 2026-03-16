import os
import random
import yaml
import numpy as np
import json
import zarr
import torch
from torch.utils.data import Dataset
import glob
import h5py



def tfm_dihedral(x, k):
    """Strip-safe augmentations. Only allows flips/rotations that keep the rectangular aspect ratio.

    Parameters
    ----------
    x : list of torch.Tensor
        A list containing the data tensors.
    k : int
        An integer from 0 to 7 defining the specific transformation.
        0: Identity (original)
        1: Flip top-bottom
        2: Flip left-right
        3: Rotate 180 deg
    """
    if k in [1,3]: x = [_x.flip(-2) for _x in x]
    if k in [2,3]: x = [_x.flip(-1) for _x in x]
    return x


def _normalize(x, y, global_max=4257):
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
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    denom = np.log1p(global_max)
    x[0::2] = torch.log1p(x[0::2]) / denom
    y[0::2] = torch.log1p(y[0::2]) / denom

    ## Map [0, 1] to [-1, 1] for diffusion
    x[0::2] = x[0::2] * 2.0 - 1.0
    y[0::2] = y[0::2] * 2.0 - 1.0

    x[1::2] = x[1::2] / np.pi
    y[1::2] = y[1::2] / np.pi

    return torch.clamp(x, -1.0, 1.0), torch.clamp(y, -1.0, 1.0)


def _strip_crop(x, y, target_h, target_w):
    """
    Forces x and y to the exact same target dimensions.

    Parameters
    ----------
    x : torch.Tensor
        The data tensors.
    y : torch.Tensor
        The target tensors.
    target_h : int
        Target height of the SAR image.
    target_w : int
        Target width of the SAR image.
    """
    c, h, w = x.shape

    final_w = (min(target_w, w) // 16) * 16
    start_w = (w - final_w) // 2

    x = x.narrow(2, start_w, final_w)
    y = y.narrow(2, start_w, final_w)

    return x, y


class H5ComplexSARDataset(Dataset):
    """
    Adapted for the Tiled H5 dataset structure.
    Structure: scene_folder/tile_name.h5
    Bands: bands/i_VV, bands/q_VV, bands/i_VV_SA1, etc.
    """

    def __init__(self, cfg, split='train', da=False):
        self.cfg = cfg
        self.da = da
        self.root_dir = cfg["data"]["root_dir"]
        self.requested_pols = cfg["data"]["subaperture_config"]["active_polarizations"]  # e.g., ["vv"]

        # Using recursive glob to find tiles inside scene folders
        self.all_h5_files = sorted(glob.glob(os.path.join(self.root_dir, "**/*.h5"), recursive=True))

        if not self.all_h5_files:
            raise FileNotFoundError(f"No .h5 files found in {self.root_dir}")

        self.split_info = cfg["data"]["split"]
        train_ratio = self.split_info.get("train", 0.8)
        valid_ratio = self.split_info.get("valid", 0.1)

        n = len(self.all_h5_files)
        random.seed(self.cfg.get("seed", 42))
        random.shuffle(self.all_h5_files)

        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))

        if split == 'train':
            self.files = self.all_h5_files[:train_end]
        elif split == 'valid':
            self.files = self.all_h5_files[train_end:valid_end]
        else:
            self.files = self.all_h5_files[valid_end:]

        print(f"Split {split}: {len(self.files)} tiles found.")

    def __len__(self):
        return len(self.files)

    def _get_band_name(self, comp, pol, sa_idx=None):
        """Helper to construct H5 keys like 'bands/i_VV_SA1'"""
        if sa_idx is None:  # Full aperture
            return f"bands/{comp}_{pol}"
        return f"bands/{comp}_{pol}_SA{sa_idx}"

    def __getitem__(self, idx):
        file_path = self.files[idx]

        with h5py.File(file_path, 'r') as f:
            x_list, y_list = [], []

            # Extract requested polarizations (e.g., VV)
            for pol in self.requested_pols:
                # Example: Input is SA1, SA2; Output is Full Aperture (target)
                # Map your config indices to the SA labels in H5
                # Assuming input_indices=[1, 2] means SA1, SA2
                inputs = self.cfg["data"]["subaperture_config"]["input_indices"]
                outputs = self.cfg["data"]["subaperture_config"]["output_indices"]

                # Load Inputs (Sub-apertures)
                for sa in inputs:
                    i_data = f[self._get_band_name('i', pol, sa)][:]
                    q_data = f[self._get_band_name('q', pol, sa)][:]
                    # Combine to (2, H, W) where 0=Mag, 1=Phase (per your normalize logic)
                    complex_arr = i_data + 1j * q_data
                    x_list.append(self._complex_to_channels(complex_arr))

                # Load Targets
                for sa in outputs:
                    # If sa index > max SA, load the full aperture (i_VV / q_VV)
                    # Adjust this logic based on how you define 'Full' in your config
                    suffix = sa if sa <= 3 else None
                    i_data = f[self._get_band_name('i', pol, suffix)][:]
                    q_data = f[self._get_band_name('q', pol, suffix)][:]
                    complex_arr = i_data + 1j * q_data
                    y_list.append(self._complex_to_channels(complex_arr))

            # Combine polarizations
            x = torch.cat(x_list, dim=0)
            y = torch.cat(y_list, dim=0)

            # Metadata extraction from H5 structure
            # Looking at your previous 'ls' - it's under bands/localIncidenceAngle
            incidence_angle = f['bands/localIncidenceAngle'][:].mean()

        # 3. Post-processing (using your original logic)
        if x.shape[-1] > x.shape[-2]:
            x, y = x.transpose(-1, -2), y.transpose(-1, -2)

        x, y = _normalize(x, y)  # Ensure global_max matches these tiles!

        # Standard Crop
        target_h = self.cfg["data"].get("image_height", x.shape[-2])
        target_w = self.cfg["data"].get("image_width", x.shape[-1])
        x, y = _strip_crop(x, y, target_h, target_w)

        if self.da:
            x, y = tfm_dihedral([x, y], random.randint(0, 3))

        meta_out = {
            "idx": idx,
            "incidence_angle": torch.tensor(float(incidence_angle)),
            "patch_id": os.path.basename(file_path)
        }

        return x, y, meta_out

    def _complex_to_channels(self, arr):
        # Your existing logic: Magnitude/Phase conversion
        mag = np.abs(arr).astype(np.float32)
        phase = np.angle(arr).astype(np.float32)
        combined = np.stack([mag, phase], axis=0)  # (2, H, W)
        return torch.from_numpy(combined)

if __name__ == "__main__":
    # import h5py
    #
    # file_path = '/shared/home/lvanderpeet/AE5822-Thesis/dataset/S1A_S3_SLC__1SDV_20160820T171616_20160820T171644_012687_013EFB_B6D5/559U_44R.h5'
    #
    #
    # def print_structure(name, obj):
    #     if isinstance(obj, h5py.Dataset):
    #         print(f"  [Dataset] {name:40} | Shape: {str(obj.shape):15} | Dtype: {obj.dtype}")
    #     elif isinstance(obj, h5py.Group):
    #         print(f"  [Group]   {name}")
    #
    #
    # with h5py.File(file_path, 'r') as f:
    #     print(f"\nStructure of {file_path.split('/')[-1]}:")
    #     print("-" * 80)
    #     f.visititems(print_structure)
    #     print("-" * 80)
    # pass
    import h5py
    import numpy as np

    file_path = '/shared/home/lvanderpeet/AE5822-Thesis/dataset/S1A_S3_SLC__1SDV_20160820T171616_20160820T171644_012687_013EFB_B6D5/559U_44R.h5'

    with h5py.File(file_path, 'r') as f:
        # 1. Inspect VV Polarization
        i_vv = f['bands/i_VV'][:]
        q_vv = f['bands/q_VV'][:]

        # Calculate Magnitude: sqrt(i^2 + q^2)
        mag_vv = np.sqrt(i_vv ** 2 + q_vv ** 2)

        # 2. Inspect an Input Sub-Aperture (SA1)
        i_sa1 = f['bands/i_VV_SA1'][:]
        q_sa1 = f['bands/q_VV_SA1'][:]
        mag_sa1 = np.sqrt(i_sa1 ** 2 + q_sa1 ** 2)

        # 3. Stats for Normalization
        print(f"--- Full VV Stats ---")
        print(f"  Mean Mag: {mag_vv.mean():.4f}")
        print(f"  Max Mag:  {mag_vv.max():.4f}")
        print(f"  99th Percentile: {np.percentile(mag_vv, 99):.4f}")

        print(f"\n--- SA1 VV Stats ---")
        print(f"  Mean Mag: {mag_sa1.mean():.4f}")
        print(f"  99th Percentile: {np.percentile(mag_sa1, 99):.4f}")

        # 4. Angle and Elevation (Context)
        angle = f['bands/localIncidenceAngle'][:]
        elev = f['bands/elevation'][:]
        print(f"\n--- Context ---")
        print(f"  Incidence Angle Range: {angle.min():.2f}° to {angle.max():.2f}°")
        print(f"  Elevation Range: {elev.min():.1f}m to {elev.max():.1f}m")
