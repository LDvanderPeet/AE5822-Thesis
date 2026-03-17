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

    ## TODO: check normalization for phase information
    x[1::2] = x[1::2] / np.pi
    y[1::2] = y[1::2] / np.pi

    return torch.clamp(x, -1.0, 1.0), torch.clamp(y, -1.0, 1.0)


def _strip_crop(x, y, target_h=96, target_w=96):
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

    start_h = max(0, (h - target_h) // 2)
    start_w = max(0, (w - target_w) // 2)

    x = x.narrow(1, start_h, target_h).narrow(2, start_w, target_w)
    y = y.narrow(1, start_h, target_h).narrow(2, start_w, target_w)

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
        self.requested_pols = [p.upper() for p in cfg["data"]["subaperture_config"]["active_polarizations"]]  # ["VV", "VH"]

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
        """Standardizes: bands/i_VV or bands/i_VV_SA1"""
        if sa_idx is None:  # Full aperture
            return f"bands/{comp}_{pol.upper()}"
        return f"bands/{comp}_{pol.upper()}_SA{sa_idx}"

    def __getitem__(self, idx):
        file_path = self.files[idx]

        with h5py.File(file_path, 'r') as f:
            x_list, y_list = [], []
            # Example: Input is SA1, SA2; Output is Full Aperture (target)
            # Map your config indices to the SA labels in H5
            # Assuming input_indices=[1, 2] means SA1, SA2
            inputs = self.cfg["data"]["subaperture_config"]["input_indices"]
            outputs = self.cfg["data"]["subaperture_config"]["output_indices"]

            for pol in self.requested_pols:
                # Load Inputs (Sub-apertures)
                for sa in inputs:
                    i_key = self._get_band_name('i', pol, sa)
                    q_key = self._get_band_name('q', pol, sa)
                    complex_arr = f[i_key][:] + 1j * f[q_key][:]
                    x_list.append(self._complex_to_channels(complex_arr))

                for sa in outputs:
                    # Logic: if sa is 0 or high, load full aperture (no SA suffix)
                    # Adjust 'sa > 3' based on your actual SA count (usually 3)
                    sa_label = sa if 1 <= sa <= 3 else None
                    i_key = self._get_band_name('i', pol, sa_label)
                    q_key = self._get_band_name('q', pol, sa_label)
                    complex_arr = f[i_key][:] + 1j * f[q_key][:]
                    y_list.append(self._complex_to_channels(complex_arr))

            x = torch.cat(x_list, dim=0)
            y = torch.cat(y_list, dim=0)

            incidence_angle = f['bands/localIncidenceAngle'][:].mean()

        if x.shape[-1] > x.shape[-2]:
            x, y = x.transpose(-1, -2), y.transpose(-1, -2)

        x, y = _normalize(x, y, global_max=self.cfg["data"].get("global_max", 4257)) # TODO: check global max for new dataset

        target_h = self.cfg["data"].get("image_height", 96)
        target_w = self.cfg["data"].get("image_width", 96)
        # x, y = _strip_crop(x, y, target_h, target_w)

        if self.da:
            x, y = tfm_dihedral([x, y], random.randint(0, 7))

        meta_out = {
            "idx": idx,
            "incidence_angle": torch.tensor(float(incidence_angle)),
            "patch_id": os.path.basename(file_path)
        }

        return x, y, meta_out

    def _complex_to_channels(self, arr):
        mag = np.abs(arr).astype(np.float32)
        phase = np.angle(arr).astype(np.float32)
        combined = np.stack([mag, phase], axis=0)  # (2, H, W)
        return torch.from_numpy(combined)

if __name__ == "__main__":
    ## ---- TEST ---- ##
    ## inspect tensor shapes and information in tiles
    ## -------------- ##
    test_cfg = {
        "data": {
            "root_dir": "/shared/home/lvanderpeet/AE5822-Thesis/dataset",  # Relative path to your dataset folder
            "subaperture_config": {
                "active_polarizations": ["VV"],
                "input_indices": [1, 2, 3],  # SA1, SA2, SA3
                "output_indices": [4]  # 4 maps to Full Aperture in our logic
            },
            "split": {"train": 1.0, "valid": 0.0},
            "image_height": 96,
            "image_width": 96,
            "global_max": 4257
        },
        "seed": 42
    }

    print("---  Testing H5ComplexSARDataset Loading ---")

    try:
        # 2. Initialize the dataset
        # We use split='train' to find the files in the B6D5 folder
        dataset = H5ComplexSARDataset(test_cfg, split='train', da=True)

        if len(dataset) == 0:
            print(" No files found. Check if 'root_dir' matches your path.")
        else:
            print(f" Found {len(dataset)} tiles.")

            # 3. Pull a sample (This runs __getitem__)
            x, y, meta = dataset[0]

            print("\n[TENSOR SHAPES]")
            print(f"  Input (x):  {x.shape}")
            print(f"  Target (y): {y.shape}")

            print("\n[VALUE RANGES]")
            print(f"  X Min/Max: {x.min():.4f} / {x.max():.4f}")
            print(f"  Y Min/Max: {y.min():.4f} / {y.max():.4f}")

            print("\n[METADATA]")
            print(f"  Patch ID: {meta['patch_id']}")
            print(f"  Incidence Angle: {meta['incidence_angle']:.2f}°")

    except Exception as e:
        print(f"\n ERROR during execution: {e}")
        import traceback

        traceback.print_exc()

    ## ---- TEST ---- ##
    ## Plot magnitude of patches
    ## -------------- ##
    # import matplotlib.pyplot as plt
    #
    # test_cfg = {
    #     "data": {
    #         "root_dir": "/shared/home/lvanderpeet/AE5822-Thesis/dataset",  # Relative path to your dataset folder
    #         "subaperture_config": {
    #             "active_polarizations": ["VV"],
    #             "input_indices": [1, 2, 3],  # SA1, SA2, SA3
    #             "output_indices": [4]  # 4 maps to Full Aperture in our logic
    #         },
    #         "split": {"train": 1.0, "valid": 0.0},
    #         "image_height": 96,
    #         "image_width": 96,
    #         "global_max": 4257
    #     },
    #     "seed": 42
    # }
    #
    # def inspect_sample(x, y, meta):
    #     """
    #     Visualizes the Magnitude of the 3 input SAs and the 1 Target.
    #     x: [6, 96, 96] -> Indices 0, 2, 4 are Magnitudes
    #     y: [2, 96, 96] -> Index 0 is Magnitude
    #     """
    #     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    #
    #     # Extract magnitudes (undoing the [-1, 1] scaling visually for better contrast)
    #     s1_mag = x[0].numpy()
    #     s2_mag = x[2].numpy()
    #     s3_mag = x[4].numpy()
    #     target_mag = y[0].numpy()
    #
    #     titles = ['Sub-Aperture 1', 'Sub-Aperture 2', 'Sub-Aperture 3', 'Target (Full)']
    #     imgs = [s1_mag, s2_mag, s3_mag, target_mag]
    #
    #     for ax, img, title in zip(axes, imgs, titles):
    #         im = ax.imshow(img, cmap='gray')
    #         ax.set_title(title)
    #         ax.axis('off')
    #
    #     plt.suptitle(f"Patch: {meta['patch_id']} | Incidence: {meta['incidence_angle']:.2f}°")
    #     plt.tight_layout()
    #     plt.show()
    #
    #
    # dataset = H5ComplexSARDataset(test_cfg, split='train', da=True)
    #
    # x, y, meta = dataset[0]
    # for i in range(3):
    #     mag_slice = x[i * 2]
    #     print(f"SA{i + 1} - Mean: {mag_slice.mean():.4f}, Std: {mag_slice.std():.4f}")
    #
    # # Check if phase is actually distributed between -1 and 1
    # phase_slice = x[1]
    # print(f"Phase - Mean: {phase_slice.mean():.4f}, Range: [{phase_slice.min():.2f}, {phase_slice.max():.2f}]")
    # inspect_sample(x, y, meta)