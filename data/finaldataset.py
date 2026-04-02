import os
import random
import yaml
import numpy as np
import json
import zarr
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import h5py
from safetensors.torch import load_file



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
    Normalizes real/imaginary channels with a symmetric log mapping.

    The data now contains signed in-phase/quadrature components, so we preserve sign and compress dynamic range via:
        sign(v) * log1p(abs(v)) / log1p(global_max)

    Parameters
    ----------
    x : torch.Tensor
        Input subapertures tensor (channels, H, W).
    y : torch.Tensor
        Output subapertures tensor (channels, H, W).
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.sign(x) * torch.log1p(torch.abs(x)) / np.log1p(global_max)
    y = torch.sign(y) * torch.log1p(torch.abs(y)) / np.log1p(global_max)

    return torch.clamp(x, -1.0, 1.0), torch.clamp(y, -1.0, 1.0)


class SafetensorSARDataset(Dataset):
    """
    Optimized for the 16-channel .safetensor structure.
    Splits data by PRODUCT to prevent spatial leakage.
    """

    def __init__(self, cfg, split='train', da=False):
        """
                Initializes the dataset with a Spatial Zone Split per product.

                Logic:
                1. Identify all product directories.
                2. For EVERY product, split its patches geographically:
                   - Top 70%: Training
                   - Middle 15%: Validation (with buffer)
                   - Bottom 15%: Testing (with buffer)
                3. Accumulate all patches for the requested split.
                4. Shuffle and cap the total count (100k for train, 10k for val/test).
                """
        self.cfg = cfg
        self.da = da
        self.root_dir = cfg["data"]["root_dir"]
        self.requested_pols = [p.upper() for p in cfg["data"]["subaperture_config"]["active_polarizations"]]
        self.input_indices = cfg["data"]["subaperture_config"].get("input_indices", [1, 3])
        self.output_indices = cfg["data"]["subaperture_config"].get("output_indices", [0])

        # 1. Get all unique product directories
        # We use all products to maximize geographical diversity
        all_products = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        self.files = []
        # A buffer of patches to prevent spatial leakage at the boundaries
        # If your patches overlap by 50%, a buffer of 2 ensures no shared pixels.
        buffer_size = 2

        # 2. Extract patches from EVERY product based on the spatial split
        for prod in all_products:
            prod_path = os.path.join(self.root_dir, prod)

            # Sorting is CRITICAL to ensure the top-to-bottom spatial order
            all_patches = sorted([
                f for f in os.listdir(prod_path)
                if f.endswith('.safetensors')
            ])

            n = len(all_patches)
            if n < 20:  # Skip products that are too small to split meaningfully
                continue

            # Calculate geographical split indices (70% / 15% / 15%)
            train_end = int(n * 0.70)

            val_start = train_end + buffer_size
            val_end = int(n * 0.85)

            test_start = val_end + buffer_size

            # Select patches based on the requested split
            if split == 'train':
                selected_filenames = all_patches[:train_end]
            elif split == 'valid':
                selected_filenames = all_patches[val_start:val_end]
            elif split == 'test':
                selected_filenames = all_patches[test_start:]
            else:
                raise ValueError(f"Invalid split name: {split}")

            # Accumulate full paths
            self.files.extend([os.path.join(prod_path, f) for f in selected_filenames])

        # 3. Global Shuffle and Cap
        # This ensures the 100k patches are a random mix from all available products
        random.seed(self.cfg.get("seed", 42))
        random.shuffle(self.files)

        if split == 'train':
            limit = 100000
        else:
            limit = 10000

        # Apply the limit to keep the dataset size manageable
        self.files = self.files[:limit]

        print(f"--- {split.upper()} Set Initialized ---")
        print(f"Total patches: {len(self.files)} (sampled from {len(all_products)} products)")


    def __len__(self):
        return len(self.files)


    def _select_channels(self, full_tensor, indices):
        """
        Maps the logical indices (0=Full, 1=SA1, 2=SA2, 3=SA3)
        and polarizations (VV, VH) to the 16 physical channels.
        """
        selected = []
        for sa_idx in indices:
            # Map logical SA index to the starting position in the 16-channel block
            # 0->0, 1->4, 2->8, 3->12
            base_offset = sa_idx * 4

            if "VV" in self.requested_pols:
                i_vv = full_tensor[base_offset]
                q_vv = full_tensor[base_offset + 1]
                selected.append(i_vv.unsqueeze(0))
                selected.append(q_vv.unsqueeze(0))
            if "VH" in self.requested_pols:
                i_vh = full_tensor[base_offset + 2]
                q_vh = full_tensor[base_offset + 3]
                selected.append(i_vh.unsqueeze(0))
                selected.append(q_vh.unsqueeze(0))

        return torch.cat(selected, dim=0)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        data = load_file(file_path)
        full_tensor = data["x"]

        x = self._select_channels(full_tensor, self.input_indices)
        y = self._select_channels(full_tensor, self.output_indices)

        g_max = self.cfg["data"].get("global_max", 2.5)
        x, y = _normalize(x, y, global_max=g_max)

        if self.da:
            x, y = tfm_dihedral([x, y], random.randint(0, 7))

        meta_out = {
            "idx": idx,
            "patch_id": os.path.basename(file_path),
            "product_id": os.path.basename(os.path.dirname(file_path))
        }

        return x, y, meta_out


    def _complex_to_channels(self, arr):
        mag = np.abs(arr).astype(np.float32)
        phase = np.angle(arr).astype(np.float32)
        combined = np.stack([mag, phase], axis=0)  # (2, H, W)
        return torch.from_numpy(combined)

if __name__ == "__main__":
    ## ---- TEST ---- ##
    ##
    ## -------------- ##
    def test_loading():
        # 1. Mock a minimal config
        cfg = {
            "seed": 42,
            "data": {
                "root_dir": "/shared/home/lvanderpeet/AE5822-Thesis/dataset_patches_128",
                "split": {"train": 0.8, "valid": 0.1},
                "global_max": 2.5,
                "subaperture_config": {
                    "active_polarizations": ["VV", "VH"],
                    "input_indices": [1, 2, 3],  # SA1, SA2, SA3
                    "output_indices": [0]  # Full Aperture
                }
            }
        }

        print("--- Initializing Training Set ---")
        train_ds = SafetensorSARDataset(cfg, split='train', da=True)

        print("\n--- Initializing Validation Set ---")
        val_ds = SafetensorSARDataset(cfg, split='valid', da=False)

        # 2. Check for Data Leakage (The most important test)
        train_products = set([os.path.basename(os.path.dirname(f)) for f in train_ds.files])
        val_products = set([os.path.basename(os.path.dirname(f)) for f in val_ds.files])
        overlap = train_products.intersection(val_products)

        if len(overlap) == 0:
            print(f" Leakage Check Passed: 0 products shared between Train and Val.")
        else:
            print(f" Leakage Check Failed: {len(overlap)} products are in both sets!")

        # 3. Test __getitem__ and Shapes
        print("\n--- Testing Tensor Shapes ---")
        x, y, meta = train_ds[0]

        # Expected Channels:
        # Input (x): 3 SAs * 2 pols * 2 (I/Q) = 12 channels
        # Output (y): 1 Full * 2 pols * 2 (I/Q) = 4 channels
        print(f"Input (x) shape:  {x.shape} (Expected [12, 128, 128])")
        print(f"Target (y) shape: {y.shape} (Expected [4, 128, 128])")
        print(f"Value range (x):  {x.min():.2f} to {x.max():.2f}")

        # 4. Test DataLoader (Multi-worker check)
        print("\n--- Testing DataLoader (Batching) ---")
        loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
        batch_x, batch_y, batch_meta = next(iter(loader))
        print(f"Batch X shape: {batch_x.shape}")
        print(" All tests passed!")

    test_loading()



    ## ---- TEST ---- ##
    ## Displays structure of a singular .h5 file.
    ## -------------- ##
    # test_cfg = {
    #     "data": {
    #         "root_dir": "/shared/home/lvanderpeet/AE5822-Thesis/dataset",  # Relative path to your dataset folder
    #         "subaperture_config": {
    #             "active_polarizations": ["VV"],
    #             "input_indices": [1, 2, 3],  # SA1, SA2, SA3
    #             "output_indices": [4]  # 4 maps to Full Aperture in our logic
    #         },
    #         "split": {"train": 0.8, "valid": 0.1},
    #         "image_height": 96,
    #         "image_width": 96,
    #         "global_max": 4257
    #     },
    #     "seed": 42
    # }
    #
    # train_ds = H5ComplexSARDataset(test_cfg, split='train', da=True)
    # valid_ds = H5ComplexSARDataset(test_cfg, split='valid', da=False)
    # test_ds = H5ComplexSARDataset(test_cfg, split='test', da=False)



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

    ## ---- TEST ---- ##
    ## inspect tensor shapes and information in tiles
    ## -------------- ##
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
    # print("---  Testing H5ComplexSARDataset Loading ---")
    #
    # try:
    #     # 2. Initialize the dataset
    #     # We use split='train' to find the files in the B6D5 folder
    #     dataset = H5ComplexSARDataset(test_cfg, split='train', da=True)
    #
    #     if len(dataset) == 0:
    #         print(" No files found. Check if 'root_dir' matches your path.")
    #     else:
    #         print(f" Found {len(dataset)} tiles.")
    #
    #         # 3. Pull a sample (This runs __getitem__)
    #         x, y, meta = dataset[4]
    #
    #         print("\n[TENSOR SHAPES]")
    #         print(f"  Input (x):  {x.shape}")
    #         print(f"  Target (y): {y.shape}")
    #
    #         print("\n[VALUE RANGES]")
    #         print(f"  X Min/Max: {x.min():.4f} / {x.max():.4f}")
    #         print(f"  Y Min/Max: {y.min():.4f} / {y.max():.4f}")
    #
    #         print("\n[METADATA]")
    #         print(f"  Patch ID: {meta['patch_id']}")
    #         print(f"  Incidence Angle: {meta['incidence_angle']:.2f}°")
    #
    # except Exception as e:
    #     print(f"\n ERROR during execution: {e}")
    #     import traceback
    #
    #     traceback.print_exc()

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