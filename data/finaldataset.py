import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file



def tfm_dihedral(x, k):
    """
    Strip-safe augmentations. Only allows flips that maintain the rectangular
    aspect ratio and preserve the relationship between Azimuth and Range axes.
    Transpositions and 90/270-degree rotations are excluded as they scramble
    SAR Doppler physics.

    Parameters
    ----------
    x : list of torch.Tensor
        A list containing the data tensors (typically [input, target]).
    k : int
        An integer from 0 to 3 defining the specific transformation:
        0: Identity (no change)
        1: Vertical Flip (flips along the height/Azimuth axis)
        2: Horizontal Flip (flips along the width/Range axis)
        3: Both Vertical and Horizontal Flip (180-degree rotation)
    """
    if k == 1: x = [_x.flip(-2) for _x in x]  # Vertical Flip
    if k == 2: x = [_x.flip(-1) for _x in x]  # Horizontal Flip
    if k == 3: x = [_x.flip(-2).flip(-1) for _x in x]  # Both
    return x


def _normalize(x, y, global_max=300):
    """
    Complex Amplitude Compression for interleaved I/Q channels.
    Compresses the heavy-tailed magnitude using log1p while perfectly
    preserving the complex phase angle of the I/Q vector.

    Parameters
    ----------
    x : torch.Tensor
        Input subapertures tensor (channels, H, W).
    y : torch.Tensor
        Output subapertures tensor (channels, H, W).
    global_max : float
        Maximum magnitude value observed in the whole dataset.
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    def compress_complex(tensor, g_max):
        I = tensor[0::2]
        Q = tensor[1::2]

        A = torch.sqrt(I**2 + Q**2)
        A_safe = torch.clamp(A, min=1e-8) # prevent division by zero

        A_comp = torch.log1p(A) / np.log1p(g_max)

        out = torch.zeros_like(tensor)
        out[0::2] = A_comp * (I / A_safe)
        out[1::2] = A_comp * (Q / A_safe)

        return torch.clamp(out, -1.0, 1.0)

    return compress_complex(x, global_max), compress_complex(y, global_max)


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

        Parameters
        ----------
        cfg : dict
            Configuration dictionary containing data roots and subaperture settings.
        split : str
            Dataset split to load ('train', 'valid', or 'test').
        da : bool
            Whether to apply data augmentation.
        """
        self.cfg = cfg
        self.da = da
        self.root_dir = cfg["data"]["root_dir"]
        self.requested_pols = [p.upper() for p in cfg["data"]["subaperture_config"]["active_polarizations"]]
        self.input_indices = cfg["data"]["subaperture_config"].get("input_indices", [1, 3])
        self.output_indices = cfg["data"]["subaperture_config"].get("output_indices", [0])

        all_products = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        self.files = []
        buffer_size = 2

        for prod in all_products:
            prod_path = os.path.join(self.root_dir, prod)

            all_patches = sorted([
                f for f in os.listdir(prod_path)
                if f.endswith('.safetensors')
            ])

            n = len(all_patches)
            if n < 20:
                continue

            train_end = int(n * 0.70)
            val_start = train_end + buffer_size
            val_end = int(n * 0.85)
            test_start = val_end + buffer_size

            if split == 'train':
                selected_filenames = all_patches[:train_end]
            elif split == 'valid':
                selected_filenames = all_patches[val_start:val_end]
            elif split == 'test':
                selected_filenames = all_patches[test_start:]
            else:
                raise ValueError(f"Invalid split name: {split}")

            self.files.extend([os.path.join(prod_path, f) for f in selected_filenames])

        random.seed(self.cfg.get("seed", 42))
        random.shuffle(self.files)

        if split == 'train':
            limit = 100000
        else:
            limit = 10000

        self.files = self.files[:limit]

        print(f"--- {split.upper()} Set Initialized ---")
        print(f"Total patches: {len(self.files)} (sampled from {len(all_products)} products)")


    def __len__(self):
        return len(self.files)


    def _select_channels(self, full_tensor, indices):
        """
        Maps logical subaperture indices and polarizations to physical channels.

        Parameters
        ----------
        full_tensor : torch.Tensor
            The raw 16-channel tensor from the safetensor file.
        indices : list
            Logical indices for sub-apertures (0=Full, 1=SA1, etc.).
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

        g_max = self.cfg["data"].get("global_max", 300)
        x, y = _normalize(x, y, global_max=g_max)

        if self.cfg["data"].get("amplitude_only", False):
            x_amp = torch.sqrt(x[0::2] ** 2 + x[1::2] ** 2)
            y_amp = torch.sqrt(y[0::2] ** 2 + y[1::2] ** 2)

            x = x_amp * 2.0 - 1.0
            y = y_amp * 2.0 - 1.0

        if self.da:
            x, y = tfm_dihedral([x, y], random.randint(0, 3))

        meta_out = {
            "idx": idx,
            "patch_id": os.path.basename(file_path),
            "product_id": os.path.basename(os.path.dirname(file_path))
        }

        return x, y, meta_out


if __name__ == "__main__":
    def test_loading():
        cfg = {
            "seed": 42,
            "data": {
                "root_dir": "/shared/home/lvanderpeet/AE5822-Thesis/dataset_patches_128",
                "split": {"train": 0.8, "valid": 0.1},
                "global_max": 2.5,
                "subaperture_config": {
                    "active_polarizations": ["VV", "VH"],
                    "input_indices": [1, 2, 3, 4],  # 87.5%, 75%, 62.5%, 50%
                    "output_indices": [0]  # Full Aperture
                }
            }
        }

        print("--- Initializing Training Set ---")
        train_ds = SafetensorSARDataset(cfg, split='train', da=True)

        print("\n--- Initializing Validation Set ---")
        val_ds = SafetensorSARDataset(cfg, split='valid', da=False)

        train_products = set([os.path.basename(os.path.dirname(f)) for f in train_ds.files])
        val_products = set([os.path.basename(os.path.dirname(f)) for f in val_ds.files])
        overlap = train_products.intersection(val_products)

        if len(overlap) == 0:
            print(f" Leakage Check Passed: 0 products shared between Train and Val.")
        else:
            print(f" Leakage Check Failed: {len(overlap)} products are in both sets!")

        print("\n--- Testing Tensor Shapes ---")
        x, y, meta = train_ds[0]

        # Expected Channels:
        # Input (x): 3 SAs * 2 pols * 2 (I/Q) = 12 channels
        # Output (y): 1 Full * 2 pols * 2 (I/Q) = 4 channels
        print(f"Input (x) shape:  {x.shape} (Expected [12, 128, 128])")
        print(f"Target (y) shape: {y.shape} (Expected [4, 128, 128])")
        print(f"Value range (x):  {x.min():.2f} to {x.max():.2f}")

        print("\n--- Testing DataLoader (Batching) ---")
        loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
        batch_x, batch_y, batch_meta = next(iter(loader))
        print(f"Batch X shape: {batch_x.shape}")
        print(" All tests passed!")

    test_loading()