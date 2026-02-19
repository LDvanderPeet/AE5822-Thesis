import os
import yaml
import numpy as np
import zarr
from tqdm import tqdm
import shutil

def quarantine_bad_patches(config_path, quarantine_dir="/shared/home/lvanderpeet/AE5822-Thesis/data/quarantine"):
    """Remove all corrupted patches and move them to /data/quarantine.

    Run this script once before training."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root_dir = cfg["data"]["root_dir"]

    if not os.path.exists(quarantine_dir):
        os.makedirs(quarantine_dir)

    patch_ids = sorted([d for d in os.listdir(root_dir) if d.isdigit()])
    removed_count = 0

    print(f"Scanning {len(patch_ids)} patches for NaNs/Infs...")
    for pid in tqdm(patch_ids):
        path = os.path.join(root_dir, pid, "vv", "sub_3")
        if not os.path.exists(path): continue

        try:
            data = zarr.open(path, mode='r')[:]

            if np.isnan(data).any() or np.isinf(data).any():
                shutil.move(os.path.join(root_dir, pid), os.path.join(quarantine_dir, pid))
                removed_count += 1
        except Exception as e:
            print(f"\nError reading patch {pid}: {e}")

    print(f"\nCleanup complete. Moved {removed_count} corrupted patches to {quarantine_dir}.")

if __name__ == "__main__":
    config_path = "/shared/home/lvanderpeet/AE5822-Thesis/config.yaml"
    quarantine_bad_patches(config_path)