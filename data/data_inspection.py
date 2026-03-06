import os
import zarr
import numpy as np
import yaml
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt

plt.rcdefaults()

plt.rcParams.update({
    "font.family": "serif",
    # This list provides fallbacks. It will try to find a serif font that exists.
    "font.serif": ["Liberation Serif", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
})


def load_complex_patch(root_dir, patch_id : str, pol, sub_folder):
    path = os.path.join(root_dir, patch_id, pol, sub_folder)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    data_zarr = zarr.open(path, mode='r')
    return np.array(data_zarr).astype(np.complex64)

def plot_patch(root_dir, patch_id, pol, sub_folder):
    data = load_complex_patch(root_dir, patch_id, pol, sub_folder)
    sample = data[0] if data.ndim == 3 else data

    mag = np.abs(sample)
    phase = np.angle(sample)

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    im1 = ax[0].imshow(mag, cmap='magma')
    ax[0].set_title(f"Patch {patch_id} Magnitude")
    plt.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(phase, cmap='gray')
    ax[1].set_title(f"Patch {patch_id} Phase")
    plt.colorbar(im2, ax=ax[1])

    plt.tight_layout()
    plt.show()

def plot_patch_dist(root_dir, patch_id, pol, sub_folder):
    data = load_complex_patch(root_dir, patch_id, pol, sub_folder)
    fig, ax = plt.subplots(1, 3, figsize=(18,5))
    sample = data.flatten()

    ax[0].hist(sample.real, bins=100, color='gray', alpha=0.7)
    ax[0].set_title(f"Real Component")
    ax[0].set_yscale("log")

    ax[1].hist(sample.imag, bins=100, color='gray', alpha=0.7)
    ax[1].set_title(f"Imaginary Component")
    ax[1].set_yscale("log")

    ax[2].hist(np.abs(sample), bins=100, color='gray', alpha=0.7)
    ax[2].set_yscale("log")
    ax[2].set_title(f"Magnitude")

    plt.suptitle(f"Component Distributions for Patch {patch_id}")
    plt.show()


def plot_global_stats(root_dir, pols, sub_folder):
    patch_ids = sorted([d for d in os.listdir(root_dir) if d.isdigit()])
    mins, means, maxes, p999s = [], [], [], []

    print(f"Checking first path: {os.path.join(root_dir, patch_ids[0], pols[0], sub_folder)}")

    print(f"Scanning {len(patch_ids)} patches...")
    for pid in tqdm(patch_ids):
        for pol in pols:
            # 1. Construct the path
            path = os.path.join(root_dir, pid, pol, sub_folder)

            # 2. Check if this specific polarization exists for this patch
            if not os.path.exists(path):
                continue  # Skip silently, just like your dataset.py does

            try:
                # 3. Load and calculate
                data = load_complex_patch(root_dir, pid, pol, sub_folder)
                mag = np.abs(data)

                # Check for NaNs to prevent empty plots
                if np.isnan(mag).any():
                    continue

                mins.append(float(np.min(mag)))
                means.append(float(np.mean(mag)))
                maxes.append(float(np.max(mag)))
                p999s.append(float(np.quantile(mag, 0.999)))

            except Exception as e:
                # This catches unexpected Zarr corruption
                print(f"Error processing {path}: {e}")
                continue
    fig, ax = plt.subplots(2, 2, figsize=(15,10))
    ax[0, 0].hist(mins, bins=50, color='white', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax[0, 0].set_title("Global Minima Distribution")

    ax[0, 1].hist(means, bins=50, color='silver', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax[0, 1].set_title("Global Means Distribution")

    ax[1, 0].hist(maxes, bins=50, color='silver', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax[1, 0].set_title("Global Maxima Distribution (Log)")
    ax[1, 0].set_yscale('log')

    ax[1, 1].hist(p999s, bins=50, color='silver', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax[1, 1].set_title("Global 99.9th Percentile Distribution")

    plt.tight_layout()
    plt.show()

def inspect_dataset(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root_dir = cfg["data"]["root_dir"]
    pols = cfg["data"]["subaperture_config"]["active_polarizations"]
    looks = len(cfg["data"]["subaperture_config"]["input_indices"]) + \
            len(cfg["data"]["subaperture_config"]["output_indices"])
    sub_folder = f"sub_{looks}"

    patch_ids = sorted([d for d in os.listdir(root_dir) if d.isdigit()])

    all_patch_maxes = []
    all_patch_999 = []

    print(f"Inspecting {len(patch_ids)} patches in {sub_folder}...")

    for pid in tqdm(patch_ids):
        for pol in pols:
            path = os.path.join(root_dir, pid, pol, sub_folder)
            if not os.path.exists(path):
                continue

            data = np.array(zarr.open(path, mode='r')).astype(np.complex64)
            mag = np.abs(data)

            all_patch_maxes.append(np.max(mag))
            all_patch_999.append(np.quantile(mag, 0.999))

    all_patch_maxes = np.array(all_patch_maxes)
    all_patch_999 = np.array(all_patch_999)

    print("\n--- STATISTICAL REPORT ---")
    print(f"Absolute Dataset Max:      {np.max(all_patch_maxes):.2f}")
    print(f"Mean of Patch Maxes:       {np.mean(all_patch_maxes):.2f}")
    print(f"99.9th Percentile (Global): {np.quantile(all_patch_999, 0.99):.2f} <--- RECOMMENDED GLOBAL_MAX")
    print(f"95th Percentile (Global):   {np.quantile(all_patch_999, 0.95):.2f}")

    bad_patch_idx = np.argmax(all_patch_maxes)
    print(f"Extreme Outlier found in Patch: {patch_ids[bad_patch_idx // len(pols)]}")

    plt.figure(figsize=(10, 5))
    plt.hist(all_patch_999, bins=100, color='blue', alpha=0.7, label='99.9th Percentiles')
    plt.axvline(np.quantile(all_patch_999, 0.99), color='red', linestyle='--', label='Suggested Scale')
    plt.title("Distribution of High-Intensity Values Across All Patches")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency (Patches)")
    plt.legend()
    plt.yscale('log')
    plt.show()


def quarantine_bad_patches(config_path, quarantine_dir="./quarantine"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root_dir = cfg["data"]["root_dir"]

    if not os.path.exists(quarantine_dir):
        os.makedirs(quarantine_dir)

    patch_ids = sorted([d for d in os.listdir(root_dir) if d.isdigit()])
    removed_count = 0

    print(f"Scanning {len(patch_ids)} patches for NaNs/Infs...")
    for pid in tqdm(patch_ids):
        # We check 'sub_3' as it's the most common input
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
    # inspect_dataset("/shared/home/lvanderpeet/AE5822-Thesis/config.yaml")
    config_path = "/shared/home/lvanderpeet/AE5822-Thesis/config.yaml"
    root_dir = "/shared/home/lvanderpeet/AE5822-Thesis/data"
    patch_id = "0003248"
    pol = "vv"
    sub_folder = "sub_3"
    # plot_patch_dist(root_dir, patch_id, pol, sub_folder)
    plot_patch(root_dir, patch_id, pol, sub_folder)

    # plot_global_stats(root_dir, pol, sub_folder)
