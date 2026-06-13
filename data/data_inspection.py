import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, uniform_filter

# ---- Matplotlib Global Settings ----
plt.rcdefaults()
plt.rcParams.update({
    "font.family": "serif",
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


# ==========================================
# 1. Single Patch Inspection
# ==========================================
def inspect_single_patch(file_path):
    """Loads a single safetensor patch and plots the Magnitude + I/Q Distributions."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = load_file(file_path)["x"]
    i_comp = data[0].numpy()
    q_comp = data[1].numpy()

    mag = np.sqrt(i_comp ** 2 + q_comp ** 2)
    mean_val, std_val = np.mean(mag), np.std(mag)
    vmax_disp = mean_val + (3 * std_val)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot A: Amplitude Image
    axes[0].imshow(np.clip(mag, 0, vmax_disp), cmap='gray')
    axes[0].set_title(f"SAR Magnitude (Clipped at 3σ: {vmax_disp:.2f})")
    axes[0].axis('off')

    # Plot B: I/Q Scatter
    axes[1].scatter(i_comp.flatten(), q_comp.flatten(), s=1, alpha=0.1, c='blue')
    axes[1].set_title("Complex Plane (I vs Q)")
    axes[1].set_xlabel("In-Phase (I)")
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].grid(True)

    # Plot C: Magnitude Histogram (Log Scale)
    axes[2].hist(mag.flatten(), bins=100, color='darkred', alpha=0.7)
    axes[2].set_title("Magnitude Distribution (Log Y)")
    axes[2].set_xlabel("Absolute Amplitude")
    axes[2].set_yscale('log')

    plt.suptitle(f"Patch Analysis: {os.path.basename(file_path)}", fontsize=14)
    plt.tight_layout()
    plt.show()


# ==========================================
# 2. Full Tile Reconstruction
# ==========================================
def plot_full_original_tile(h5_file_path, downsample_factor=4, group_name="bands"):
    """
    Loads a full, un-patched SAR tile directly from an HDF5 file.
    Calculates the Full Aperture VV magnitude and plots it.
    """
    if not os.path.exists(h5_file_path):
        print(f"File not found: {h5_file_path}")
        return


    with h5py.File(h5_file_path, 'r') as f:
        if group_name not in f:
            print(f"Error: Group '{group_name}' not found in the H5 file. Available keys: {list(f.keys())}")
            return

        group = f[group_name]

        if "i_VV" not in group or "q_VV" not in group:
            print(
                f"Error: 'i_VV' or 'q_VV' not found in group '{group_name}'. Available datasets: {list(group.keys())}")
            return

        print(f"Applying downsampling factor of {downsample_factor}x...")
        i_comp = group["i_VV"][::downsample_factor, ::downsample_factor].astype(np.float32)
        q_comp = group["q_VV"][::downsample_factor, ::downsample_factor].astype(np.float32)

    print(f"Loaded array shape: {i_comp.shape}")
    print("Calculating Magnitude...")
    mag = np.sqrt(np.square(i_comp) + np.square(q_comp))

    active_pixels = mag[mag > 0]
    if len(active_pixels) == 0:
        print("Warning: The loaded array is entirely zeros.")
        return

    mean_val, std_val = np.mean(active_pixels), np.std(active_pixels)
    vmax_disp = mean_val + (4 * std_val)

    print("Rendering full scene...")
    plt.figure(figsize=(14, 10))
    plt.imshow(np.clip(mag, 0, vmax_disp), cmap='gray')
    plt.title(
        f"Original SAR Tile (VV Polarization)\n{os.path.basename(h5_file_path)}\n(Downsampled {downsample_factor}x)")
    plt.colorbar(label="Amplitude")
    plt.axis('off')

    save_filename = f"Full_Tile_{os.path.basename(h5_file_path).replace('.h5', '')}.pdf"
    plt.savefig(save_filename, bbox_inches='tight')
    print(f"Saved high-res render to: {save_filename}")

    plt.show()


# ==========================================
# 3. Global Dataset Statistics
# ==========================================
def _process_single_file_stats(path):
    """Helper for parallel processing."""
    try:
        data = load_file(path)["x"]
        sum_x = torch.sum(data, dim=(1, 2)).numpy()
        sum_x2 = torch.sum(data ** 2, dim=(1, 2)).numpy()
        count = data.shape[1] * data.shape[2]
        min_v = data.amin(dim=(1, 2)).numpy()
        max_v = data.amax(dim=(1, 2)).numpy()
        return sum_x, sum_x2, count, min_v, max_v
    except Exception:
        return None


def compute_dataset_statistics(root_dir, n_jobs=16):
    """Calculates Mean, Std, Min, Max across the entire dataset."""
    print("Collecting file list...")
    all_paths = glob.glob(os.path.join(root_dir, "**/*.safetensors"), recursive=True)

    print(f"Starting stats calculation for {len(all_paths)} files...")
    results = Parallel(n_jobs=n_jobs)(delayed(_process_single_file_stats)(p) for p in tqdm(all_paths))
    results = [r for r in results if r is not None]

    if not results:
        print("No valid data found.")
        return

    total_sum = np.zeros(16)
    total_sum_x2 = np.zeros(16)
    total_count = 0
    global_min = np.full(16, np.inf)
    global_max = np.full(16, -np.inf)

    for s, s2, c, mi, ma in results:
        total_sum += s
        total_sum_x2 += s2
        total_count += c
        global_min = np.minimum(global_min, mi)
        global_max = np.maximum(global_max, ma)

    mean = total_sum / total_count
    var = (total_sum_x2 / total_count) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-10))

    print("\n--- Global Stats per Channel ---")
    print(f"{'Chan':<5} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
    for i in range(16):
        print(f"{i:<5} | {mean[i]:.4f} | {std[i]:.4f} | {global_min[i]:.4f} | {global_max[i]:.4f}")


def detect_polarimetric_reflectors(h5_file_path, min_sigma=7.0, min_scr=15.0, min_pol_ratio=5.0):
    if not os.path.exists(h5_file_path):
        return

    print(f"Loading 1x resolution chunk: {os.path.basename(h5_file_path)}...")

    with h5py.File(h5_file_path, 'r') as f:
        i_vv = f["bands"]["i_VV"][()].astype(np.float32)
        q_vv = f["bands"]["q_VV"][()].astype(np.float32)

        has_vh = "i_VH" in f["bands"]
        if has_vh:
            i_vh = f["bands"]["i_VH"][()].astype(np.float32)
            q_vh = f["bands"]["q_VH"][()].astype(np.float32)
        else:
            print("WARNING: No VH polarization found. Skipping polarimetric filter.")

    mag_vv = np.sqrt(np.square(i_vv) + np.square(q_vv))
    active_pixels = mag_vv[mag_vv > 0]
    if len(active_pixels) == 0: return

    mean_vv, std_vv = np.mean(active_pixels), np.std(active_pixels)

    global_thresh = mean_vv + (min_sigma * std_vv)
    is_bright = mag_vv > global_thresh
    is_peak = maximum_filter(mag_vv, size=5) == mag_vv

    w_out, w_in = 15, 5
    sum_outer = uniform_filter(mag_vv, size=w_out) * (w_out ** 2)
    sum_inner = uniform_filter(mag_vv, size=w_in) * (w_in ** 2)
    clutter_mean = (sum_outer - sum_inner) / ((w_out ** 2) - (w_in ** 2))
    clutter_mean[clutter_mean == 0] = 1e-6

    scr_vv = mag_vv / clutter_mean
    spatial_targets = is_bright & is_peak & (scr_vv > min_scr)

    if has_vh:
        mag_vh = np.sqrt(np.square(i_vh) + np.square(q_vh))
        mag_vh[mag_vh == 0] = 1e-6
        pol_ratio = mag_vv / mag_vh

        final_targets = spatial_targets & (pol_ratio > min_pol_ratio)
    else:
        final_targets = spatial_targets

    y_coords, x_coords = np.where(final_targets)
    print(f"Filtered down to {len(y_coords)} Polarimetric Corner Reflectors.")

    vmax_disp = mean_vv + (4 * std_vv)
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(np.clip(mag_vv, 0, vmax_disp), cmap='gray')

    if len(x_coords) > 0:
        ax.scatter(x_coords, y_coords, facecolors='none', edgecolors='red', s=120, linewidths=2.0)

    title = f"Multi-Pol CR Detection\nFile: {os.path.basename(h5_file_path)}"
    if has_vh: title += f" (SCR>{min_scr}, VV/VH>{min_pol_ratio})"
    ax.set_title(title)
    ax.axis('off')

    save_filename = f"Surat_MultiPol_{os.path.basename(h5_file_path).replace('.h5', '.pdf')}"
    plt.savefig(save_filename, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # ==========================================
    # Main Execution Block
    # Uncomment the function you want to run!
    # ==========================================

    DATASET_ROOT = "/shared/home/lvanderpeet/AE5822-Thesis/dataset_patches_128"
    FULL_TILE_PATH = "/shared/home/lvanderpeet/AE5822-Thesis/dataset/S1C_S6_SLC__1SDV_20250417T084009_20250417T084038_001930_003C95_78FA/299D_1489R.h5"

    TEST_PATCH = "/shared/home/lvanderpeet/AE5822-Thesis/dataset_patches_128/S1A_S4_SLC__1SDV_20160629T224400_20160629T224425_011932_012621_65C3/512U_564L__2880_512.safetensors"

    test_subaperture_linearity(TEST_PATCH)

    ## 1. ---- Inspect a single patch ----
    SAMPLE_PATCH = os.path.join(DATASET_ROOT, "S1A_S4_SLC__1SDV_20160629T224400_20160629T224425_011932_012621_65C3",
                                "512U_564L__2880_512.safetensors")
    inspect_single_patch(SAMPLE_PATCH)

    ## 2. ---- Reconstruct and Plot a FULL TILE ----
    plot_full_original_tile(FULL_TILE_PATH, downsample_factor=1)

    ## 3. ---- Compute Global Statistics (Takes time) ----
    compute_dataset_statistics(DATASET_ROOT, n_jobs=16)
    detect_polarimetric_reflectors(FULL_TILE_PATH)