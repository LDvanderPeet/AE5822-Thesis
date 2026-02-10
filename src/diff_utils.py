import matplotlib.pyplot as plt
import torch
import os
import yaml
import shutil
import numpy as np

def setup_run_directory(config_path):
    """
    Creates a folder based on the 'save: name:' in config.yaml
    and copies the config file for reproducibility
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["save"]["name"]
    run_folder_name = f"run-{run_name}"

    base_project_dir = os.path.dirname(config_path)
    all_runs_dir = os.path.join(base_project_dir, "runs")
    run_dir = os.path.join(all_runs_dir, run_folder_name)
    viz_dir = os.path.join(run_dir, "visuals")

    os.makedirs(viz_dir, exist_ok=True)

    shutil.copy2(config_path, os.path.join(run_dir, "config.yaml"))

    return run_dir, viz_dir


# def visualize_reconstruction(model, val_loader, device, epoch, viz_dir, sa_index=0):
#     model.eval()
#     # Get a single batch
#     x, y, _ = next(iter(val_loader))
#     x, y = x.to(device), y.to(device)
#
#     with torch.no_grad():
#         output = model(x)
#
#     # Move back to CPU for plotting
#     # Assuming x is [B, C, H, W] and y is [B, C, H, W]
#     # We take the first image in the batch [0] and the specific subaperture [sa_index]
#     original = y[0, sa_index].cpu().numpy()
#     reconstructed = output[0, sa_index].cpu().numpy()
#     error = np.abs(original - reconstructed)
#
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     fig.suptitle(f"Epoch {epoch} - Subaperture {sa_index} Comparison", fontsize=16)
#
#     im0 = axes[0].imshow(original, cmap='magma')
#     axes[0].set_title("Original (Ground Truth)")
#     fig.colorbar(im0, ax=axes[0])
#
#     im1 = axes[1].imshow(reconstructed, cmap='magma')
#     axes[1].set_title("UNet Reconstruction")
#     fig.colorbar(im1, ax=axes[1])
#
#     im2 = axes[2].imshow(error, cmap='seismic')  # Seismic is great for diffs (red/blue)
#     axes[2].set_title("Residual (Difference)")
#     fig.colorbar(im2, ax=axes[2])
#
#     plt.tight_layout()
#     save_path = os.path.join(viz_dir, f"epoch_{epoch}_sa{sa_index}.png")
#     plt.savefig(f"reconstruction_epoch_{epoch}.png")
#     plt.close()  # Close to free up memory

def visualize_reconstruction(model, val_loader, device, epoch, viz_dir, sa_index=0):
    model.eval()

    try:
        x, y, _ = next(iter(val_loader))
    except StopIteration:
        return

    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        output = model(x)

    original = y[0, sa_index].cpu().numpy()

    out_idx = 0 if output.shape[1] == 1 else sa_index
    reconstructed = output[0, out_idx].cpu().numpy()

    abs_error = np.abs(original - reconstructed)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Epoch {epoch} - Subaperture {sa_index} Analysis", fontsize=16)

    v_min, v_max = original.min(), original.max()

    im0 = axes[0].imshow(original, cmap='magma', vmin=v_min, vmax=v_max)
    axes[0].set_title("Original (GT)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(reconstructed, cmap='magma')
    axes[1].set_title("Reconstruction (Auto-scaled)")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(abs_error, cmap='viridis')
    axes[2].set_title("Absolute Error")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    save_path = os.path.join(viz_dir, f"epoch_{epoch}_sa{sa_index}.png")
    plt.savefig(save_path)
    plt.close()

