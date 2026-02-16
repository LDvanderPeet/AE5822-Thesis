import matplotlib.pyplot as plt
import torch
import os
import yaml
import shutil
import numpy as np
import csv
from datetime import datetime
import pandas as pd

def log_to_csv(run_dir, data_dict):
    """
    Saves current epoch metrics to a CSV file for tracking.

    If the file does not exist, it creates it and writes a header based on the keys in data_dict.

    Parameters
    ----------
    run_dir : str
        Path to the specific experiment directory.
    data_dict : dict
        Dictionary of training metrics and their respective values
        e.g., {"epoch": 1, "train_loss": 0.004, "val_loss": 0.0035, "lr": 1e-4}.
    """
    csv_path = os.path.join(run_dir, "results.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames = data_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)

def setup_run_directory(config_path):
    """
    Initialize a unique experiment folder with a timestamp and configuration copy.

    Ensures reproducibility by saving the exact config.yaml and results in one folder.

    Parameters
    ----------
    config_path : str
        Path to the config.yaml file.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["save"]["name"]
    time_stamp = datetime.now().strftime("%m%d_%H%M")
    run_folder_name = f"{time_stamp}-run-{run_name}"

    base_project_dir = os.path.dirname(config_path)
    all_runs_dir = os.path.join(base_project_dir, "runs")
    run_dir = os.path.join(all_runs_dir, run_folder_name)
    viz_dir = os.path.join(run_dir, "visuals")

    os.makedirs(viz_dir, exist_ok=True)

    shutil.copy2(config_path, os.path.join(run_dir, "config.yaml"))

    return run_dir, viz_dir


def visualize_reconstruction(model, val_loader, device, epoch, viz_dir, sa_index=0):
    """
    Generate and save comparison between Ground Truth and Model Prediction.

    Parameters
    ----------
    model : torch.nn.Module
        The trained SAR model.
    val_loader : torch.utils.data.DataLoader
        Validation data loader to pull a sample from.
    epoch : int
        Current epoch number for labeling of filename.
    viz_dir : str
        Directory where the resulting .png will be saved.
    sa_index : int
        The index of the subaperture to visualize.
    """
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

def generate_final_plots(run_dir):
    """
    Analyze the results.csv to produce a learning curve.

    Parameters
    ----------
    run_dir : str
        The experiment directory containing 'results.csv'.
    """
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        print("No results.csv found")
        return

    df = pd.read_csv(csv_path)

    ig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Axis: Loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)', color='tab:blue')
    ax1.plot(df['epoch'], df['train_loss'].astype(float), label='Train Loss', color='tab:blue', linestyle='--')
    ax1.plot(df['epoch'], df['val_loss'].astype(float), label='Val Loss', color='tab:blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Right Axis: Learning Rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='tab:red')
    ax2.step(df['epoch'], df['lr'].astype(float), color='tab:red', where='post', alpha=0.7)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f"Training History: {os.path.basename(run_dir)}")
    ax1.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "learning_curve.pdf"))
    plt.close()
