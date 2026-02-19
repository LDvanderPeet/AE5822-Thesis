import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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


def denormalize(tensor, global_max=4257):
    """Reverses the unit sphere normalization."""
    return tensor * global_max


def get_physical_magnitude(tensor, global_max=4257):
    """Converts normalized real/imaginary components back to the physical magnitude."""
    re = tensor[:, 0::2, ...]
    im = tensor[:, 1::2, ...]

    mag_norm = torch.sqrt(re ** 2 + im ** 2)

    return mag_norm * global_max


def visualize_reconstruction(model, val_loader, device, epoch, viz_dir, sa_index=0, global_max=4257, sample_idx=238):
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
        x, y, meta = next(iter(val_loader))
    except StopIteration:
        return

    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        output = model(x)

    # def to_mag(tensor):
    #     re = tensor[:, 0::2, ...]
    #     im = tensor[:, 1::2, ...]
    #     mag = torch.sqrt(re**2 + im**2)
    #     return mag[0,0].cpu().numpy()

    gt_mag = get_physical_magnitude(y, global_max)[0, 0].cpu().numpy()
    pred_mag = get_physical_magnitude(output, global_max)[0, 0].cpu().numpy()
    err_mag = np.abs(gt_mag - pred_mag)

    # gt_mag = to_mag(y)
    # pred_mag = to_mag(output)
    # err_mag = np.abs(gt_mag - pred_mag)

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    fig.suptitle(f"Epoch {epoch} - Subaperture {sa_index} Analysis", fontsize=16)

    im0 = axes[0].imshow(gt_mag, cmap='magma', vmin=0, vmax=global_max)
    axes[0].set_title("Original Magnitude")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_mag, cmap='magma')
    axes[1].set_title("Reconstruction (Auto-scaled)")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(err_mag, cmap='viridis')
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

def sar_collate_fn(batch):
    """Converts the batch image size to the minimum shape that is divisible by 16."""
    xs, ys, metas = zip(*batch)
    min_w = min([img.shape[2] for img in xs])
    xs_aligned = [img[:, :, :min_w] for img in xs]
    ys_aligned = [target[:, :, :min_w] for target in ys]
    return torch.stack(xs_aligned), torch.stack(ys_aligned), metas

def complex_magnitude_loss(pred, target, alpha=0.5):
    """
    Combined loss for Real/Imaginary SAR data
    """
    mse_loss = nn.MSELoss()(pred, target)



def get_criterion(cfg):
    loss_type = cfg["training"]["loss_function"]
    params = cfg["training"]["loss_params"]

    if loss_type == "mse":
        return nn.MSELoss()

    elif loss_type == "mae":
        return nn.L1Loss()

    elif loss_type == "complex_magnitude":
        alpha = params.get("alpha", 0.5)

        def magnitude_weighted_loss(pred, target):
            mse_loss = nn.MSELoss()(pred, target)

            eps = 1e-8
            pred_real, pred_imag = pred[:, 0::1, ...], pred[:, 1::2, ...]
            true_real, true_imag = target[:, 0::1, ...], target[:, 1::2, ...]

            mag_pred = torch.sqrt(pred_real ** 2 + pred_imag ** 2 + eps)
            mag_true = torch.sqrt(true_real ** 2 + true_imag ** 2 + eps)

            mag_loss = nn.MSELoss()(mag_pred, mag_true)
            return (1 - alpha) * mse_loss + alpha * mag_loss
        return magnitude_weighted_loss

    else:
        raise ValueError(f"Unknown loss function: {loss_type}")



