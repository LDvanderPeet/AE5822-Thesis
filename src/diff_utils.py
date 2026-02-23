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


class DiffusionEngine:
    """
    Manages the forward and reverse diffusion processes for SAR image generation.

    Handles noise scheduling, the forward diffusion (adding noise), and the reverse diffusion step (denoising) using
    the DDPM formulation.

    Parameters
    ----------
    num_steps : int
        Total number of diffusion timesteps (T). Standard 1000 steps, as proposed in the paper:
        'Denoising Probabilistic Diffusion Models'.
    beta_start : float
        The initial variance scale for the first time step.
    beta_end : float
        The final variance scale for the last timestep.
    device : str
        The torch device (e.g., "cuda" or "cpu") for computation.
    """
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_steps = num_steps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, y_0, noise, t):
        """
        Applies forward diffusion to a clean sample.

        Uses the reparameterization trick to sample $y_t$ directly from $y_0$.

        Parameters
        ----------
        y_0 : torch.Tensor
            The ground truth target tensor.
        noise : torch.Tensor
            The Gaussian noise to be added.
        t : torch.Tensor
            The specific timesteps for each sample in the batch.
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * y_0 + sqrt_one_minus * noise

    def step(self, model_output, t, y_t):
        """
        Performs a single reverse diffusion step.

        Estimates $y_{t-1}$ from $y_t$ using the model's noise prediction.

        Parameters
        ----------
        model_output : torch.Tensor
            The noise predicted by the U-Net.
        t : int
            The current timestep index.
        y_t : torch.Tensor
            The noisy sample at the current timestep.

        Returns
        -------
        torch.Tensor
            The denoised sample for the previous timestep.
        """
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(self.alphas[t]).view(-1, 1, 1, 1)

        mean = (1 / sqrt_alpha_t) * (y_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * model_output)

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(y_t)
            variance = torch.sqrt(self.posterior_variance[t].view(-1, 1, 1, 1))
            return mean + variance * noise

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
    """
    Reverses the unit sphere normalization.

    Parameters
    ----------
    tensor : torch.Tensor
        The normalized data tensor.
    global_max : float
        The scalar used during the normalization phase. (Set at 4257 through manual inspection of max. value of data).

    Returns
    -------
    torch.Tensor
        The tensor scaled back to original units.
    """
    return tensor * global_max


def get_physical_magnitude(tensor, global_max=4257):
    """
    Calculates the physical magnitude from interleaved real/imaginary components.

    Parameters
    ----------
    tensor : torch.Tensor
        The normalized data tensor.
    global_max : float
        The scalar used during the normalization phase. (Set at 4257 through manual inspection of max. value of data).

    Returns
    -------
    torch.Tensor
        The physical magnitude of the complex SAR data.
    """
    re = tensor[:, 0::2, ...]
    im = tensor[:, 1::2, ...]

    mag_norm = torch.sqrt(re ** 2 + im ** 2)

    return mag_norm * global_max

@torch.no_grad()
def visualize_reconstruction(model, val_loader, device, epoch, viz_dir, engine=None, sa_index=0, global_max=4257, sample_idx=238):
    """
    Generates and saves comparison between ground truth and model predictions.

    Supports both standard regression and diffusion-based sampling. It calculated the physical magnitude for
    visualization and computes the absolute error map.

    Parameters
    ----------
    model : torch.nn.Module
        The trained SAR model.
    val_loader : torch.utils.data.DataLoader
        Validation data loader to pull a sample from.
    device : str
        The torch device (e.g., "cuda" or "cpu") for computation.
    epoch : int
        Current epoch number for labeling of filename.
    viz_dir : str
        Directory where the resulting .png will be saved.
    engine : DiffusionEngine, optional
        If provided, the model is treated as a diffusion model and undergoes sampling.
    sa_index : int
        The index of the subaperture to visualize.
    global_max : float
       The scalar used during the normalization phase. (Set at 4257 through manual inspection of max. value of data).
    sample_idx : int
        Specific dataset index to visualize for consistency.
    """
    model.eval()

    dataset = val_loader.dataset
    x, y, meta = dataset[sample_idx]

    # try:
    #     x, y, meta = next(iter(val_loader))
    # except StopIteration:
    #     return

    x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)

    if engine is not None:
        cur_y = torch.randn_like(y)

        for t in reversed(range(engine.num_steps)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            model_input = torch.cat([x, cur_y], dim=1)
            pred_noise = model(model_input, t_batch)
            cur_y = engine.step(pred_noise, t, cur_y)
        output = cur_y

    else:
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

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)', color='tab:blue')
    ax1.plot(df['epoch'], df['train_loss'].astype(float), label='Train Loss', color='tab:blue', linestyle='--')
    ax1.plot(df['epoch'], df['val_loss'].astype(float), label='Val Loss', color='tab:blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

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
    """
    Custom collate function for SAR patches of varying widths.

    Ensures that all items in a batch are cropped to the minimum width found within that batch, and that the width is
    compatible with architecture strides.

    Parameters
    ----------
    batch : list
        List of tuples (x, y, meta) from the Dataset.

    Returns
    -------
    tuple
        (stacked inputs, Stacked targets, List of metadata dicts).
    """
    xs, ys, metas = zip(*batch)
    min_w = min([img.shape[2] for img in xs])
    xs_aligned = [img[:, :, :min_w] for img in xs]
    ys_aligned = [target[:, :, :min_w] for target in ys]
    return torch.stack(xs_aligned), torch.stack(ys_aligned), metas


def get_criterion(cfg):
    """
    Factory function to initialize the training loss function.

    Supports "mse", "mae", and "complex_magnitude". TODO: implement more loss functions.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing 'training' parameters.

    Returns
    -------
    callable
        The loss function taking (pred, target) as arguments.
    """
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
            pred_real, pred_imag = pred[:, 0::2, ...], pred[:, 1::2, ...]
            true_real, true_imag = target[:, 0::2, ...], target[:, 1::2, ...]

            mag_pred = torch.sqrt(pred_real ** 2 + pred_imag ** 2 + eps)
            mag_true = torch.sqrt(true_real ** 2 + true_imag ** 2 + eps)

            mag_loss = nn.MSELoss()(mag_pred, mag_true)
            return (1 - alpha) * mse_loss + alpha * mag_loss
        return magnitude_weighted_loss

    else:
        raise ValueError(f"Unknown loss function: {loss_type}")



