from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from data import PairedDataModule
from src.PixelDiffusion import PixelDiffusionConditional
from evaluate import load_config, build_model_from_config, to_complex_channels

from src.DenoisingDiffusionProcess.beta_schedules import get_beta_schedule


def save_pure_sar_image(data: np.ndarray, save_dir: Path, filename: str) -> None:
    """Saves a 2D matrix with absolute zero margin or padding borders."""
    if data.ndim == 3:
        data = data[0]
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    ax.axis('off')

    ax.imshow(data, cmap='gray', vmin=0.0, vmax=1.0, aspect='auto')

    full_path = save_dir / filename
    fig.savefig(full_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close(fig)


def scale_for_diagram(mag: np.ndarray) -> np.ndarray:
    """
    Applies a robust quantile cap to handle SAR dynamic range skew,
    ensuring consistent geometric contrast from t=0 to t=999.
    """
    vmin = float(np.percentile(mag, 1))
    vmax = float(np.percentile(mag, 99))

    if vmax <= vmin:
        vmax = vmin + 1e-5

    clipped = np.clip(mag, vmin, vmax)

    return (clipped - vmin) / (vmax - vmin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sequential multi-timestep noise states.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--tile", type=int, default=790, help="Target Tile_NUMBER.")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    checkpoint_path = eval_cfg.get("checkpoint_path")

    output_dir = Path(eval_cfg.get("save_dir", "evaluation_results")) / f"noise_progression_tile_{args.tile:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.get("seed", 42), workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = PairedDataModule.from_config(config)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    model = build_model_from_config(config, checkpoint_path, device)

    num_timesteps = config.get("model", {}).get("num_timesteps", 1000)
    schedule_type = config.get("model", {}).get("schedule", "cosine")

    betas = get_beta_schedule(schedule_type, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    target_timesteps = [0, 100, 200, 400, 600, 800, 900]
    total_evaluated_patches = 0
    found_target = False

    print(f"\n>>> Processing dataset stream for Tile_NUMBER {args.tile}...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if found_target:
                break

            deg_batch, ref_batch = batch[0].to(device), batch[1].to(device)
            is_amp_only = deg_batch.shape[1] == 1

            for i in range(ref_batch.shape[0]):
                if total_evaluated_patches == args.tile:
                    print(f">>> Found target tile! Simulating forward process timeline...")

                    x0_log = ref_batch[i].unsqueeze(0).float().to(device)
                    x0_phys = model._inverse_signed_log_normalize(x0_log)

                    if is_amp_only:
                        x0_mag = x0_phys[0].cpu().numpy()
                    else:
                        x0_mag = np.abs(to_complex_channels(x0_phys.cpu())[0].numpy())

                    x0_mag_tensor = torch.from_numpy(x0_mag).to(device)
                    noise = torch.randn_like(x0_mag_tensor) * torch.std(x0_mag_tensor)

                    for t in target_timesteps:
                        if t == 0:
                            xt_mag = x0_mag
                        else:
                            alpha_bar = alphas_cumprod[t]
                            xt_final = torch.sqrt(alpha_bar) * x0_mag_tensor + torch.sqrt(1.0 - alpha_bar) * noise
                            xt_mag = xt_final.cpu().numpy()

                        filename = f"step_t_{t:03d}.png"
                        save_pure_sar_image(scale_for_diagram(xt_mag), output_dir, filename)
                        print(f"    -> Exported: {filename}")

                    print(f"\n>>> Sequence complete. Clean textures stored at: {output_dir}")
                    found_target = True
                    break

                total_evaluated_patches += 1

    if not found_target:
        print(f">>> Error: Tile_NUMBER {args.tile} not found in the test dataset split.")


if __name__ == "__main__":
    main()