# from __future__ import annotations
#
# import argparse
# import sys
# import yaml
# import json
# from pathlib import Path
#
# root_dir = Path(__file__).resolve().parents[1]
# if str(root_dir) not in sys.path:
#     sys.path.insert(0, str(root_dir))
#
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import pytorch_lightning as pl
#
# from data import PairedDataModule
# from src.PixelDiffusion import PixelDiffusionConditional
# from evaluate import load_config, build_model_from_config, to_complex_channels, compute_image_metrics
#
#
# def save_pure_sar_image(data: np.ndarray, save_dir: Path, filename: str) -> None:
#     if data.ndim == 3:
#         data = data[0]
#     if isinstance(data, torch.Tensor):
#         data = data.detach().cpu().numpy()
#
#     fig, ax = plt.subplots(figsize=(4, 4))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
#     ax.axis('off')
#     ax.imshow(data, cmap='gray', vmin=0.0, vmax=1.0, aspect='auto')
#     fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#
#
# def scale_for_diagram(mag: np.ndarray) -> np.ndarray:
#     vmin = float(np.percentile(mag, 1))
#     vmax = float(np.percentile(mag, 99))
#     if vmax <= vmin:
#         vmax = vmin + 1e-5
#     clipped = np.clip(mag, vmin, vmax)
#     return (clipped - vmin) / (vmax - vmin)
#
#
# def main() -> None:
#     parser = argparse.ArgumentParser(description="Run recursive ASR inference on a single tile.")
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--tile", type=int, default=790)
#     parser.add_argument("--iterations", type=int, default=5, help="Number of recursive passes.")
#     args = parser.parse_args()
#
#     config = load_config(args.config)
#     eval_cfg = config.get("evaluation", {})
#     checkpoint_path = eval_cfg.get("checkpoint_path")
#
#     output_dir = Path(eval_cfg.get("save_dir", "evaluation_results")) / f"recursive_tile_{args.tile:03d}"
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     pl.seed_everything(config.get("seed", 42), workers=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     datamodule = PairedDataModule.from_config(config)
#     datamodule.setup("test")
#     test_loader = datamodule.test_dataloader()
#     model = build_model_from_config(config, checkpoint_path, device)
#
#     total_evaluated_patches = 0
#     found_target = False
#     progression_metrics = {}
#
#     print(f"\n>>> Starting Recursive Inference for Tile {args.tile} ({args.iterations} iterations)...")
#
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(test_loader):
#             if found_target:
#                 break
#
#             deg_batch, ref_batch = batch[0].to(device), batch[1].to(device)
#             is_amp_only = deg_batch.shape[1] == 1
#
#             for i in range(ref_batch.shape[0]):
#                 if total_evaluated_patches == args.tile:
#
#                     ref_tensor = ref_batch[i].unsqueeze(0)
#                     current_condition = deg_batch[i].unsqueeze(0)
#
#                     ref_phys = model._inverse_signed_log_normalize(ref_tensor.float()).cpu()[0]
#                     ref_mag = ref_phys[0].numpy() if is_amp_only else np.abs(to_complex_channels(ref_phys)[0].numpy())
#
#                     save_pure_sar_image(scale_for_diagram(ref_mag), output_dir, "iter_ref_x0.png")
#
#                     deg_phys = model._inverse_signed_log_normalize(current_condition.float()).cpu()[0]
#                     deg_mag = deg_phys[0].numpy() if is_amp_only else np.abs(to_complex_channels(deg_phys)[0].numpy())
#                     save_pure_sar_image(scale_for_diagram(deg_mag), output_dir, "iter_00_degraded.png")
#
#                     psnr, rmse, mae, ssim = compute_image_metrics(ref_mag, deg_mag)
#                     progression_metrics["iter_00"] = {"PSNR": psnr, "SSIM": ssim, "RMSE": rmse}
#                     print(f"    -> Iteration 00 (Base Degraded) | PSNR: {psnr:.2f} | SSIM: {ssim:.3f}")
#
#                     for iter_num in range(1, args.iterations + 1):
#                         rec_tensor = model.predict_step((current_condition, ref_tensor), batch_idx)
#
#                         rec_phys = model._inverse_signed_log_normalize(rec_tensor.float()).cpu()[0]
#                         rec_mag = rec_phys[0].numpy() if is_amp_only else np.abs(
#                             to_complex_channels(rec_phys)[0].numpy())
#
#                         psnr, rmse, mae, ssim = compute_image_metrics(ref_mag, rec_mag)
#                         progression_metrics[f"iter_{iter_num:02d}"] = {"PSNR": psnr, "SSIM": ssim, "RMSE": rmse}
#
#                         save_pure_sar_image(scale_for_diagram(rec_mag), output_dir, f"iter_{iter_num:02d}.png")
#                         print(f"    -> Iteration {iter_num:02d} | PSNR: {psnr:.2f} | SSIM: {ssim:.3f}")
#
#                         current_condition = rec_tensor
#
#                     with open(output_dir / "recursive_metrics.json", "w") as f:
#                         json.dump(progression_metrics, f, indent=4)
#
#                     found_target = True
#                     break
#
#                 total_evaluated_patches += 1
#
#
# if __name__ == "__main__":
#     main()


from __future__ import annotations

import argparse
import sys
import yaml
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pytorch_lightning as pl

from data import PairedDataModule
from src.PixelDiffusion import PixelDiffusionConditional
from evaluate import (
    load_config,
    build_model_from_config,
    to_complex_channels,
    compute_image_metrics,
    point_target_analysis_single,
    local_interferometric_coherence
)


def save_pure_sar_image(data: np.ndarray, save_dir: Path, filename: str) -> None:
    if data.ndim == 3:
        data = data[0]
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    ax.axis('off')
    ax.imshow(data, cmap='gray', vmin=0.0, vmax=1.0, aspect='auto')
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def scale_for_diagram(mag: np.ndarray) -> np.ndarray:
    vmin = float(np.percentile(mag, 1))
    vmax = float(np.percentile(mag, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-5
    clipped = np.clip(mag, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin)


def plot_recursive_evolution(tracking_dict: dict, title: str, ylabel: str, save_path: Path, is_phase: bool = False,
                             x_ticks: np.ndarray = None) -> None:
    """
    Compiles tracked 1D profiles across iterations into a single highly legible scientific plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    num_iters = len([k for k in tracking_dict.keys() if "Iter" in k and "00" not in k])
    iter_count = 0

    for key, profile in tracking_dict.items():
        x = x_ticks if x_ticks is not None else np.arange(len(profile))

        if key == "Reference":
            ax.plot(x, profile, label=key, color='black', linewidth=2.5, zorder=10)
        elif "00" in key:
            ax.plot(x, profile, label="Iter 00 (Degraded)", color='blue', linestyle='-.', linewidth=2, zorder=9)
        else:
            # Create a gradient from light red to dark red to show progression
            color = cm.Reds(0.4 + 0.6 * (iter_count / max(1, num_iters - 1)))
            ax.plot(x, profile, label=key, color=color, linewidth=1.5, zorder=5 + iter_count)
            iter_count += 1

    ax.set_title(title)
    ax.set_xlabel("Range Axis [px]" if is_phase else "Spatial Window [Upsampled Pixels]")
    ax.set_ylabel(ylabel)

    if is_phase:
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recursive ASR inference on a single tile.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tile", type=int, default=790)
    parser.add_argument("--iterations", type=int, default=5, help="Number of recursive passes.")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    checkpoint_path = eval_cfg.get("checkpoint_path")

    output_dir = Path(eval_cfg.get("save_dir", "evaluation_results")) / f"recursive_tile_{args.tile:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.get("seed", 42), workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = PairedDataModule.from_config(config)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    model = build_model_from_config(config, checkpoint_path, device)

    total_evaluated_patches = 0
    found_target = False

    # Tracking Arrays for 1D Plots
    progression_metrics = {}
    tracking_az = {}
    tracking_rg = {}
    tracking_phase = {}

    print(f"\n>>> Starting Recursive Inference for Tile {args.tile} ({args.iterations} iterations)...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if found_target:
                break

            deg_batch, ref_batch = batch[0].to(device), batch[1].to(device)
            is_amp_only = deg_batch.shape[1] == 1

            for i in range(ref_batch.shape[0]):
                if total_evaluated_patches == args.tile:

                    # ==========================================
                    # 1. SETUP BASELINES
                    # ==========================================
                    ref_tensor = ref_batch[i].unsqueeze(0)
                    current_condition = deg_batch[i].unsqueeze(0)

                    ref_phys = model._inverse_signed_log_normalize(ref_tensor.float()).cpu()[0]
                    deg_phys = model._inverse_signed_log_normalize(current_condition.float()).cpu()[0]

                    ref_mag = ref_phys[0].numpy() if is_amp_only else np.abs(to_complex_channels(ref_phys)[0].numpy())
                    deg_mag = deg_phys[0].numpy() if is_amp_only else np.abs(to_complex_channels(deg_phys)[0].numpy())

                    ref_cplx = to_complex_channels(ref_phys)[0] if not is_amp_only else None
                    deg_cplx = to_complex_channels(deg_phys)[0] if not is_amp_only else None

                    save_pure_sar_image(scale_for_diagram(ref_mag), output_dir, "iter_ref_x0.png")
                    save_pure_sar_image(scale_for_diagram(deg_mag), output_dir, "iter_00_degraded.png")

                    # ==========================================
                    # 2. LOG ITERATION 00 (Baseline)
                    # ==========================================
                    psnr, rmse, mae, ssim = compute_image_metrics(ref_mag, deg_mag)

                    # Store IRF profiles
                    irf_00 = point_target_analysis_single(torch.tensor(deg_mag), torch.tensor(ref_mag),
                                                          torch.tensor(deg_mag))
                    tracking_az["Reference"] = irf_00["profiles"]["ref_azimuth"]
                    tracking_az["Iter 00"] = irf_00["profiles"]["deg_azimuth"]
                    tracking_rg["Reference"] = irf_00["profiles"]["ref_range"]
                    tracking_rg["Iter 00"] = irf_00["profiles"]["deg_range"]

                    az_res_00 = irf_00["deg_irf"]["azimuth"]["resolution_px"]

                    # Store Phase profiles
                    if not is_amp_only:
                        ref_cplx_np = ref_cplx.cpu().numpy()
                        deg_cplx_np = deg_cplx.cpu().numpy()

                        # Find the peak coordinate for the 1D phase slice
                        peak_idx = np.argmax(ref_mag)
                        py, px = np.unravel_index(peak_idx, ref_mag.shape)
                        start_px, end_px = max(0, px - 10), min(ref_mag.shape[-1], px + 11)
                        range_axis_ticks = np.arange(start_px, end_px)

                        tracking_phase["Reference"] = np.angle(ref_cplx_np[py, start_px:end_px])
                        tracking_phase["Iter 00"] = np.angle(deg_cplx_np[py, start_px:end_px])

                        phase_diff_00 = torch.angle(deg_cplx * torch.conj(ref_cplx))
                        mean_phase_00 = float(torch.mean(torch.abs(phase_diff_00)).item())
                        coh_00 = float(torch.mean(local_interferometric_coherence(deg_cplx, ref_cplx)).item())
                    else:
                        mean_phase_00, coh_00, range_axis_ticks, py = None, None, None, 0

                    progression_metrics["iter_00"] = {
                        "PSNR": psnr, "SSIM": ssim, "RMSE": rmse,
                        "Azimuth_Res_px": az_res_00, "Coherence_Gamma": coh_00, "Phase_Error_Rad": mean_phase_00
                    }

                    print(
                        f"    -> Iteration 00 (Base Degraded) | PSNR: {psnr:.2f} | SSIM: {ssim:.3f} | Az_Res: {az_res_00:.2f}px | Coh: {coh_00:.3f}")

                    # ==========================================
                    # 3. RECURSIVE LOOP
                    # ==========================================
                    for iter_num in range(1, args.iterations + 1):
                        rec_tensor = model.predict_step((current_condition, ref_tensor), batch_idx)

                        rec_phys = model._inverse_signed_log_normalize(rec_tensor.float()).cpu()[0]
                        rec_mag = rec_phys[0].numpy() if is_amp_only else np.abs(
                            to_complex_channels(rec_phys)[0].numpy())
                        rec_cplx = to_complex_channels(rec_phys)[0] if not is_amp_only else None

                        psnr, rmse, mae, ssim = compute_image_metrics(ref_mag, rec_mag)

                        # Store IRF Profiles
                        irf_rec = point_target_analysis_single(torch.tensor(deg_mag), torch.tensor(ref_mag),
                                                               torch.tensor(rec_mag))
                        tracking_az[f"Iter {iter_num:02d}"] = irf_rec["profiles"]["rec_azimuth"]
                        tracking_rg[f"Iter {iter_num:02d}"] = irf_rec["profiles"]["rec_range"]
                        az_res_rec = irf_rec["rec_irf"]["azimuth"]["resolution_px"]

                        # Store Phase Profiles
                        if not is_amp_only:
                            rec_cplx_np = rec_cplx.cpu().numpy()
                            tracking_phase[f"Iter {iter_num:02d}"] = np.angle(rec_cplx_np[py, start_px:end_px])

                            phase_diff = torch.angle(rec_cplx * torch.conj(ref_cplx))
                            mean_phase = float(torch.mean(torch.abs(phase_diff)).item())
                            coh = float(torch.mean(local_interferometric_coherence(rec_cplx, ref_cplx)).item())
                        else:
                            mean_phase, coh = None, None

                        progression_metrics[f"iter_{iter_num:02d}"] = {
                            "PSNR": psnr, "SSIM": ssim, "RMSE": rmse,
                            "Azimuth_Res_px": az_res_rec, "Coherence_Gamma": coh, "Phase_Error_Rad": mean_phase
                        }

                        save_pure_sar_image(scale_for_diagram(rec_mag), output_dir, f"iter_{iter_num:02d}.png")

                        print(
                            f"    -> Iteration {iter_num:02d}               | PSNR: {psnr:.2f} | SSIM: {ssim:.3f} | Az_Res: {az_res_rec:.2f}px | Coh: {coh:.3f}")

                        # Feed output back into condition
                        current_condition = rec_tensor

                    # ==========================================
                    # 4. EXPORT JSON AND PLOTS
                    # ==========================================
                    with open(output_dir / "recursive_metrics.json", "w") as f:
                        json.dump(progression_metrics, f, indent=4)

                    print("\n>>> Generating 1D Analytical Overlays...")
                    plot_recursive_evolution(tracking_az, "Azimuth Impulse Response Function (Recursive)", "Amplitude",
                                             output_dir / "plot_irf_azimuth.png")
                    plot_recursive_evolution(tracking_rg, "Range Impulse Response Function (Recursive)", "Amplitude",
                                             output_dir / "plot_irf_range.png")

                    if not is_amp_only:
                        plot_recursive_evolution(tracking_phase, f"1D Range Phase Profile (Azimuth Line {py})",
                                                 "Phase [Rad]", output_dir / "plot_phase_1d.png", is_phase=True,
                                                 x_ticks=range_axis_ticks)

                    print(f">>> Complete. Plots saved to: {output_dir}")
                    found_target = True
                    break

                total_evaluated_patches += 1


if __name__ == "__main__":
    main()