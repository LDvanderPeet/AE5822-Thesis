from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import gaussian_kde, wasserstein_distance
import scipy.stats as stats

from data import PairedDataModule
from src.PixelDiffusion import PixelDiffusionConditional


EPS = 1e-12


@dataclass
class IRFMetrics:
    resolution_px: float
    pslr_db: float
    islr_db: float


@dataclass
class IRFResult:
    azimuth: IRFMetrics
    range_: IRFMetrics


def generate_and_save_scenes(
        deg_cplx: torch.Tensor,
        ref_cplx: torch.Tensor,
        rec_cplx: torch.Tensor,
        save_path: Path,
        sample_idx: int
) -> None:
    """
    Generates high-res LaTeX comparative plots for visual analysis.

    1. Generates the amplitude plots for degradation, reconstruction, and reference. Scales amplitude using a dynamic
    range threshold ($\mu$ + 3 $\sigma$) and calculates spatial absolute errors.
    2. Calculates phase or degradation, reconstruction, and reference

    Parameters
    ----------
    deg_cplx : torch.Tensor
        The degraded complex-valued SAR subaperture array.
    ref_cplx : torch.Tensor
        The reference/ground truth complex-valued SAR subaperture array.
    rec_cplx : torch.Tensor
        The reconstructed complex-valued model prediction array.
    save_path : Path
        Base directory where output images will be stored.
    sample_idx : int
        The index of the data sample (used for unique patch naming).
    """
    if deg_cplx.ndim == 3:
        deg_cplx = deg_cplx[0]
    if ref_cplx.ndim == 3:
        ref_cplx = ref_cplx[0]
    if rec_cplx.ndim == 3:
        rec_cplx = rec_cplx[0]

    deg = deg_cplx.numpy()
    ref = ref_cplx.numpy()
    rec = rec_cplx.numpy()

    amp_deg_raw = np.abs(deg)
    amp_ref_raw = np.abs(ref)
    amp_rec_raw = np.abs(rec)

    def scale_for_plot(mag: np.ndarray) -> np.ndarray:
        mean = np.mean(mag)
        std = np.std(mag)
        vmax = float(mean + 3.0 * std)
        if vmax <= 0.0:
            vmax = 1.0
        return np.clip(mag / vmax, 0.0, 1.0)

    amp_deg_scaled = scale_for_plot(amp_deg_raw)
    amp_ref_scaled = scale_for_plot(amp_ref_raw)
    amp_rec_scaled = scale_for_plot(amp_rec_raw)

    amp_deg_error = np.abs(amp_ref_scaled - amp_deg_scaled)
    amp_rec_error = np.abs(amp_ref_scaled - amp_rec_scaled)
    vmax_error = max(np.percentile(amp_deg_error, 99), np.percentile(amp_rec_error, 99))
    if vmax_error <= 0.0:
        vmax_error = 1.0

    interferogram_deg = deg * np.conj(ref)
    interferogram_rec = rec * np.conj(ref)
    phase_err_deg = np.angle(interferogram_deg)
    phase_err_rec = np.angle(interferogram_rec)

    w = 5
    kernel = np.ones((w, w)) / (w * w)
    import scipy.signal
    def local_mean(img):
        return scipy.signal.convolve2d(img, kernel, mode='same')

    e_deg = local_mean(amp_deg_raw ** 2)
    e_ref = local_mean(amp_ref_raw ** 2)
    e_rec = local_mean(amp_rec_raw ** 2)

    coh_deg = np.clip(np.abs(local_mean(interferogram_deg)) / np.sqrt(np.clip(e_ref * e_deg, 1e-12, None)), 0.0, 1.0)
    coh_rec = np.clip(np.abs(local_mean(interferogram_rec)) / np.sqrt(np.clip(e_ref * e_rec, 1e-12, None)), 0.0, 1.0)

    # Find the brightest pixel using the raw, unclipped reference amplitude
    peak_idx = np.argmax(amp_ref_raw)
    py, px = np.unravel_index(peak_idx, amp_ref_raw.shape)

    start_px = max(0, px-5)
    end_px = min(amp_ref_raw.shape[-1], px + 6)
    range_axis_ticks = np.arange(start_px, end_px)

    phase_deg_window = np.angle(deg[py, start_px:end_px])
    phase_ref_window = np.angle(ref[py, start_px:end_px])
    phase_rec_window = np.angle(rec[py, start_px:end_px])

    patch_dir = save_path / f"patch_{sample_idx:03d}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    def save_single_map(data: np.ndarray, filename: str, cmap: str, vmin: float, vmax: float,
                        cbar_label: str, ticks=None, ticklabels=None):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("Range [px]")
        ax.set_ylabel("Azimuth [px]")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=16)
        if ticks is not None:
            cbar.set_ticks(ticks)
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)

        fig.savefig(patch_dir / filename, dpi=300, bbox_inches='tight', transparent=False)
        plt.close(fig)

    save_single_map(amp_ref_scaled, "01a_amp_reference.png", cmap='gray', vmin=0.0, vmax=1.0,
                    cbar_label="Scaled Amplitude")
    save_single_map(amp_deg_scaled, "01b_amp_degraded.png", cmap='gray', vmin=0.0, vmax=1.0,
                    cbar_label="Scaled Amplitude")
    save_single_map(amp_rec_scaled, "01c_amp_reconstructed.png", cmap='gray', vmin=0.0, vmax=1.0,
                    cbar_label="Scaled Amplitude")

    save_single_map(amp_deg_error, "02a_amp_error_degraded.png", cmap='hot', vmin=0.0, vmax=vmax_error,
                   cbar_label="Absolute Delta")
    save_single_map(amp_rec_error, "02b_amp_error_reconstructed.png", cmap='hot', vmin=0.0, vmax=vmax_error,
                    cbar_label="Absolute Delta")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range_axis_ticks, phase_ref_window, label="Reference Phase",
            color='black', marker='o', linewidth=1.5, markersize=6)
    ax.plot(range_axis_ticks, phase_deg_window, label="Degraded Phase",
            color='blue', linestyle='-', marker='s', linewidth=1.5, markersize=6)
    ax.plot(range_axis_ticks, phase_rec_window, label="Reconstructed Phase",
            color='red', linestyle='--', marker='x', linewidth=1.5, markersize=6)
    ax.set_xlabel("Range [px]")
    ax.set_xlim(start_px - 0.5, end_px - 0.5)
    ax.set_ylabel("Phase [Rad]")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
    ax.set_title(f"Range Phase Profile at Azimuth {py}")
    ax.legend(loc='upper right')
    fig.savefig(patch_dir / "04_phase_profile_1d.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    save_single_map(phase_err_deg, "04a_phase_error_degraded.png", cmap='twilight', vmin=-np.pi, vmax=np.pi,
                    cbar_label="Error [Rad]", ticks=[-np.pi, 0, np.pi], ticklabels=['$-\pi$', '0', '$\pi$'])
    save_single_map(phase_err_rec, "04b_phase_error_reconstructed.png", cmap='twilight', vmin=-np.pi, vmax=np.pi,
                    cbar_label="Error [Rad]", ticks=[-np.pi, 0, np.pi], ticklabels=['$-\pi$', '0', '$\pi$'])

    save_single_map(coh_deg, "05a_coherence_degraded.png", cmap='viridis', vmin=0.0, vmax=1.0,
                    cbar_label="Coherence $\gamma$")
    save_single_map(coh_rec, "05b_coherence_reconstructed.png", cmap='viridis', vmin=0.0, vmax=1.0,
                    cbar_label="Coherence $\gamma$")

    plt.rcParams.update(plt.rcParamsDefault)


def evaluate_hypothesis_ratio(values: list[float], margin: float = 0.05, conf: float = 0.95) -> dict[str, Any]:
    """
    Performs a two-sided t-test to evaluate confidence intervals for a metric ratio.

    Determines whether the $(1 - \alpha)$ confidence interval of a reconstructed/ground truth
    ratio falls entirely within a specified acceptable boundary ($1 \pm \text{margin}$). Used
    specifically to rigorously test the 5% margin resolution preservation hypothesis.

    Parameters
    ----------
    values : list of float
        Ratios of the evaluated feature (e.g., $\text{Resolution}_{\text{pred}} / \text{Resolution}_{\text{gt}}$).
    margin : float, default=0.05
        The maximum allowed structural deviation from unity (e.g., 0.05 for a 5% limit).
    conf : float, default=0.95
        The statistical confidence level (e.g., 0.95 for a 95% CI).

    Returns
    -------
    dict
        A summary containing the pass/fail boolean, mean ratio value, lower/upper
        confidence bounds, and target margin.
    """
    if not values:
        return {"passed": False, "error": "No data"}
    data = np.array(values)
    mean = np.mean(data)
    sem = stats.sem(data)
    moe = sem * stats.t.ppf((1 + conf) / 2., len(data) - 1)
    lower_bound = mean - moe
    upper_bound = mean + moe
    passed = (lower_bound >= 1.0 - margin) and (upper_bound <= 1.0 + margin)
    return {
        "passed": bool(passed),
        "mean_ratio": float(mean),
        "ci_lower": float(lower_bound),
        "ci_upper": float(upper_bound),
        "margin_target": f"+/- {margin*100}%"
    }


def evaluate_hypothesis_threshold(values: list[float], max_threshold: float, conf: float = 0.95) -> dict[str, Any]:
    """
    Evaluates if the upper bound of a metric's confidence interval falls below a maximum threshold.

    Utilizes a Student's t-distribution to establish an upper bound at the specified
    confidence level, verifying that reconstruction errors do not exceed severe degradation limits.

    Parameters
    ----------
    values : list of float
        Metric errors collected across the evaluation dataset (e.g., standard deviation of phase error).
    max_threshold : float
        The maximum allowable upper-bound value (e.g., 5.0 degrees for phase consistency).
    conf : float, default=0.95
        The statistical confidence level.

    Returns
    -------
    dict
        A evaluation log containing the pass/fail result, mean metric value, estimated
        upper confidence bound, and targeted max threshold limit.
    """
    if not values:
        return {"passed": False, "error": "No data"}
    data = np.array(values)
    mean = np.mean(data)
    sem = stats.sem(data)
    moe = sem * stats.t.ppf((1 + conf) / 2., len(data) - 1)
    upper_bound = mean + moe
    passed = upper_bound <= max_threshold
    return {
        "passed": bool(passed),
        "mean_value": float(mean),
        "upper_bound": float(upper_bound),
        "threshold": float(max_threshold),
    }


def load_config(config_path: str) -> dict[str, Any]:
    """
    Parses an experiment runtime configuration file.

    Parameters
    ----------
    config_path : str
        The local path string pointing to a `config.yaml` file.

    Returns
    -------
    dict
        The parsed dictionary of configuration groups.
    """
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model_from_config(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> PixelDiffusionConditional:
    """
    Constructs and restores a PixelDiffusionConditional model instance using saved weights.

    Extracts architectural parameters (U-Net channel multipliers, EMA flags, scheduling parameters)
    from the dictionary configuration and maps inputs to designated condition index labels.

    Parameters
    ----------
    config : dict
        The loaded comprehensive project configuration map.
    checkpoint_path : str
        The local system file system path pointing to a PyTorch Lightning checkpoint file (`.ckpt`).
    device : torch.device
        Target runtime device structure for computational mapping (`cuda` or `cpu`).

    Returns
    -------
    PixelDiffusionConditional
        An instantiated, evaluation-ready generative model.
    """
    model_cfg = config.get("model", {})
    opt_cfg = config.get("optimization", {})
    lr_sched_cfg = opt_cfg.get("reduce_lr_on_plateau", {})
    unet_cfg = model_cfg.get("unet", {})
    ema_cfg = model_cfg.get("ema", {})
    data_cfg = config.get("data", {})
    sa_cfg = data_cfg.get("subaperture_config", {}) if isinstance(data_cfg, dict) else {}
    input_indices = sa_cfg.get("input_indices", [])
    input_condition_labels = [f"SA{int(idx)}" if int(idx) > 0 else "FA" for idx in input_indices] # TODO: update naming

    model = PixelDiffusionConditional.load_from_checkpoint(
        checkpoint_path,
        condition_channels=model_cfg.get("in_channels", 2),
        generated_channels=model_cfg.get("out_channels", 2),
        input_condition_labels=input_condition_labels if input_condition_labels else None,
        target_label="FA",
        num_timesteps=model_cfg.get("num_timesteps", 1000),
        schedule=model_cfg.get("schedule", "linear"),
        noise_offset=model_cfg.get("noise_offset", 0.0),
        loss_fn=model_cfg.get("loss_fn", "mse"),
        hybrid_base_loss=model_cfg.get("hybrid_base_loss", "mse"),
        hybrid_ms_ssim_t_limt=model_cfg.get("hybrid_ms_ssim_t_limit", 10),
        model_dim=unet_cfg.get("dim", 64),
        model_dim_mults=tuple(unet_cfg.get("dim_mults", [1, 2, 4, 8])),
        model_channels=unet_cfg.get("channels"),
        model_out_dim=unet_cfg.get("out_dim"),
        lr=opt_cfg.get("lr", 1e-3),
        lr_scheduler_factor=lr_sched_cfg.get("factor", 0.5),
        lr_scheduler_patience=lr_sched_cfg.get("patience", 10),
        ema_enabled=ema_cfg.get("enabled", False),
        ema_beta=ema_cfg.get("beta", 0.9999),
        ema_update_every=ema_cfg.get("update_every", 1),
        ema_update_after_step=ema_cfg.get("update_after_step", 0),
        map_location=device,
    )
    model.eval().to(device)
    return model


def to_complex_channels(t: torch.Tensor) -> torch.Tensor:
    """
    Transforms multi-channel interleaved real representations into native complex tensors.

    Unpacks alternate channel indices ($0::2$ as real, $1::2$ as imaginary) back
    into a matching single complex-valued matrix array structure.

    Parameters
    ----------
    t : torch.Tensor
        Real-valued tensor with size $(C, H, W)$ or complex tensor.

    Returns
    -------
    torch.Tensor
        A complex-valued tensor of shape $(C/2, H, W)$ containing native complex elements.
    """
    if torch.is_complex(t):
        return t
    if t.shape[0] >= 2 and t.shape[0] % 2 == 0:
        real = t[0::2]
        imag = t[1::2]
        return torch.complex(real, imag)
    return torch.complex(t, torch.zeros_like(t))


def magnitude_map(t: torch.Tensor) -> torch.Tensor:
    """
    Extracts the structural physical magnitude array from a complex or interleaved tensor.

    Parameters
    ----------
    t : torch.Tensor
        Input spatial representation array tensor.

    Returns
    -------
    torch.Tensor
        The calculated physical magnitude matrix ($|S| = \sqrt{Re^2 + Im^2}$).
    """
    cplx = to_complex_channels(t)
    mag = torch.abs(cplx)
    return mag[0] if mag.ndim == 3 else mag


def _resolution_3db(profile: np.ndarray) -> float:
    """
    Calculates the -3dB spatial width (main lobe width) of a 1D point-target response slice.

    Determines the system's geometric resolution boundaries by tracking the left and
    right spatial pixel thresholds relative to the main lobe peak down to the half-power level ($1/\sqrt{2}$).

    Parameters
    ----------
    profile : np.ndarray
        A 1D spatial intensity row or column profile array centered on a point-scatterer target.

    Returns
    -------
    float
        The full width at half maximum (FWHM) spatial metric value measured in pixel units.
    """
    p = profile.astype(np.float64)
    peak_idx = int(np.argmax(p))
    peak_val = max(p[peak_idx], EPS)
    thr = peak_val / np.sqrt(2.0)
    left = peak_idx
    while left > 0 and p[left] >= thr:
        left -= 1
    right = peak_idx
    while right < len(p) - 1 and p[right] >= thr:
        right += 1
    return float(max(1, right - left))


def _pslr_islr(profile: np.ndarray, main_lobe_width: float) -> tuple[float, float]:
    """
    Extracts Peak Side Lobe Ratio (PSLR) and Integrated Side Lobe Ratio (ISLR).

    Isolates the core target resolution zone (main lobe) from background energy
    dispersions (side lobes) to evaluate reconstruction sidelobe focusing capabilities.

    Parameters
    ----------
    profile : np.ndarray
        A 1D spatial intensity target row or column profile array.
    main_lobe_width : float
        The bounded width dimension of the main lobe lobe area.

    Returns
    -------
    pslr_db : float
        Peak Side Lobe Ratio in decibels. Measures relative peak side energy.
    islr_db : float
        Integrated Side Lobe Ratio in decibels. Measures total side-lobe integration energy.
    """
    p = profile.astype(np.float64)
    peak_idx = int(np.argmax(p))
    peak_val = max(p[peak_idx], EPS)
    half = max(1, int(round(main_lobe_width / 2.0)))
    lo, hi = max(0, peak_idx - half), min(len(p), peak_idx + half + 1)
    main = p[lo:hi]
    side = np.concatenate([p[:lo], p[hi:]]) if lo > 0 or hi < len(p) else np.array([EPS])
    pslr_db = 20.0 * np.log10(max(side.max(), EPS) / peak_val)
    islr_db = 10.0 * np.log10(max(np.sum(side ** 2), EPS) / max(np.sum(main ** 2), EPS))
    return float(pslr_db), float(islr_db)


def upsample_irf_window(window: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """
    Applies high-resolution bicubic interpolation to a localized target analysis grid.

    Improves measurement granularity for sub-pixel main-lobe width assessments.

    Parameters
    ----------
    window : torch.Tensor
        A clipped spatial intensity window surrounding a target peak signature.
    factor : int, default=8
        The integer grid upsampling factor multiplier.

    Returns
    -------
    torch.Tensor
        The upsampled patch tensor.
    """
    return F.interpolate(window.unsqueeze(0).unsqueeze(0), scale_factor=factor, mode="bicubic", align_corners=False)[0, 0]


def point_target_analysis_single(
    deg_mag: torch.Tensor,
    ref_mag: torch.Tensor,
    rec_mag: torch.Tensor,
    window_radius: int = 8,
    upsample_factor: int = 8,
) -> dict[str, Any]:
    """
    Executes a complete 2D point target Impulse Response Function (IRF) analysis.

    Locates the brightest point in the ground truth amplitude patch, extracts a local bounding box
    across both prediction and reference targets, upsamples the arrays, and returns
    comprehensive comparative measurements for geometric resolution, PSLR, and ISLR along
    the azimuth and range dimensions.

    Parameters
    ----------
    deg_mag : torch.Tensor
        The magnitude image map extracted from the degraded complex data.
    ref_mag : torch.Tensor
        The magnitude image map extracted from the ground truth complex data.
    rec_mag : torch.Tensor
        The magnitude image map extracted from the reconstructed model prediction.
    window_radius : int, default=8
        The pixel window radius clipped around the coordinate peak.
    upsample_factor : int, default=8
        The interpolative scale used to increase spatial dimension visibility.

    Returns
    -------
    dict
        An extensive diagnostic payload comprising key coordinate coordinates,
        independent dimension metric models, and 1D profile lists.
    """
    h, w = ref_mag.shape
    peak_idx = torch.argmax(ref_mag)
    py = int((peak_idx // w).item())
    px = int((peak_idx % w).item())

    y0 = max(0, py - window_radius)
    y1 = min(h, py + window_radius + 1)
    x0 = max(0, px - window_radius)
    x1 = min(w, px + window_radius + 1)

    deg_win = deg_mag[y0:y1, x0:x1]
    ref_win = ref_mag[y0:y1, x0:x1]
    rec_win = rec_mag[y0:y1, x0:x1]

    deg_up = upsample_irf_window(deg_win, factor=upsample_factor)
    ref_up = upsample_irf_window(ref_win, factor=upsample_factor)
    rec_up = upsample_irf_window(rec_win, factor=upsample_factor)

    def extract_profile(up_win):
        idx = torch.argmax(up_win)
        y, x = int((idx // up_win.shape[1]).item()), int((idx % up_win.shape[1]).item())
        return up_win[y, :].detach().cpu().numpy(), up_win[:, x].detach().cpu().numpy()

    deg_az, deg_rg = extract_profile(deg_up)
    ref_az, ref_rg = extract_profile(ref_up)
    rec_az, rec_rg = extract_profile(rec_up)

    def profile_metrics(profile: np.ndarray) -> IRFMetrics:
        res = _resolution_3db(profile)
        pslr, islr = _pslr_islr(profile, main_lobe_width=res)
        return IRFMetrics(resolution_px=res / upsample_factor, pslr_db=pslr, islr_db=islr)

    return {
        "ref_peak_xy": [py, px],
        "crop_bounds": [y0, y1, x0, x1],
        "ref_irf": {"azimuth": profile_metrics(ref_az).__dict__, "range": profile_metrics(ref_rg).__dict__},
        "deg_irf": {"azimuth": profile_metrics(deg_az).__dict__, "range": profile_metrics(deg_rg).__dict__},
        "rec_irf": {"azimuth": profile_metrics(rec_az).__dict__, "range": profile_metrics(rec_rg).__dict__},
        "profiles": {
            "ref_azimuth": ref_az.tolist(), "ref_range": ref_rg.tolist(),
            "deg_azimuth": deg_az.tolist(), "deg_range": deg_rg.tolist(),
            "rec_azimuth": rec_az.tolist(), "rec_range": rec_rg.tolist(),
        },
    }


def aggregate_irf(irf_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Computes statistical ensemble distributions across point target results.

    Parameters
    ----------
    irf_results : list of dict
        A compiled stack of individual point target results from `point_target_analysis_single`.

    Returns
    -------
    dict
        A restructured dictionary storing means and standard deviations categorized by target type,
        axis, and specific metric keys.
    """
    if not irf_results: return {}

    def collect(path: tuple[str, str, str]) -> list[float]:
        return [item[path[0]][path[1]][path[2]] for item in irf_results]

    summary = {}
    for group in ["ref_irf", "deg_irf", "rec_irf"]:
        summary[group] = {}
        for axis in ["azimuth", "range"]:
            summary[group][axis] = {}
            for metric in ["resolution_px", "pslr_db", "islr_db"]:
                vals = collect((group, axis, metric))
                summary[group][axis][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


def save_irf_plots(irf_results: list[dict[str, Any]], save_dir: Path, max_plots: int = 5) -> None:
    """
    Exports combined verification line plots for azimuth and range profiles.

    Parameters
    ----------
    irf_results : list of dict
        A compiled stack of individual point target results.
    save_dir : Path
        Output destination system path.
    max_plots : int, default=5
        The absolute maximum limit of plot graphics saved.
    """
    for i, res in enumerate(irf_results[:max_plots]):
        prof = res["profiles"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax_idx, axis in enumerate(["azimuth", "range"]):
            axes[ax_idx].plot(prof[f"ref_{axis}"], label="Reference", color='black', linewidth=1.5)
            axes[ax_idx].plot(prof[f"deg_{axis}"], label="Degraded", color='blue', linestyle='-.', linewidth=1.5)
            axes[ax_idx].plot(prof[f"rec_{axis}"], label="Reconstruction", color='red', linestyle='--', linewidth=1.5)
            axes[ax_idx].set_title(f"{axis.capitalize()} IRF Profile")
            axes[ax_idx].set_xlabel("Pixel")
            axes[ax_idx].set_ylabel("Amplitude")
            axes[ax_idx].legend()

        fig.tight_layout()
        fig.savefig(save_dir / f"irf_profiles_sample_{i:03d}.png", dpi=160)
        plt.close(fig)


def kde_distribution_distance(
    deg_values: np.ndarray,
    ref_values: np.ndarray,
    rec_values: np.ndarray,
    save_dir: Path,
) -> dict[str, float]:
    """
    Evaluates statistical similarity between predicted and ground truth intensity distributions.

    Fits Gaussian Kernel Density Estimates (KDE) to log-transformed magnitude values ($\log_{1p}(|SLC|)$)
    and measures the Wasserstein Distance ($W_1$) and Kullback-Leibler (KL) Divergence to quantify how well
    the model preserves speckle statistics and structural features.

    Parameters
    ----------
    deg_values : np.ndarray
        Flattened log-magnitude degraded array elements.
    ref_values : np.ndarray
        Flattened log-magnitude reference array elements.
    rec_values : np.ndarray
        Flattened log-magnitude predicted array elements.
    save_dir : Path
        Output path destination directory for plot storage.

    Returns
    -------
    dict
        A metric breakdown recording `wasserstein` and `kl_divergence` values.
    """
    deg_values = np.asarray(deg_values, dtype=np.float64)
    ref_values = np.asarray(ref_values, dtype=np.float64)
    rec_values = np.asarray(rec_values, dtype=np.float64)

    lo = float(min(ref_values.min(), rec_values.min(), deg_values.min()))
    hi = float(max(ref_values.max(), rec_values.max(), deg_values.max()))
    grid = np.linspace(lo, hi, 512)

    kde_deg, kde_ref, kde_rec = gaussian_kde(deg_values), gaussian_kde(ref_values), gaussian_kde(rec_values)
    pdf_deg, pdf_ref, pdf_rec = np.clip(kde_deg(grid), EPS, None), np.clip(kde_ref(grid), EPS, None), np.clip(
        kde_rec(grid), EPS, None)

    pdf_deg /= np.trapz(pdf_deg, grid)
    pdf_ref /= np.trapz(pdf_ref, grid)
    pdf_rec /= np.trapz(pdf_rec, grid)

    w1_deg, w1_rec = float(wasserstein_distance(ref_values, deg_values)), float(
        wasserstein_distance(ref_values, rec_values))
    kl_deg = float(np.trapz(pdf_ref * np.log(pdf_ref / pdf_deg), grid))
    kl_rec = float(np.trapz(pdf_ref * np.log(pdf_ref / pdf_rec), grid))

    plt.figure(figsize=(8, 5))
    plt.plot(grid, pdf_ref, label="Reference KDE", color='black')
    plt.plot(grid, pdf_deg, label="Degraded KDE", color='blue', linestyle='-.')
    plt.plot(grid, pdf_rec, label="Reconstruction KDE", color='red', linestyle='--')
    plt.xlabel("log1p(|SLC|)")
    plt.ylabel("Density")
    plt.title("KDE of Log-Magnitude Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "kde_log_magnitude.png", dpi=160)
    plt.close()

    return {
        "degraded_vs_ref": {"wasserstein": w1_deg, "kl_divergence": kl_deg},
        "reconstructed_vs_ref": {"wasserstein": w1_rec, "kl_divergence": kl_rec}
    }


def local_interferometric_coherence(
    rec_cplx: torch.Tensor,
    ref_cplx: torch.Tensor,
    window_size: int = 5,
) -> torch.Tensor:
    """
    Computes local complex interferometric coherence ($\gamma$) within a moving window.

    Measures the phase and amplitude correlation between prediction and reference complex data:
    $$\gamma = \frac{|\langle S_{\text{rec}} S_{\text{ref}}^* \rangle|}{\sqrt{\langle |S_{\text{rec}}|^2 \rangle \langle |S_{\text{ref}}|^2 \rangle}}$$

    Parameters
    ----------
    rec_cplx : torch.Tensor
        The native complex model reconstruction tensor.
    ref_cplx : torch.Tensor
        The native complex ground truth tensor.
    window_size : int, default=5
        The spatial length dimensions used for the moving local mean kernel window.

    Returns
    -------
    torch.Tensor
        A 2D spatial image matrix mapping coherence magnitude bounds from 0.0 to 1.0.
    """
    if rec_cplx.ndim == 3:
        rec_cplx = rec_cplx[0]
    if ref_cplx.ndim == 3:
        ref_cplx = ref_cplx[0]

    interferogram = rec_cplx * torch.conj(ref_cplx)
    rec_pow = torch.abs(rec_cplx) ** 2
    ref_pow = torch.abs(ref_cplx) ** 2

    kernel = torch.ones((1, 1, window_size, window_size), device=rec_cplx.device, dtype=rec_pow.dtype)
    norm = float(window_size * window_size)

    def mean_filter(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x.unsqueeze(0).unsqueeze(0), kernel, padding=window_size // 2) / norm

    e_int_re = mean_filter(interferogram.real)
    e_int_im = mean_filter(interferogram.imag)
    e_rec = mean_filter(rec_pow)
    e_ref = mean_filter(ref_pow)

    num = torch.sqrt(e_int_re ** 2 + e_int_im ** 2)
    den = torch.sqrt(torch.clamp(e_rec * e_ref, min=EPS))
    coh = num / torch.clamp(den, min=EPS)
    return coh.squeeze(0).squeeze(0)


def equivalent_number_of_looks(cplx: torch.Tensor) -> float:
    """
    Estimates the Equivalent Number of Looks (ENL) of a complex SAR patch region.

    Quantifies speckle reduction and radiometric resolution performance by evaluating the
    intensity distribution statistics:
    $$\text{ENL} = \frac{\mu_I^2}{\sigma_I^2}$$

    Parameters
    ----------
    cplx : torch.Tensor
        A complex native channel array tensor patch.

    Returns
    -------
    float
        The estimated ENL parameter value.
    """
    intensity = torch.abs(cplx) ** 2
    intensity = torch.to(torch.float64).flatten()

    mean_i = torch.mean(intensity)
    var_i = torch.var(intensity, unbiased=False)
    enl = (mean_i ** 2) / torch.clamp(var_i, min=EPS)
    return float(enl.item())


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    """
    Main orchestration function running the full SAR evaluation suite on the test dataset.

    Executes a structured loop that handles model inference, converts outputs back into native
    complex representations, and computes all requested metrics (IRF parameters, phase coherence,
    ENL matching, and KDE distances). Finally, it runs confidence interval hypothesis tests
    for resolution and phase constraints, exporting the aggregated results to `metrics.json`.

    Parameters
    ----------
    config : dict
        The global configuration dictionary loaded via the runtime configuration parser.

    Returns
    -------
    dict
        A dictionary containing structured metric stats and pass/fail hypothesis results.
    """
    eval_cfg = config.get("evaluation", {})
    tests = eval_cfg.get("tests", {})

    checkpoint_path = eval_cfg.get("checkpoint_path")
    if not checkpoint_path: raise ValueError("Missing evaluation.checkpoint_path in config.")

    save_dir = Path(eval_cfg.get("save_dir", "evaluation_results"))
    save_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.get("seed", 42), workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = PairedDataModule.from_config(config)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    print("\n" + "=" * 60)
    print(">>> LIVE DATA DIAGNOSTIC")
    print(f"Target Root Folder: {config['data']['root_dir']}")
    print(f"Total files detected in Dataset: {len(datamodule.test_dataset)}")
    print(f"Total batches in Loader: {len(test_loader)}")
    print("=" * 60 + "\n")

    model = build_model_from_config(config, checkpoint_path, device)

    run_scenes = bool(tests.get("generate_scenes", False))
    run_point_target = bool(tests.get("point_target_analysis", False))
    run_kde = bool(tests.get("kde_distance", False))
    run_phase = bool(tests.get("phase_coherence", False))
    run_enl = bool(tests.get("enl_comparison", False))

    scene_counter = 0
    max_scenes = 5
    irf_results = []
    max_kde_samples = int(eval_cfg.get("max_kde_samples", 400_000))

    ref_logmag, deg_logmag, rec_logmag = [], [], []

    phase_stats = {
        "degraded": {"mean_abs_phase_diff": [], "coherence_mean": [], "coherence_std": [], "std_phase_diff_rad": []},
        "reconstructed": {"mean_abs_phase_diff": [], "coherence_mean": [], "coherence_std": [],
                          "std_phase_diff_rad": []}
    }
    enl_stats = {
        "ref_enl": [], "deg_enl": [], "deg_abs_error": [], "deg_rel_error": [],
        "rec_enl": [], "rec_abs_error": [], "rec_rel_error": []
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            deg_batch, ref_batch = batch
            deg_batch, ref_batch = deg_batch.to(device, non_blocking=True), ref_batch.to(device, non_blocking=True)
            rec_batch = model.predict_step((deg_batch, ref_batch), batch_idx)

            for i in range(rec_batch.shape[0]):
                deg_i, ref_i, rec_i = deg_batch[i], ref_batch[i], rec_batch[i]

                if run_scenes and scene_counter < max_scenes:
                    deg_cpu = to_complex_channels(deg_i).detach().cpu()
                    ref_cpu = to_complex_channels(ref_i).detach().cpu()
                    rec_cpu = to_complex_channels(rec_i).detach().cpu()
                    generate_and_save_scenes(deg_cpu, ref_cpu, rec_cpu, save_dir, scene_counter)
                    scene_counter += 1
                    if scene_counter >= max_scenes:
                        print(
                            f"\n>>> [SUCCESS] Generated {max_scenes} target scenes. Short-circuiting evaluation run to save resources.")
                        return {}

                if run_point_target:
                    irf_results.append(point_target_analysis_single(
                        magnitude_map(deg_i), magnitude_map(ref_i), magnitude_map(rec_i)
                    ))

                if run_kde:
                    ref_logmag.append(
                        torch.log1p(torch.abs(to_complex_channels(ref_i))).flatten().detach().cpu().numpy())
                    deg_logmag.append(
                        torch.log1p(torch.abs(to_complex_channels(deg_i))).flatten().detach().cpu().numpy())
                    rec_logmag.append(
                        torch.log1p(torch.abs(to_complex_channels(rec_i))).flatten().detach().cpu().numpy())

                if run_phase:
                    ref_cplx, deg_cplx, rec_cplx = to_complex_channels(ref_i), to_complex_channels(
                        deg_i), to_complex_channels(rec_i)
                    for state, target in [("degraded", deg_cplx), ("reconstructed", rec_cplx)]:
                        phase_diff = torch.angle(target * torch.conj(ref_cplx))
                        phase_stats[state]["mean_abs_phase_diff"].append(
                            float(torch.mean(torch.abs(phase_diff)).item()))
                        phase_stats[state]["std_phase_diff_rad"].append(float(torch.std(phase_diff).item()))
                        coh = local_interferometric_coherence(target, ref_cplx, window_size=5)
                        phase_stats[state]["coherence_mean"].append(float(torch.mean(coh).item()))
                        phase_stats[state]["coherence_std"].append(float(torch.std(coh).item()))

                if run_enl:
                    ref_enl = equivalent_number_of_looks(to_complex_channels(ref_i))
                    deg_enl = equivalent_number_of_looks(to_complex_channels(deg_i))
                    rec_enl = equivalent_number_of_looks(to_complex_channels(rec_i))

                    enl_stats["ref_enl"].append(ref_enl)

                    enl_stats["deg_enl"].append(deg_enl)
                    enl_stats["deg_abs_error"].append(abs(deg_enl - ref_enl))
                    enl_stats["deg_rel_error"].append(abs(deg_enl - ref_enl) / max(abs(ref_enl), EPS))

                    enl_stats["rec_enl"].append(rec_enl)
                    enl_stats["rec_abs_error"].append(abs(rec_enl - ref_enl))
                    enl_stats["rec_rel_error"].append(abs(rec_enl - ref_enl) / max(abs(ref_enl), EPS))

            if run_kde and sum(arr.size for arr in ref_logmag) > max_kde_samples:
                break

    results: dict[str, Any] = {
        "config": {"checkpoint_path": checkpoint_path, "save_dir": str(save_dir), "tests": tests}}

    if run_point_target:
        results["point_target_analysis"] = {"num_samples": len(irf_results), "summary": aggregate_irf(irf_results)}
        save_irf_plots(irf_results, save_dir=save_dir)

    if run_kde:
        results["kde_distance"] = kde_distribution_distance(
            np.concatenate(deg_logmag)[:max_kde_samples], np.concatenate(ref_logmag)[:max_kde_samples],
            np.concatenate(rec_logmag)[:max_kde_samples], save_dir=save_dir
        )

    if run_phase:
        results["phase_coherence"] = {
            state: {k: {"mean": float(np.mean(v)) if v else None, "std": float(np.std(v)) if v else None} for k, v in
                    stats.items()}
            for state, stats in phase_stats.items()
        }

    if run_enl:
        results["enl_comparison"] = {
            k: {"mean": float(np.mean(v)) if v else None, "std": float(np.std(v)) if v else None}
            for k, v in enl_stats.items() if k != "samples"
        }

    hypothesis_results = {}
    if run_point_target and irf_results:
        az_ratios = [r["rec_irf"]["azimuth"]["resolution_px"] / r["ref_irf"]["azimuth"]["resolution_px"] for r in
                     irf_results]
        rg_ratios = [r["rec_irf"]["range"]["resolution_px"] / r["ref_irf"]["range"]["resolution_px"] for r in
                     irf_results]
        hypothesis_results["H1_Azimuth_Resolution_Reconstruction"] = evaluate_hypothesis_ratio(az_ratios, margin=0.05)
        hypothesis_results["H2_Range_Resolution_Reconstruction"] = evaluate_hypothesis_ratio(rg_ratios, margin=0.05)

    if run_phase and phase_stats["reconstructed"]["std_phase_diff_rad"]:
        std_phase_deg = [np.rad2deg(val) for val in phase_stats["reconstructed"]["std_phase_diff_rad"]]
        hypothesis_results["H4_Phase_Consistency_Reconstruction"] = evaluate_hypothesis_threshold(std_phase_deg,
                                                                                                  max_threshold=5.0)

    results["hypothesis_testing"] = hypothesis_results

    with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PixelDiffusionConditional model on SAR-specific metrics.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML with evaluation section.")
    args = parser.parse_args()

    config = load_config(args.config)
    results = evaluate(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
