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


def load_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model_from_config(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> PixelDiffusionConditional:
    model_cfg = config.get("model", {})
    opt_cfg = config.get("optimization", {})
    lr_sched_cfg = opt_cfg.get("reduce_lr_on_plateau", {})
    unet_cfg = model_cfg.get("unet", {})
    ema_cfg = model_cfg.get("ema", {})
    data_cfg = config.get("data", {})
    sa_cfg = data_cfg.get("subaperture_config", {}) if isinstance(data_cfg, dict) else {}
    input_indices = sa_cfg.get("input_indices", [])
    input_condition_labels = [f"SA{int(idx)}" if int(idx) > 0 else "FA" for idx in input_indices]

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
    """Convert CHW tensors to complex channels.

    Supports:
    - complex tensors
    - 2-channel real/imag pairs in channel dimension
    """
    if torch.is_complex(t):
        return t

    if t.shape[0] >= 2 and t.shape[0] % 2 == 0:
        real = t[0::2]
        imag = t[1::2]
        return torch.complex(real, imag)

    return torch.complex(t, torch.zeros_like(t))


def magnitude_map(t: torch.Tensor) -> torch.Tensor:
    cplx = to_complex_channels(t)
    mag = torch.abs(cplx)
    return mag[0] if mag.ndim == 3 else mag


def _resolution_3db(profile: np.ndarray) -> float:
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
    p = profile.astype(np.float64)
    peak_idx = int(np.argmax(p))
    peak_val = max(p[peak_idx], EPS)
    half = max(1, int(round(main_lobe_width / 2.0)))
    lo = max(0, peak_idx - half)
    hi = min(len(p), peak_idx + half + 1)

    main = p[lo:hi]
    side = np.concatenate([p[:lo], p[hi:]]) if lo > 0 or hi < len(p) else np.array([EPS])

    pslr_db = 20.0 * np.log10(max(side.max(), EPS) / peak_val)
    islr_db = 10.0 * np.log10(max(np.sum(side ** 2), EPS) / max(np.sum(main ** 2), EPS))
    return float(pslr_db), float(islr_db)


def upsample_irf_window(window: torch.Tensor, factor: int = 8) -> torch.Tensor:
    w = window.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(w, scale_factor=factor, mode="bicubic", align_corners=False)
    return up[0, 0]


def point_target_analysis_single(
    gt_mag: torch.Tensor,
    pred_mag: torch.Tensor,
    window_radius: int = 8,
    upsample_factor: int = 8,
) -> dict[str, Any]:
    h, w = gt_mag.shape
    peak_idx = torch.argmax(gt_mag)
    py = int((peak_idx // w).item())
    px = int((peak_idx % w).item())

    y0 = max(0, py - window_radius)
    y1 = min(h, py + window_radius + 1)
    x0 = max(0, px - window_radius)
    x1 = min(w, px + window_radius + 1)

    gt_win = gt_mag[y0:y1, x0:x1]
    pred_win = pred_mag[y0:y1, x0:x1]

    gt_up = upsample_irf_window(gt_win, factor=upsample_factor)
    pred_up = upsample_irf_window(pred_win, factor=upsample_factor)

    gh, gw = gt_up.shape
    gidx = torch.argmax(gt_up)
    gpy = int((gidx // gw).item())
    gpx = int((gidx % gw).item())

    pred_local_idx = torch.argmax(pred_up)
    ppy = int((pred_local_idx // gw).item())
    ppx = int((pred_local_idx % gw).item())

    gt_az = gt_up[gpy, :].detach().cpu().numpy()
    gt_rg = gt_up[:, gpx].detach().cpu().numpy()
    pred_az = pred_up[ppy, :].detach().cpu().numpy()
    pred_rg = pred_up[:, ppx].detach().cpu().numpy()

    def profile_metrics(profile: np.ndarray) -> IRFMetrics:
        res = _resolution_3db(profile)
        pslr, islr = _pslr_islr(profile, main_lobe_width=res)
        return IRFMetrics(resolution_px=res / upsample_factor, pslr_db=pslr, islr_db=islr)

    gt_irf = IRFResult(azimuth=profile_metrics(gt_az), range_=profile_metrics(gt_rg))
    pred_irf = IRFResult(azimuth=profile_metrics(pred_az), range_=profile_metrics(pred_rg))

    return {
        "gt_peak_xy": [py, px],
        "crop_bounds": [y0, y1, x0, x1],
        "gt_irf": {
            "azimuth": gt_irf.azimuth.__dict__,
            "range": gt_irf.range_.__dict__,
        },
        "pred_irf": {
            "azimuth": pred_irf.azimuth.__dict__,
            "range": pred_irf.range_.__dict__,
        },
        "profiles": {
            "gt_azimuth": gt_az.tolist(),
            "gt_range": gt_rg.tolist(),
            "pred_azimuth": pred_az.tolist(),
            "pred_range": pred_rg.tolist(),
        },
    }


def aggregate_irf(irf_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not irf_results:
        return {}

    def collect(path: tuple[str, str, str]) -> list[float]:
        vals = []
        for item in irf_results:
            vals.append(item[path[0]][path[1]][path[2]])
        return vals

    summary = {}
    for group in ["gt_irf", "pred_irf"]:
        summary[group] = {}
        for axis in ["azimuth", "range"]:
            summary[group][axis] = {}
            for metric in ["resolution_px", "pslr_db", "islr_db"]:
                vals = collect((group, axis, metric))
                summary[group][axis][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
    return summary


def save_irf_plots(irf_results: list[dict[str, Any]], save_dir: Path, max_plots: int = 5) -> None:
    for i, res in enumerate(irf_results[:max_plots]):
        prof = res["profiles"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(prof["gt_azimuth"], label="GT")
        axes[0].plot(prof["pred_azimuth"], label="Prediction")
        axes[0].set_title("Azimuth IRF Profile")
        axes[0].set_xlabel("Pixel")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()

        axes[1].plot(prof["gt_range"], label="GT")
        axes[1].plot(prof["pred_range"], label="Prediction")
        axes[1].set_title("Range IRF Profile")
        axes[1].set_xlabel("Pixel")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(save_dir / f"irf_profiles_sample_{i:03d}.png", dpi=160)
        plt.close(fig)


def kde_distribution_distance(
    gt_values: np.ndarray,
    pred_values: np.ndarray,
    save_dir: Path,
) -> dict[str, float]:
    gt_values = np.asarray(gt_values, dtype=np.float64)
    pred_values = np.asarray(pred_values, dtype=np.float64)

    lo = float(min(gt_values.min(), pred_values.min()))
    hi = float(max(gt_values.max(), pred_values.max()))
    grid = np.linspace(lo, hi, 512)

    kde_gt = gaussian_kde(gt_values)
    kde_pred = gaussian_kde(pred_values)
    pdf_gt = kde_gt(grid)
    pdf_pred = kde_pred(grid)

    pdf_gt = np.clip(pdf_gt, EPS, None)
    pdf_pred = np.clip(pdf_pred, EPS, None)

    pdf_gt /= np.trapz(pdf_gt, grid)
    pdf_pred /= np.trapz(pdf_pred, grid)

    w1 = float(wasserstein_distance(gt_values, pred_values))
    kl = float(np.trapz(pdf_gt * np.log(pdf_gt / pdf_pred), grid))

    plt.figure(figsize=(8, 5))
    plt.plot(grid, pdf_gt, label="GT KDE")
    plt.plot(grid, pdf_pred, label="Prediction KDE")
    plt.xlabel("log1p(|SLC|)")
    plt.ylabel("Density")
    plt.title("KDE of Log-Magnitude Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "kde_log_magnitude.png", dpi=160)
    plt.close()

    return {"wasserstein": w1, "kl_divergence": kl}


def local_interferometric_coherence(
    pred_cplx: torch.Tensor,
    gt_cplx: torch.Tensor,
    window_size: int = 5,
) -> torch.Tensor:
    if pred_cplx.ndim == 3:
        pred_cplx = pred_cplx[0]
    if gt_cplx.ndim == 3:
        gt_cplx = gt_cplx[0]

    interferogram = pred_cplx * torch.conj(gt_cplx)
    pred_pow = torch.abs(pred_cplx) ** 2
    gt_pow = torch.abs(gt_cplx) ** 2

    kernel = torch.ones((1, 1, window_size, window_size), device=pred_cplx.device, dtype=pred_pow.dtype)
    norm = float(window_size * window_size)

    def mean_filter(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x.unsqueeze(0).unsqueeze(0), kernel, padding=window_size // 2) / norm

    e_int_re = mean_filter(interferogram.real)
    e_int_im = mean_filter(interferogram.imag)
    e_pred = mean_filter(pred_pow)
    e_gt = mean_filter(gt_pow)

    num = torch.sqrt(e_int_re ** 2 + e_int_im ** 2)
    den = torch.sqrt(torch.clamp(e_pred * e_gt, min=EPS))
    coh = num / torch.clamp(den, min=EPS)
    return coh.squeeze(0).squeeze(0)


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = config.get("evaluation", {})
    tests = eval_cfg.get("tests", {})

    checkpoint_path = eval_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("Missing evaluation.checkpoint_path in config.")

    save_dir = Path(eval_cfg.get("save_dir", "evaluation_results"))
    save_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.get("seed", 42), workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = PairedDataModule.from_config(config)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    model = build_model_from_config(config, checkpoint_path, device)

    run_point_target = bool(tests.get("point_target_analysis", False))
    run_kde = bool(tests.get("kde_distance", False))
    run_phase = bool(tests.get("phase_coherence", False))

    irf_results: list[dict[str, Any]] = []
    max_kde_samples = int(eval_cfg.get("max_kde_samples", 400_000))
    gt_logmag, pred_logmag = [], []
    phase_stats = {
        "mean_abs_phase_diff": [],
        "coherence_mean": [],
        "coherence_std": [],
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model.predict_step((x, y), batch_idx)

            for i in range(pred.shape[0]):
                pred_i = pred[i]
                y_i = y[i]

                if run_point_target:
                    gt_mag = magnitude_map(y_i)
                    pred_mag = magnitude_map(pred_i)
                    irf_results.append(point_target_analysis_single(gt_mag, pred_mag))

                if run_kde:
                    gt_mag = torch.abs(to_complex_channels(y_i)).flatten()
                    pred_mag = torch.abs(to_complex_channels(pred_i)).flatten()

                    gt_log = torch.log1p(gt_mag).detach().cpu().numpy()
                    pred_log = torch.log1p(pred_mag).detach().cpu().numpy()

                    gt_logmag.append(gt_log)
                    pred_logmag.append(pred_log)

                if run_phase:
                    gt_cplx = to_complex_channels(y_i)
                    pred_cplx = to_complex_channels(pred_i)

                    phase_diff = torch.angle(pred_cplx * torch.conj(gt_cplx))
                    phase_stats["mean_abs_phase_diff"].append(float(torch.mean(torch.abs(phase_diff)).item()))

                    coh = local_interferometric_coherence(pred_cplx, gt_cplx, window_size=5)
                    phase_stats["coherence_mean"].append(float(torch.mean(coh).item()))
                    phase_stats["coherence_std"].append(float(torch.std(coh).item()))

            if run_kde and sum(arr.size for arr in gt_logmag) > max_kde_samples:
                break

    results: dict[str, Any] = {
        "config": {
            "checkpoint_path": checkpoint_path,
            "save_dir": str(save_dir),
            "tests": tests,
        }
    }

    if run_point_target:
        results["point_target_analysis"] = {
            "num_samples": len(irf_results),
            "summary": aggregate_irf(irf_results),
            "samples": irf_results,
        }
        save_irf_plots(irf_results, save_dir=save_dir)

    if run_kde:
        gt_vals = np.concatenate(gt_logmag, axis=0)[:max_kde_samples]
        pred_vals = np.concatenate(pred_logmag, axis=0)[:max_kde_samples]
        results["kde_distance"] = {
            "num_samples": int(gt_vals.size),
            **kde_distribution_distance(gt_vals, pred_vals, save_dir=save_dir),
        }

    if run_phase:
        results["phase_coherence"] = {
            k: {
                "mean": float(np.mean(v)) if v else None,
                "std": float(np.std(v)) if v else None,
            }
            for k, v in phase_stats.items()
        }

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
