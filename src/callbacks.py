import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from scipy.stats import gaussian_kde
from pytorch_lightning.callbacks import Callback


class WandBPlottingCallback(Callback):
    """
    Handles all WandB plotting, histograms, and KDEs during validation.
    Keeps the core LightningModule clean and strictly mathematical.
    """

    def __init__(self, target_label="FA", phase_hist_max_batches=8):
        super().__init__()
        self.target_label = target_label
        self._phase_hist_max_batches = phase_hist_max_batches

        # Internal state for epoch-level histograms (Moved out of the model!)
        self._val_hist_real_samples = []
        self._val_hist_imag_samples = []
        self._val_hist_mag_samples = []
        self._val_phase_err_samples = []
        self._val_hist_max_samples = 200_000
        self._val_hist_samples_per_batch = 8_192

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset histogram accumulators at the start of each validation epoch."""
        self._val_hist_real_samples = []
        self._val_hist_imag_samples = []
        self._val_hist_mag_samples = []
        self._val_phase_err_samples = []

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not trainer.is_global_zero:
            return

        input_batch, target_batch = batch

        # Plot full image panels and KDE ONLY for the first batch
        if batch_idx == 0:
            pred_batch = pl_module.predict_step(batch, batch_idx)
            ema_pred_batch = None
            if pl_module.ema is not None:
                ema_pred_batch = pl_module._predict_with_ema_model(input_batch)

            self._log_val_reconstruction(pl_module, input_batch, pred_batch, target_batch, ema_pred_batch)
            self._log_val_magnitude_kde_db(pl_module, pred_batch, target_batch)

        # Accumulate histograms for the first N batches
        if batch_idx < self._phase_hist_max_batches:
            # Re-use pred_batch if we already calculated it for batch 0
            if batch_idx != 0:
                pred_batch = pl_module.predict_step(batch, batch_idx)
            self._accumulate_histograms(pl_module, pred_batch, target_batch)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self._plot_and_log_histograms(pl_module)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _to_plot_image(self, pl_module, tensor):
        """Convert CHW tensor to a matplotlib-friendly image array using CORRECT math."""
        image = tensor.detach().float().cpu()

        # Safely un-normalize using the model's updated complex math
        phys_image = pl_module._inverse_signed_log_normalize(image).cpu()

        if phys_image.shape[0] >= 2:
            cplx = pl_module._to_complex_channels(phys_image)
            mag = torch.abs(cplx).numpy()
            if mag.ndim == 3:
                mag = mag[0]  # Grab first polarization (e.g., VV)
        else:
            mag = phys_image[0].abs().numpy()

        mean = np.mean(mag)
        std = np.std(mag)
        vmax = float(mean + 3.0 * std)
        if vmax <= 0.0:
            vmax = 1.0

        return np.clip(mag / vmax, 0.0, 1.0), "gray"

    def _build_input_panels(self, pl_module, input_tensor):
        image = input_tensor.detach().float().cpu()
        labels = pl_module.input_condition_labels

        if not labels:
            num_inputs = max(1, image.shape[0] // 2) if image.shape[0] % 2 == 0 else image.shape[0]
            labels = [f"input_{i + 1}" for i in range(num_inputs)]

        num_inputs = len(labels)
        if num_inputs <= 1 or image.shape[0] % num_inputs != 0:
            img, cmap = self._to_plot_image(pl_module, image)
            return [(img, cmap, labels[0] if labels else "x")]

        channels_per_input = image.shape[0] // num_inputs
        panels = []
        for idx, label in enumerate(labels):
            start = idx * channels_per_input
            end = (idx + 1) * channels_per_input
            img, cmap = self._to_plot_image(pl_module, image[start:end])
            panels.append((img, cmap, label))
        return panels

    # -------------------------------------------------------------------------
    # Plotting & Logging Execution
    # -------------------------------------------------------------------------

    def _log_val_reconstruction(self, pl_module, input_batch, pred_batch, target_batch, ema_pred_batch=None):
        if pl_module.logger is None or not hasattr(pl_module.logger, "experiment"):
            return

        pred_img, pred_cmap = self._to_plot_image(pl_module, pred_batch[0])
        y_img, y_cmap = self._to_plot_image(pl_module, target_batch[0])
        input_panels = self._build_input_panels(pl_module, input_batch[0])

        plot_items = [*input_panels, (pred_img, pred_cmap, f"{self.target_label}_pred")]

        if ema_pred_batch is not None:
            ema_img, ema_cmap = self._to_plot_image(pl_module, ema_pred_batch[0])
            plot_items.append((ema_img, ema_cmap, f"{self.target_label}_pred_ema"))

        plot_items.append((y_img, y_cmap, self.target_label))

        fig, axes = plt.subplots(1, len(plot_items), figsize=(4 * len(plot_items), 4))
        if len(plot_items) == 1:
            axes = [axes]

        for ax, (img, cmap, title) in zip(axes, plot_items):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
        fig.tight_layout()

        pl_module.logger.experiment.log({"val/reconstruction": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    def _log_val_magnitude_kde_db(self, pl_module, pred_batch, target_batch):
        if pl_module.logger is None or not hasattr(pl_module.logger, "experiment"):
            return

        eps = 1e-8
        pred_physical = pl_module._inverse_signed_log_normalize(pred_batch.detach().float())
        target_physical = pl_module._inverse_signed_log_normalize(target_batch.detach().float())
        pred_cplx = pl_module._to_complex_channels(pred_physical)[0]
        target_cplx = pl_module._to_complex_channels(target_physical)[0]

        pred_db = (20.0 * torch.log10(torch.clamp(torch.abs(pred_cplx), min=eps))).flatten().cpu().numpy()
        target_db = (20.0 * torch.log10(torch.clamp(torch.abs(target_cplx), min=eps))).flatten().cpu().numpy()

        if pred_db.size == 0 or target_db.size == 0:
            return

        lo = float(min(np.percentile(pred_db, 0.2), np.percentile(target_db, 0.2)))
        hi = float(max(np.percentile(pred_db, 99.8), np.percentile(target_db, 99.8)))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return

        grid = np.linspace(lo, hi, 512)
        pred_kde = gaussian_kde(pred_db)(grid)
        target_kde = gaussian_kde(target_db)(grid)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, target_kde, label=f"{self.target_label} (reference)", linewidth=2)
        ax.plot(grid, pred_kde, label=f"{self.target_label}_pred (reconstruction)", linewidth=2)
        ax.set_xlabel("Magnitude (dB)")
        ax.set_ylabel("Density")
        ax.set_title("Validation Magnitude KDE (dB)")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

        pl_module.logger.experiment.log({"val/magnitude_kde_db": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    def _accumulate_histograms(self, pl_module, pred_batch, target_batch):
        reconstruction_physical = pl_module._inverse_signed_log_normalize(pred_batch.detach().float())
        recon_complex = pl_module._to_complex_channels(reconstruction_physical)

        real_vals = recon_complex.real.flatten().cpu().numpy()
        imag_vals = recon_complex.imag.flatten().cpu().numpy()
        mag_vals = torch.abs(recon_complex).flatten().cpu().numpy()

        def _subsample(values):
            if values.size <= self._val_hist_samples_per_batch:
                return values
            idx = np.linspace(0, values.size - 1, self._val_hist_samples_per_batch, dtype=int)
            return values[idx]

        self._val_hist_real_samples.append(_subsample(real_vals))
        self._val_hist_imag_samples.append(_subsample(imag_vals))
        self._val_hist_mag_samples.append(_subsample(mag_vals))

        target_physical = pl_module._inverse_signed_log_normalize(target_batch.detach().float())
        target_cplx = pl_module._to_complex_channels(target_physical)

        phase_err = torch.abs(torch.angle(recon_complex * torch.conj(target_cplx))).flatten().cpu().numpy()
        self._val_phase_err_samples.append(_subsample(phase_err))

    def _plot_and_log_histograms(self, pl_module):
        if pl_module.logger is None or not hasattr(pl_module.logger, "experiment") or len(
                self._val_hist_real_samples) == 0:
            return

        real_vals = np.concatenate(self._val_hist_real_samples)
        imag_vals = np.concatenate(self._val_hist_imag_samples)
        mag_vals = np.concatenate(self._val_hist_mag_samples)

        def _cap_samples(vals):
            if vals.size > self._val_hist_max_samples:
                idx = np.linspace(0, vals.size - 1, self._val_hist_max_samples, dtype=int)
                return vals[idx]
            return vals

        real_vals, imag_vals, mag_vals = map(_cap_samples, [real_vals, imag_vals, mag_vals])

        bins = 120
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(real_vals, bins=bins, color="steelblue", alpha=0.85)
        axes[0].set_title("Real (I)")
        axes[1].hist(imag_vals, bins=bins, color="darkorange", alpha=0.85)
        axes[1].set_title("Imag (Q)")
        axes[2].hist(mag_vals, bins=bins, color="seagreen", alpha=0.85)
        axes[2].set_title("Magnitude")

        for ax in axes:
            ax.grid(alpha=0.25, linestyle="--")
        fig.suptitle("Validation Reconstruction Histogram (Physical Scale)")
        fig.tight_layout()

        pl_module.logger.experiment.log({"val/reconstruction_histograms": wandb.Image(fig)}, commit=False)
        plt.close(fig)

        phase_vals = _cap_samples(np.concatenate(self._val_phase_err_samples))

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.hist(phase_vals, bins=120, color="mediumpurple", alpha=0.9)
        ax.set_title("Validation Phase Error Histogram")
        ax.set_xlabel("|Wrapped phase error| [rad]")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25, linestyle="--")
        fig.tight_layout()

        pl_module.logger.experiment.log({"val/phase_error_histogram": wandb.Image(fig)}, commit=False)
        plt.close(fig)


class DC2SCNPhaseRowCallback(pl.Callback):
    """
    Extracts a 1D slice of the phase at the brightest target (like DC2SCN)
    and plots the Reference Phase vs. Predicted Phase to Weights & Biases.
    """

    def __init__(self, num_samples=1):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger is None or not hasattr(trainer.logger, 'experiment'):
            return

        val_dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_dataloader))

        x, y = batch
        x = x[:self.num_samples].to(pl_module.device)
        y = y[:self.num_samples].to(pl_module.device)

        with torch.no_grad():
            if hasattr(pl_module.model, 'sample'):
                preds = pl_module.model.sample(x)
            else:
                preds = pl_module(x)

        target_phys = pl_module._inverse_signed_log_normalize(y)
        pred_phys = pl_module._inverse_signed_log_normalize(preds.clamp(-1.0, 1.0))

        target_cplx = pl_module._to_complex_channels(target_phys)
        pred_cplx = pl_module._to_complex_channels(pred_phys)

        plots = []

        for i in range(self.num_samples):
            mag = torch.abs(target_cplx[i, 0])  # Shape: [H, W]

            max_idx = torch.argmax(mag)
            y_max, x_max = np.unravel_index(max_idx.cpu().numpy(), mag.shape)

            target_phase = torch.angle(target_cplx[i, 0, y_max, :]).cpu().numpy()
            pred_phase = torch.angle(pred_cplx[i, 0, y_max, :]).cpu().numpy()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(target_phase, label="Reference Phase", color='blue', linestyle='--', linewidth=2)
            ax.plot(pred_phase, label="Predicted Phase", color='red', alpha=0.7, linewidth=2)

            ax.set_title(f"1D Phase Slice at Row {y_max} (Peak Target)")
            ax.set_xlabel("Range (Pixels)")
            ax.set_ylabel("Phase (Radians)")
            ax.set_ylim(-np.pi, np.pi)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            plt.tight_layout()

            plots.append(wandb.Image(fig, caption=f"Sample {i} - Row {y_max}"))
            plt.close(fig)  # Free memory to prevent memory leaks

        trainer.logger.experiment.log({
            "val/dc2scn_phase_slice": plots,
            "global_step": trainer.global_step
        })