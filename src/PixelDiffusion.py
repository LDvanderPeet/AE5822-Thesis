import numpy as np
import math
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .DenoisingDiffusionProcess import *
from .EMA import EMAWrapper



def _build_hybrid_diffusion_loss(num_timesteps, base_loss='mse', ms_ssim_t_limit=10):
    """Create timestep-aware hybrid loss for diffusion training.

    For early timesteps (t <= ms_ssim_t_limit), optimize 1 - MS-SSIM.
    For later timesteps, optimize the configured base loss (L1/L2)."""
    del num_timesteps

    base_loss_name = base_loss.lower()
    if base_loss_name == 'mse':
        base_loss_fn = lambda pred, target: F.mse_loss(pred, target, reduction='none').flatten(1).mean(dim=1)
    elif base_loss_name in {'mae', 'l1'}:
        base_loss_fn = lambda pred, target: F.l1_loss(pred, target, reduction='none').flatten(1).mean(dim=1)
    else:
        raise ValueError(f"Unsupported hybrid base loss '{base_loss}'. Expected one of 'mse', 'mae', or 'l1'.")

    def _hybrid_diffusion_loss(noise, noise_hat, t):
        base_per_sample = base_loss_fn(noise_hat, noise)

        noise_01 = (noise + 1.0) * 0.5
        noise_hat_01 = (noise_hat + 1.0) * 0.5
        ms_ssim_per_sample = multiscale_structural_similarity_index_measure(
            noise_hat_01,
            noise_01,
            data_range=1.0,
            reduction='none'
        )
        ms_ssim_loss_per_sample = 1.0 - ms_ssim_per_sample

        use_ms_ssim = (t <= ms_ssim_t_limit)
        hybrid_per_sample = torch.where(use_ms_ssim, ms_ssim_loss_per_sample, base_per_sample)
        return hybrid_per_sample.mean()

    return _hybrid_diffusion_loss

def _l1_diffusion_loss(noise, noise_hat, t):
    """Wrapper for L1 loss to accept the timestep 't' argument"""
    del t
    return F.l1_loss(noise_hat, noise)


class PixelDiffusionConditional(pl.LightningModule):
    """Conditional pixel-space diffusion Lightning module.

    Expects each batch to be `(x, y)` where:
    - `x`: condition tensor
    - `y`: target tensor to reconstruct/generate
    """
    def __init__(self,
                 condition_channels=3,
                 generated_channels=3,
                 input_condition_labels=None,
                 target_label="FA",
                 num_timesteps=1000,
                 schedule='linear',
                 noise_offset=0.0,
                 loss_fn='mse',
                 hybrid_base_loss='mse',
                 hybrid_ms_ssim_t_limt=10,
                 model_dim=64,
                 model_dim_mults=(1,2,4,8),
                 model_channels=None,
                 model_out_dim=None,
                 lr=1e-3,
                 lr_scheduler_factor=0.5,
                 lr_scheduler_patience=10,
                 ema_enabled=False,
                 ema_beta=0.9999,
                 ema_update_every=1,
                 ema_update_after_step=0,
                 phase_hist_max_batches=8,
                 data_global_max=4257.0,
                 wandb_save_config_file=True,
                 config_path=None,
                 wandb_config_artifact_name=None):

        super().__init__()
        self.lr = lr
        self.lr_scheduler_factor=lr_scheduler_factor
        self.lr_scheduler_patience=lr_scheduler_patience
        self.ema_enabled = ema_enabled
        self.ema_beta = float(ema_beta)
        self.ema_update_every = int(ema_update_every)
        self.ema_update_after_step = int(ema_update_after_step)
        self.ema_step = 0
        self.data_global_max = float(data_global_max)
        self._log_norm_scale = float(np.log1p(max(self.data_global_max, 1e-8)))
        self._val_hist_real_samples = []
        self._val_hist_imag_samples = []
        self._val_hist_mag_samples = []
        self._val_hist_max_samples = 200_000
        self._val_hist_samples_per_batch = 8_192,
        self._val_phase_err_samples = []
        self._phase_hist_max_batches = int(phase_hist_max_batches)
        self.wandb_save_config_file = bool(wandb_save_config_file)
        self.config_path = config_path
        self.wandb_config_artifact_name = wandb_config_artifact_name
        self._wandb_config_saved = False
        self.loss_name = loss_fn.lower()
        self.input_condition_labels = list(input_condition_labels) if input_condition_labels is not None else None
        self.target_label = str(target_label)
        if self.loss_name == 'mse':
            resolved_loss_fn = None
        elif self.loss_name == 'hybrid':
            resolved_loss_fn = _build_hybrid_diffusion_loss(
                num_timesteps=num_timesteps,
                base_loss=hybrid_base_loss,
                ms_ssim_t_limit=hybrid_ms_ssim_t_limt
            )
        elif self.loss_name == 'mae':
            resolved_loss_fn = _l1_diffusion_loss
        else:
            raise ValueError(f"Unsupported model.loss_fn {self.loss_name}. Expected one of 'mse', 'mae', or 'hybrid'.")

        # Core conditional diffusion process used by training, validation, and prediction.
        self.model=DenoisingDiffusionConditionalProcess(generated_channels=generated_channels,
                                                        condition_channels=condition_channels,
                                                        loss_fn=resolved_loss_fn,
                                                        schedule=schedule,
                                                        noise_offset=noise_offset,
                                                        num_timesteps=num_timesteps,
                                                        model_dim=model_dim,
                                                        model_dim_mults=model_dim_mults,
                                                        model_channels=model_channels,
                                                        model_out_dim=model_out_dim)
        self.ema = (
            EMAWrapper(
                model=self.model,
                decay=self.ema_beta,
                apply_every_n_steps=self.ema_update_every,
                start_step=self.ema_update_after_step,
            )
            if self.ema_enabled
            else None
        )

    def on_fit_start(self):
        """Upload the active config file once at fit start (global zero only)."""
        if (
                not self.wandb_save_config_file
                or self._wandb_config_saved
                or self.logger is None
                or self.trainer is None
                or not self.trainer.is_global_zero
                or not hasattr(self.logger, "experiment")
        ):
            return
        self._save_config_to_wandb()

    def _save_config_to_wandb(self):
        """Upload the exact config YAML used for this run to W&B."""
        if self.config_path is None:
            return
        config_file = Path(self.config_path).expanduser().resolve()
        if not config_file.exists():
            return

        run = self.logger.experiment
        run.save(str(config_file), policy="now")

        import wandb
        artifact = wandb.Artifact(
            name=self.wandb_config_artifact_name or f"config-{run.id}",
            type="config",
            description="Training config used for this run.",
        )
        artifact.add_file(str(config_file), name=config_file.name)
        run.log_artifact(artifact)
        self._wandb_config_saved = True

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Lightning inference helper; returns output mapped back to [0, 1]."""
        return self.output_T(self.model(*args, **kwargs))

    def input_T(self, input):
        # Model internally expects values in [-1, 1].
        # return (input.clip(0, 1).mul(2)).sub(1)
        return input.clamp(-1, 1)


    def output_T(self, input):
        # Inverse mapping from [-1, 1] back to [0, 1] for visualization/metrics.
        # return input.add(1).div(2)
        return input.clamp(-1, 1)

    def training_step(self, batch, batch_idx):   
        """Lightning train hook for conditional diffusion."""
        input,output=batch
        loss = self.model.p_loss(self.input_T(output),self.input_T(input))
        
        self.log('train_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA shadow weights after each optimization step."""
        del outputs, batch, batch_idx
        if self.ema is None:
            return
        self.ema.update(self.model, step=self.global_step)

    def validation_step(self, batch, batch_idx):     
        """Lightning validation hook.

        Logs `val_loss` for scheduler control and, on the first validation batch,
        computes a full denoising reconstruction plus image/metric logging.
        """
        input,output=batch
        loss = self.model.p_loss(self.input_T(output),self.input_T(input))
        self._accumulate_val_histogram_samples(output)
        
        self.log('val_loss',loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)

        pred_batch = None
        should_predict_for_phase_hist = batch_idx < self._phase_hist_max_batches
        if batch_idx == 0 or should_predict_for_phase_hist:
            pred_batch = self.predict_step(batch, batch_idx)
            self._accumulate_phase_error_samples(pred_batch, output)

        if batch_idx == 0:
            self._accumulate_val_histogram_samples(pred_batch)
            psnr, ssim, l1, phase_coh, phase_err = self._compute_reconstruction_metrics(pred_batch, output)
            self.log('val_recon_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_ssim', ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_l1', l1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_phase_coherence', phase_coh, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_phase_error', phase_err, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            ema_pred_batch = self._predict_with_ema_model(input)
            if ema_pred_batch is not None:
                ema_psnr, ema_ssim, ema_l1, ema_phase_coh, ema_phase_err = self._compute_reconstruction_metrics(ema_pred_batch, output)
                self.log('val_recon_psnr_ema', ema_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('val_recon_ssim_ema', ema_ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('val_recon_l1_ema', ema_l1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('val_recon_phase_coherence_ema', ema_phase_coh, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True)
                self.log('val_recon_phase_error_ema', ema_phase_err, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True)
            self._log_val_reconstruction(input, pred_batch, output, ema_pred_batch=ema_pred_batch)
            self._log_val_magnitude_kde_db(pred_batch, output)

        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_hist_real_samples = []
        self._val_hist_imag_samples = []
        self._val_hist_mag_samples = []

    def on_validation_epoch_end(self) -> None:
        self._log_val_reconstruction_histograms()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Lightning predict hook that runs the full denoising chain.

        This uses `DenoisingDiffusionConditionalProcess.forward`, which starts from
        random noise and iteratively denoises to produce the final reconstruction.
        """
        del batch_idx, dataloader_idx
        input,_ = batch
        pred = self.model(self.input_T(input))
        return self.output_T(pred)

    @torch.no_grad()
    def _predict_with_ema_model(self, input_batch):
        """Run prediction through EMA shadow model if EMA is enabled."""
        if self.ema is None:
            return None
        pred = self.ema.ema_model(self.input_T(input_batch))
        return self.output_T(pred)


    def configure_optimizers(self):
        """Create optimizer and ReduceLROnPlateau scheduler monitored on `val_loss`."""
        optimizer = torch.optim.AdamW(
            list(filter(lambda p: p.requires_grad, self.model.parameters())),
            lr=self.lr,
        )
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=self.lr_scheduler_factor,
                                      patience=self.lr_scheduler_patience)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "monitor": "val_loss"}}

    def _to_plot_image(self, tensor):
        """Convert CHW tensor to a matplotlib-friendly image array."""
        image = tensor.detach().float().cpu()

        if image.shape[0] >= 2:
            global_max = 2.5
            log_max = np.log1p(global_max)
            i = torch.sign(image[0]) * (torch.expm1(torch.abs(image[0]) * log_max))
            q = torch.sign(image[1]) * (torch.expm1(torch.abs(image[1]) * log_max))
            mag = torch.sqrt(i.square() + q.square()).numpy()
        else:
            mag = image[0].abs().numpy()

        scale = float(np.mean(mag) + 3.0 * np.std(mag))
        if scale <= 0.0:
            scale = 1.0
        mag = np.clip(mag / scale, 0.0, 1.0)
        return mag, 'gray'

    def _compute_reconstruction_metrics(self, pred_batch, target_batch):
        """Compute batch-level reconstruction metrics from [-1, 1] tensors."""
        pred = pred_batch.detach().float().clamp(-1, 1)
        target = target_batch.detach().float().clamp(-1, 1)

        l1 = F.l1_loss(pred, target)

        pred_cplx = self._to_complex_channels(pred)
        target_cplx = self._to_complex_channels(target)

        pred_mag = torch.abs(pred_cplx)
        target_mag = torch.abs(target_cplx)

        pred_mag_np = pred_mag.cpu().numpy()
        target_mag_np = target_mag.cpu().numpy()

        psnr_vals = []
        ssim_vals = []
        phase_coh_vals = []
        phase_err_vals = []

        for i in range(pred_mag_np.shape[0]):
            p_mag = pred_mag_np[i]
            t_mag = target_mag_np[i]

            psnr_vals.append(peak_signal_noise_ratio(t_mag, p_mag, data_range=math.sqrt(2)))

            ssim_vals.append(
                structural_similarity(
                    t_mag,
                    p_mag,
                    data_range=math.sqrt(2),
                    channel_axis=0,
                    win_size=11,
                    gaussian_weights=True,
                    sigma=1.5,
                )
            )
            phase_coh_vals.append(self._phase_coherence_mean(pred_cplx[i], target_cplx[i]))
            phase_err_vals.append(self._phase_error_mean_abs(pred_cplx[i], target_cplx[i]))

        psnr = torch.tensor(psnr_vals, device=pred.device, dtype=pred.dtype).mean()
        ssim = torch.tensor(ssim_vals, device=pred.device, dtype=pred.dtype).mean()
        phase_coh = torch.tensor(phase_coh_vals, device=pred.device, dtype=pred.dtype).mean()
        phase_err = torch.tensor(phase_err_vals, device=pred.device, dtype=pred.dtype).mean()

        return psnr, ssim, l1, phase_coh, phase_err

    @staticmethod
    def _to_complex_channels(tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(tensor):
            return tensor

        # Dimension-agnostic channel slicing
        if tensor.ndim == 4:  # [B, C, H, W]
            channel_dim = 1
        elif tensor.ndim == 3:  # [C, H, W]
            channel_dim = 0
        else:
            return torch.complex(tensor, torch.zeros_like(tensor))

        num_channels = tensor.shape[channel_dim]

        if num_channels >= 2 and num_channels % 2 == 0:
            if tensor.ndim == 4:
                real = tensor[:, 0::2, :, :]
                imag = tensor[:, 1::2, :, :]
            else:
                real = tensor[0::2, :, :]
                imag = tensor[1::2, :, :]
            return torch.complex(real, imag)

        return torch.complex(tensor, torch.zeros_like(tensor))

    @staticmethod
    def _phase_coherence_mean(pred_cplx: torch.Tensor, target_cplx: torch.Tensor) -> float:
        eps = 1e-8
        interferogram = pred_cplx * torch.conj(target_cplx)
        pred_pow = torch.abs(pred_cplx) ** 2
        target_pow = torch.abs(target_cplx) ** 2

        kernel = torch.ones((1, 1, 5, 5), device=pred_cplx.device, dtype=pred_pow.dtype)
        norm = 25.0

        def _mean_filter(x: torch.Tensor) -> torch.Tensor:
            original_shape = x.shape
            # Safe 4D reshape prevents the 5D crash
            x_reshaped = x.view(-1, 1, original_shape[-2], original_shape[-1])
            filtered = F.conv2d(x_reshaped, kernel, padding=2) / norm
            return filtered.view(original_shape)

        e_int_re = _mean_filter(interferogram.real)
        e_int_im = _mean_filter(interferogram.imag)
        e_pred = _mean_filter(pred_pow)
        e_target = _mean_filter(target_pow)

        numerator = torch.sqrt(e_int_re ** 2 + e_int_im ** 2)
        denominator = torch.sqrt(torch.clamp(e_pred * e_target, min=eps))
        coherence = numerator / torch.clamp(denominator, min=eps)
        return float(torch.mean(coherence).item())

    @staticmethod
    def _phase_error_mean_abs(pred_cplx: torch.Tensor, target_cplx: torch.Tensor) -> float:
        phase_diff = torch.angle(pred_cplx * torch.conj(target_cplx))
        return float(torch.mean(torch.abs(phase_diff)).item())


    def _log_val_reconstruction(self, input_batch, pred_batch, target_batch, ema_pred_batch=None):
        """Log validation reconstruction panels for inputs, prediction(s), and target."""
        if self.logger is None or self.trainer is None or not self.trainer.is_global_zero:
            return

        try:
            import matplotlib.pyplot as plt
            import wandb
        except ImportError:
            return

        pred_img, pred_cmap = self._to_plot_image(pred_batch[0])
        y_img, y_cmap = self._to_plot_image(target_batch[0])
        input_panels = self._build_input_panels(input_batch[0])
        plot_items = [*input_panels, (pred_img, pred_cmap, f"{self.target_label}_pred")]

        if ema_pred_batch is not None:
            ema_img, ema_cmap = self._to_plot_image(ema_pred_batch[0])
            plot_items.append((ema_img, ema_cmap, f"{self.target_label}_pred_ema"))
        plot_items.append((y_img, y_cmap, self.target_label))

        num_plots = len(plot_items)
        fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
        if num_plots == 1:
            axes = [axes]
        for ax, (img, cmap, title) in zip(axes, plot_items):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
        fig.tight_layout()

        self.logger.experiment.log(
            {"val/reconstruction": wandb.Image(fig)},
            step=self.global_step,
        )
        plt.close(fig)

    def _accumulate_val_histogram_samples(self, reconstruction_batch: torch.Tensor):
        """Collect subsampled histogram values from reconstruction on physical I/Q scale."""
        reconstruction_batch = reconstruction_batch.detach().float()
        reconstruction_physical = self._inverse_signed_log_normalize(reconstruction_batch)
        recon_complex = self._to_complex_channels(reconstruction_physical)

        real_vals = recon_complex.real.flatten().cpu().numpy()
        imag_vals = recon_complex.imag.flatten().cpu().numpy()
        mag_vals = torch.abs(recon_complex).flatten().cpu().numpy()

        def _subsample(values: np.ndarray) -> np.ndarray:
            if values.size <= self._val_hist_samples_per_batch:
                return values
            idx = np.linspace(0, values.size - 1, self._val_hist_samples_per_batch, dtype=int)
            return values[idx]

        self._val_hist_real_samples.append(_subsample(real_vals))
        self._val_hist_imag_samples.append(_subsample(imag_vals))
        self._val_hist_mag_samples.append(_subsample(mag_vals))

    def _log_val_reconstruction_histograms(self):
        """Log end-of-epoch real/imag/magnitude histogram figure for reconstruction."""
        if (
            self.logger is None
            or self.trainer is None
            or not self.trainer.is_global_zero
            or len(self._val_hist_real_samples) == 0
        ):
            return

        try:
            import matplotlib.pyplot as plt
            import wandb
        except ImportError:
            return

        real_vals = np.concatenate(self._val_hist_real_samples)
        imag_vals = np.concatenate(self._val_hist_imag_samples)
        mag_vals = np.concatenate(self._val_hist_mag_samples)

        if real_vals.size > self._val_hist_max_samples:
            idx = np.linspace(0, real_vals.size - 1, self._val_hist_max_samples, dtype=int)
            real_vals = real_vals[idx]
        if imag_vals.size > self._val_hist_max_samples:
            idx = np.linspace(0, imag_vals.size - 1, self._val_hist_max_samples, dtype=int)
            imag_vals = imag_vals[idx]
        if mag_vals.size > self._val_hist_max_samples:
            idx = np.linspace(0, mag_vals.size - 1, self._val_hist_max_samples, dtype=int)
            mag_vals = mag_vals[idx]

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

        self.logger.experiment.log(
            {
                "val/reconstruction_histograms": wandb.Image(fig),
            },
            step=self.global_step,
        )
        plt.close(fig)

    def _accumulate_phase_error_samples(self, pred_batch: torch.Tensor, target_batch: torch.Tensor):
        pred_physical = self._inverse_signed_log_normalize(pred_batch.detach().float())
        target_physical = self._inverse_signed_log_normalize(target_batch.detach().float())
        pred_cplx = self._to_complex_channels(pred_physical)
        target_cplx = self._to_complex_channels(target_physical)

        phase_err = torch.abs(torch.angle(pred_cplx * torch.conj(target_cplx))).flatten().cpu().numpy()
        if phase_err.size > self._val_hist_samples_per_batch:
            idx = np.linspace(0, phase_err.size - 1, self._val_hist_samples_per_batch, dtype=int)
            phase_err = phase_err[idx]
        self._val_phase_err_samples.append(phase_err)

    def _log_val_phase_error_histogram(self):
        if (
            self.logger is None
            or self.trainer is None
            or not self.trainer.is_global_zero
            or len(self._val_phase_err_samples) == 0
        ):
            return

        try:
            import matplotlib.pyplot as plt
            import wandb
        except ImportError:
            return

        phase_vals = np.concatenate(self._val_phase_err_samples)
        if phase_vals.size > self._val_hist_max_samples:
            idx = np.linspace(0, phase_vals.size - 1, self._val_hist_max_samples, dtype=int)
            phase_vals = phase_vals[idx]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.hist(phase_vals, bins=120, color="mediumpurple", alpha=0.9)
        ax.set_title("Validation Phase Error Histogram")
        ax.set_xlabel("|Wrapped phase error| [rad]")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25, linestyle="--")
        fig.tight_layout()
        self.logger.experiment.log({"val/phase_error_histogram": wandb.Image(fig)}, step=self.global_step)
        plt.close(fig)

    def _log_val_magnitude_kde_db(self, pred_batch: torch.Tensor, target_batch: torch.Tensor):
        """Log KDE overlay of reconstruction vs reference magnitudes in dB for one validation image."""
        if self.logger is None or self.trainer is None or not self.trainer.is_global_zero:
            return

        try:
            import matplotlib.pyplot as plt
            import wandb
            from scipy.stats import gaussian_kde
        except ImportError:
            return

        eps = 1e-8
        max_samples = 200_000

        pred_physical = self._inverse_signed_log_normalize(pred_batch.detach().float())
        target_physical = self._inverse_signed_log_normalize(target_batch.detach().float())
        pred_cplx = self._to_complex_channels(pred_physical)[0]
        target_cplx = self._to_complex_channels(target_physical)[0]

        pred_db = (20.0 * torch.log10(torch.clamp(torch.abs(pred_cplx), min=eps))).flatten().cpu().numpy()
        target_db = (20.0 * torch.log10(torch.clamp(torch.abs(target_cplx), min=eps))).flatten().cpu().numpy()

        if pred_db.size == 0 or target_db.size == 0:
            return

        if pred_db.size > max_samples:
            idx = np.linspace(0, pred_db.size - 1, max_samples, dtype=int)
            pred_db = pred_db[idx]
        if target_db.size > max_samples:
            idx = np.linspace(0, target_db.size - 1, max_samples, dtype=int)
            target_db = target_db[idx]

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

        self.logger.experiment.log(
            {"val/magnitude_kde_db": wandb.Image(fig)},
            step=self.global_step,
        )
        plt.close(fig)

    def _inverse_signed_log_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Undo signed log normalization back to physical I/Q scale."""
        return torch.sign(tensor) * torch.expm1(torch.abs(tensor) * self._log_norm_scale)

    def _build_input_panels(self, input_tensor: torch.Tensor):
        """Split condition channels into per-input images and attach human-readable labels."""
        image = input_tensor.detach().float().cpu()

        if self.input_condition_labels:
            labels = self.input_condition_labels
        else:
            if image.shape[0] % 2 == 0:
                num_inputs = max(1, image.shape[0] // 2)
            else:
                num_inputs = image.shape[0]
            labels = [f"input_{i + 1}" for i in range(num_inputs)]

        num_inputs = len(labels)
        if num_inputs <= 1:
            img, cmap = self._to_plot_image(image)
            label = labels[0] if labels else "x"
            return [(img, cmap, label)]

        if image.shape[0] % num_inputs != 0:
            img, cmap = self._to_plot_image(image)
            return [(img, cmap, "x")]

        channels_per_input = image.shape[0] // num_inputs
        panels = []
        for idx, label in enumerate(labels):
            start = idx * channels_per_input
            end = (idx + 1) * channels_per_input
            img, cmap = self._to_plot_image(image[start:end])
            panels.append((img, cmap, label))
        return panels