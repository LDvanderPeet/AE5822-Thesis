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
        
        self.log('val_loss',loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)

        if batch_idx == 0:
            pred_batch = self.predict_step(batch, batch_idx)
            psnr, ssim, l1, phase_coh = self._compute_reconstruction_metrics(pred_batch, output)
            self.log('val_recon_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_ssim', ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_l1', l1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_phase_coherence', phase_coh, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            ema_pred_batch = self._predict_with_ema_model(input)
            if ema_pred_batch is not None:
                ema_psnr, ema_ssim, ema_l1, ema_phase_coh = self._compute_reconstruction_metrics(ema_pred_batch, output)
                self.log('val_recon_psnr_ema', ema_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('val_recon_ssim_ema', ema_ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('val_recon_l1_ema', ema_l1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                ema_psnr, ema_ssim, ema_l1, ema_phase_coh = self._compute_reconstruction_metrics(ema_pred_batch, output)
            self._log_val_reconstruction(input, pred_batch, output, ema_pred_batch=ema_pred_batch)

        return loss

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

        psnr = torch.tensor(psnr_vals, device=pred.device, dtype=pred.dtype).mean()
        ssim = torch.tensor(ssim_vals, device=pred.device, dtype=pred.dtype).mean()
        phase_coh = torch.tensor(phase_coh_vals, device=pred.device, dtype=pred.dtype).mean()

        return psnr, ssim, l1, phase_coh

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

