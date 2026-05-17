import numpy as np
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .DenoisingDiffusionProcess import *
from .EMA import EMAWrapper



def _build_hybrid_diffusion_loss(
        inverse_norm_fn,
        base_loss='mse',
        ms_ssim_t_limit=10,
        ms_ssim_weight=15.0,
        phase_weight=0.5,
):
    """Create timestep-aware hybrid loss for diffusion training.

    For early timesteps (t <= ms_ssim_t_limit), optimize 1 - MS-SSIM + Circular Phase loss.
    For later timesteps, optimize the configured base loss (L1/L2)."""

    base_loss_name = base_loss.lower()
    if base_loss_name == 'mse':
        base_loss_fn = lambda pred, target: F.mse_loss(pred, target, reduction='none').flatten(1).mean(dim=1)
    elif base_loss_name in {'mae', 'l1'}:
        base_loss_fn = lambda pred, target: F.l1_loss(pred, target, reduction='none').flatten(1).mean(dim=1)
    else:
        raise ValueError(f"Unsupported hybrid base loss '{base_loss}'. Expected one of 'mse', 'mae', or 'l1'.")

    def _hybrid_diffusion_loss(noise, noise_hat, t, x_t, x_0, forward_process):
        # Base loss (MSE/L1) is still calculated directly on the NOISE
        base_per_sample = base_loss_fn(noise_hat, noise)

        mask = (t <= ms_ssim_t_limit)

        if not mask.any():
            loss = base_per_sample.mean()
            details = {
                "base_loss_mean": loss.detach(),
                "ms_ssim_loss_mean": torch.tensor(0.0, device=loss.device),
                "phase_loss_mean": torch.tensor(0.0, device=loss.device),
                "advanced_loss_mean": torch.tensor(0.0, device=loss.device),
                "hybrid_loss_mean": loss.detach(),
                "advanced_t_fraction": torch.tensor(0.0, device=loss.device),
            }
            return loss, details

        # Extract the masked values for advanced physics loss
        adv_noise_hat = noise_hat[mask] # Predicted noise
        adv_x_t = x_t[mask]             # Noisy image at timestep t
        adv_x_0 = x_0[mask]             # Ground truth clean image
        adv_t = t[mask]                 # Timesteps

        # Extract cumulative alphas from the noise scheduler
        alphas_cumprod_t = forward_process.alphas_cumprod[adv_t].view(-1, 1, 1, 1)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)

        # CRITICAL MATH: Calculate the predicted clean image (x_0_hat) from the predicted noise
        pred_x0_norm = (adv_x_t - sqrt_one_minus_alphas_cumprod_t * adv_noise_hat) / sqrt_alphas_cumprod_t
        target_x0_norm = adv_x_0

        # Un-normalize the IMAGES, not the noise!
        pred_physical = inverse_norm_fn(pred_x0_norm.clamp(-1.0, 1.0))
        target_physical = inverse_norm_fn(target_x0_norm.clamp(-1.0, 1.0))

        # Convert to complex channels
        pred_cplx = PixelDiffusionConditional._to_complex_channels(pred_physical)
        target_cplx = PixelDiffusionConditional._to_complex_channels(target_physical)

        pred_mag = torch.abs(pred_cplx)
        target_mag = torch.abs(target_cplx)

        batch_max = target_mag.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        target_mag_01 = target_mag / batch_max

        ## Phase math
        pred_phase = torch.angle(pred_cplx)
        target_phase = torch.angle(target_cplx)
        phase_diff = torch.abs(pred_phase - target_phase)
        circular_phase_diff = torch.min(phase_diff, 2 * torch.pi - phase_diff)
        phase_loss_subset = (circular_phase_diff * target_mag_01).flatten(1).mean(dim=1)

        ## MS-SSIM math
        pred_real_01 = ((pred_cplx.real / batch_max).clamp(-1.0, 1.0) + 1.0) / 2.0
        target_real_01 = ((target_cplx.real / batch_max).clamp(-1.0, 1.0) + 1.0) / 2.0
        pred_imag_01 = ((pred_cplx.imag / batch_max).clamp(-1.0, 1.0) + 1.0) / 2.0
        target_imag_01 = ((target_cplx.imag / batch_max).clamp(-1.0, 1.0) + 1.0) / 2.0

        custom_betas = (0.0517, 0.3295, 0.3462, 0.2726)

        ms_ssim_real = multiscale_structural_similarity_index_measure(
            pred_real_01, target_real_01, data_range=1.0, reduction='none', betas=custom_betas
        )
        ms_ssim_imag = multiscale_structural_similarity_index_measure(
            pred_imag_01, target_imag_01, data_range=1.0, reduction='none', betas=custom_betas
        )

        ms_ssim_subset = (ms_ssim_real + ms_ssim_imag) / 2.0
        ms_ssim_loss_subset = 1.0 - ms_ssim_subset

        advanced_penalty_subset = (ms_ssim_weight * ms_ssim_loss_subset) + (phase_weight * phase_loss_subset)

        advanced_per_sample = torch.zeros_like(base_per_sample)
        advanced_per_sample[mask] = advanced_penalty_subset

        hybrid_per_sample = base_per_sample + advanced_per_sample
        loss = hybrid_per_sample.mean()

        details = {
            "base_loss_mean": base_per_sample.mean().detach(),
            "ms_ssim_loss_mean": ms_ssim_loss_subset.mean().detach(),
            "phase_loss_mean": phase_loss_subset.mean().detach(),
            "advanced_loss_mean": advanced_penalty_subset.mean().detach(),
            "hybrid_loss_mean": loss.detach(),
            "advanced_t_fraction": mask.float().mean().detach(),
        }
        return loss, details

    return _hybrid_diffusion_loss

def _l1_diffusion_loss(noise, noise_hat, t, **kwargs):
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
                 hybrid_ms_ssim_weight=15.0,
                 hybrid_phase_weight=0.5,
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
                 data_global_max=300.0,
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

        self.wandb_save_config_file = bool(wandb_save_config_file)
        self.config_path = config_path
        self.wandb_config_artifact_name = wandb_config_artifact_name
        self._wandb_config_saved = False

        self.input_condition_labels = list(input_condition_labels) if input_condition_labels is not None else None
        self.target_label = str(target_label)
        self.loss_name = loss_fn.lower()

        if self.loss_name == 'mse':
            resolved_loss_fn = None
        elif self.loss_name == 'hybrid':
            resolved_loss_fn = _build_hybrid_diffusion_loss(
                inverse_norm_fn=self._inverse_signed_log_normalize,
                base_loss=hybrid_base_loss,
                ms_ssim_t_limit=hybrid_ms_ssim_t_limt,
                ms_ssim_weight=hybrid_ms_ssim_weight,
                phase_weight=hybrid_phase_weight,
            )
        elif self.loss_name == 'mae':
            resolved_loss_fn = _l1_diffusion_loss
        else:
            raise ValueError(f"Unsupported model.loss_fn {self.loss_name}. Expected one of 'mse', 'mae', or 'hybrid'.")

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
        return input.clamp(-1, 1)


    def output_T(self, input):
        return input.clamp(-1, 1)

    def training_step(self, batch, batch_idx):   
        """Lightning train hook for conditional diffusion."""
        input, output = batch
        if self.loss_name == 'hybrid':
            loss, loss_details = self.model.p_loss(self.input_T(output), self.input_T(input), return_details=True)
            self._log_hybrid_loss_terms(loss_details, stage='train')
        else:
            loss = self.model.p_loss(self.input_T(output), self.input_T(input))

        self.log('train_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA shadow weights after each optimization step."""
        del outputs, batch, batch_idx
        if self.ema is None:
            return
        self.ema.update(self.model, commit=False)

    def validation_step(self, batch, batch_idx):     
        """Lightning validation hook.

        Logs `val_loss` for scheduler control and, on the first validation batch,
        computes a full denoising reconstruction plus image/metric logging.
        """
        input, output = batch
        if self.loss_name == 'hybrid':
            loss, loss_details = self.model.p_loss(self.input_T(output), self.input_T(input), return_details=True)
            self._log_hybrid_loss_terms(loss_details, stage='val')
        else:
            loss = self.model.p_loss(self.input_T(output), self.input_T(input))
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Only compute reconstruction metrics on first batch to save computation time
        if batch_idx == 0:
            pred_batch = self.predict_step(batch, batch_idx)
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

    def _log_hybrid_loss_terms(self, loss_details, stage: str):
        """Log individual hybrid-loss components for scale diagnostics."""
        log_on_step = stage == "train"
        self.log(f'{stage}_hybrid/base_loss', loss_details['base_loss_mean'], on_step=log_on_step, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f'{stage}_hybrid/ms_ssim_loss', loss_details['ms_ssim_loss_mean'], on_step=log_on_step, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f'{stage}_hybrid/phase_loss', loss_details['phase_loss_mean'], on_step=log_on_step, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f'{stage}_hybrid/advanced_loss', loss_details['advanced_loss_mean'], on_step=log_on_step, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f'{stage}_hybrid/loss', loss_details['hybrid_loss_mean'], on_step=log_on_step, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log(f'{stage}_hybrid/advanced_t_fraction', loss_details['advanced_t_fraction'], on_step=log_on_step,
                 on_epoch=True, prog_bar=False, logger=True)

    def _compute_reconstruction_metrics(self, pred_batch, target_batch):
        """Compute batch-level reconstruction metrics from [-1, 1] tensors."""
        pred = pred_batch.detach().float().clamp(-1, 1)
        target = target_batch.detach().float().clamp(-1, 1)

        l1 = F.l1_loss(pred, target)

        # Compute phase metrics on physical I/Q values (undo signed-log normalization first).
        pred_cplx_norm = self._to_complex_channels(pred)
        target_cplx_norm = self._to_complex_channels(target)

        pred_mag_norm = torch.abs(pred_cplx_norm).cpu().numpy()
        target_mag_norm = torch.abs(target_cplx_norm).cpu().numpy()

        psnr_vals, ssim_vals = [], []
        for i in range(pred_mag_norm.shape[0]):
            p_mag = pred_mag_norm[i]
            t_mag = target_mag_norm[i]

            psnr_vals.append(peak_signal_noise_ratio(t_mag, p_mag, data_range=1.0))
            ssim_vals.append(structural_similarity(
                t_mag,
                p_mag,
                data_range=1.0,
                channel_axis=0,
                win_size=11,
                gaussian_weights=True,
                sigma=1.5,
                )
            )
        psnr = torch.tensor(psnr_vals, device=pred.device, dtype=pred.dtype).mean()
        ssim = torch.tensor(ssim_vals, device=pred.device, dtype=pred.dtype).mean()

        pred_phys = self._inverse_signed_log_normalize(pred)
        target_phys = self._inverse_signed_log_normalize(target)

        pred_cplx_phys = self._to_complex_channels(pred_phys)
        target_cplx_phys = self._to_complex_channels(target_phys)

        phase_coh_vals, phase_err_vals = [], []
        for i in range(pred_cplx_phys.shape[0]):
            phase_coh_vals.append(self._phase_coherence_mean(pred_cplx_phys[i], target_cplx_phys[i]))
            phase_err_vals.append(self._phase_error_mean_abs(pred_cplx_phys[i], target_cplx_phys[i]))

        phase_coh = torch.tensor(phase_coh_vals, device=pred.device, dtype=pred.dtype).mean()
        phase_err = torch.tensor(phase_err_vals, device=pred.device, dtype=pred.dtype).mean()

        return psnr, ssim, l1, phase_coh, phase_err

    def _inverse_signed_log_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Undo Complex Amplitude Compression back to physical I/Q scale.."""
        out = torch.zeros_like(tensor)

        if tensor.ndim == 4:
            I_comp = tensor[:, 0::2]
            Q_comp = tensor[:, 1::2]
        else:
            I_comp = tensor[0::2]
            Q_comp = tensor[1::2]

        A_comp = torch.sqrt(I_comp ** 2 + Q_comp ** 2)
        A_comp_safe = torch.clamp(A_comp, min=1e-8)

        # Reverse the log1p scaling
        A_phys = torch.expm1(A_comp * self._log_norm_scale)

        # Restore Cartesian vectors
        if tensor.ndim == 4:
            out[:, 0::2] = A_phys * (I_comp / A_comp_safe)
            out[:, 1::2] = A_phys * (Q_comp / A_comp_safe)
        else:
            out[0::2] = A_phys * (I_comp / A_comp_safe)
            out[1::2] = A_phys * (Q_comp / A_comp_safe)

        return out

    @staticmethod
    def _to_complex_channels(tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(tensor):
            return tensor

        channel_dim = 1 if tensor.ndim == 4 else 0
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
