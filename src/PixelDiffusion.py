import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .DenoisingDiffusionProcess import *
from .EMA import EMAWrapper



def _build_hybrid_diffusion_loss(num_timesteps):
    """Create timestep-aware hybrid loss for diffusion training."""
    denom = max(float(num_timesteps - 1, 1.0))

    def _hybrid_diffusion_loss(noise, noise_hat, t):
        mse_per_sample = F.mse_loss(noise_hat, noise, reduction='none').flatten(1).mean(dim=1)

        noise_01 = (noise + 1.0) * 0.5
        noise_hat_01 = (noise_hat + 1.0) * 0.5
        ms_ssim = multiscale_structural_similarity_index_measure(
            noise_hat_01,
            noise_hat_01,
            data_range=1.0,
            reduction='none'
        )

        t_norm = t.float() / denom
        ms_ssim_weight = 1.0 - t_norm
        hybrid_per_sample = (1.0 - ms_ssim_weight) * mse_per_sample + ms_ssim_weight * (1.0 - ms_ssim)
        return hybrid_per_sample.mean()

    return _build_hybrid_diffusion_loss

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
                 num_timesteps=1000,
                 schedule='linear',
                 noise_offset=0.0,
                 loss_fn='mse',
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
                 ema_update_after_step=0):

        super().__init__()
        self.lr = lr
        self.lr_scheduler_factor=lr_scheduler_factor
        self.lr_scheduler_patience=lr_scheduler_patience
        self.ema_enabled = ema_enabled
        self.ema_beta = float(ema_beta)
        self.ema_update_every = int(ema_update_every)
        self.ema_update_after_step = int(ema_update_after_step)
        self.ema_step = 0
        
        self.loss_name = loss_fn.lower()
        if self.loss_name == 'mse':
            resolved_loss_fn = None
        elif self.loss_name == 'hybrid':
            resolved_loss_fn = _build_hybrid_diffusion_loss(num_timesteps)
        elif self.loss_name == 'mae':
            resolved_loss_fn = _l1_diffusion_loss
        else:
            raise ValueError(f"Unsupported model.loss_fn {self.loss_name}. Expected one of 'mse', 'hybrid'.")

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


    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Lightning inference helper; returns output mapped back to [0, 1]."""
        return self.output_T(self.model(*args, **kwargs))

    def input_T(self, input):
        # Model internally expects values in [-1, 1].
        return (input.clip(0, 1).mul(2)).sub(1)

    def output_T(self, input):
        # Inverse mapping from [-1, 1] back to [0, 1] for visualization/metrics.
        return input.add_(1).div(2)
    
    def training_step(self, batch, batch_idx):   
        """Lightning train hook for conditional diffusion."""
        input,output=batch
        loss = self.model.p_loss(self.input_T(output),self.input_T(input))
        
        self.log('train_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        
        return loss
            
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
            psnr, ssim, l1 = self._compute_reconstruction_metrics(pred_batch, output)
            self.log('val_recon_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_ssim', ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_l1', l1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            ema_pred_batch = self._predict_with_ema_model(input)
            if ema_pred_batch is not None:
                ema_psnr, ema_ssim, ema_l1 = self._compute_reconstruction_metrics(ema_pred_batch, output)
                self.log('val_recon_psnr_ema', ema_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('val_recon_ssim_ema', ema_ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('val_recon_l1_ema', ema_l1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        if self.ema is None:
            return None
        pred = self.ema.ema_model(self.input_T(input_batch))
        return self.output_T(pred)

    @torch.no_grad()
    def _update_ema(self):
        if self.ema is None:
            return
        self.ema.update(self.model, step=self.global_step)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        del outputs, batch, batch_idx
        self._update_ema()


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
            mag = torch.sqrt(image[0].square() + image[1].square()).numpy()
        else:
            mag = image[0].abs().numpy()

        global_max = 2.5
        log_max = np.log1p(global_max)
        mag = np.exp(mag * log_max) - 1.0

        scale = float(np.mean(mag) + 3.0 * np.std(mag))
        if scale <= 0.0:
            scale = 1.0
        mag = np.clip(mag / scale, 0.0, 1.0)
        return mag, 'gray'

    def _compute_reconstruction_metrics(self, pred_batch, target_batch):
        """Compute batch-level reconstruction metrics on [0, 1] tensors."""
        pred = pred_batch.detach().float().clamp(0, 1)
        target = target_batch.detach().float().clamp(0, 1)

        l1 = F.l1_loss(pred, target)

        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()

        psnr_vals = []
        ssim_vals = []
        for i in range(pred_np.shape[0]):
            p = pred_np[i]
            t = target_np[i]
            psnr_vals.append(peak_signal_noise_ratio(t, p, data_range=1.0))
            ssim_vals.append(
                structural_similarity(
                    t,
                    p,
                    data_range=1.0,
                    channel_axis=0,
                )
            )

        psnr = torch.tensor(psnr_vals, device=pred.device, dtype=pred.dtype).mean()
        ssim = torch.tensor(ssim_vals, device=pred.device, dtype=pred.dtype).mean()

        return psnr, ssim, l1

    def _log_val_reconstruction(self, input_batch, pred_batch, target_batch, ema_pred_batch=None):
        """Log a single `x | pred_regular | pred_ema | y` reconstruction panel to W&B."""
        if self.logger is None or self.trainer is None or not self.trainer.is_global_zero:
            return

        try:
            import matplotlib.pyplot as plt
            import wandb
        except ImportError:
            return

        x_img, x_cmap = self._to_plot_image(input_batch[0])
        pred_img, pred_cmap = self._to_plot_image(pred_batch[0])
        y_img, y_cmap = self._to_plot_image(target_batch[0])

        if ema_pred_batch is not None:
            ema_img, ema_cmap = self._to_plot_image(ema_pred_batch[0])
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(x_img, cmap=x_cmap)
        axes[0].set_title('x')
        axes[0].axis('off')
        axes[1].imshow(pred_img, cmap=pred_cmap)
        axes[1].set_title('pred_regular')
        axes[1].axis('off')
        if ema_pred_batch is not None:
            axes[2].imshow(ema_img, cmap=ema_cmap)
            axes[2].set_title('pred_ema')
            axes[2].axis('off')
            axes[3].imshow(y_img, cmap=y_cmap)
            axes[3].set_title('y')
            axes[3].axis('off')
        else:
            axes[2].imshow(y_img, cmap=y_cmap)
            axes[2].set_title('y')
            axes[2].axis('off')
        fig.tight_layout()

        self.logger.experiment.log(
            {"val/reconstruction": wandb.Image(fig)},
            step=self.global_step,
        )
        plt.close(fig)
