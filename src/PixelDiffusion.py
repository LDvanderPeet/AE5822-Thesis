import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .DenoisingDiffusionProcess import *

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
                 model_dim=64,
                 model_dim_mults=(1,2,4,8),
                 model_channels=None,
                 model_out_dim=None,
                 lr=1e-3,
                 lr_scheduler_factor=0.5,
                 lr_scheduler_patience=10):
        super().__init__()
        self.lr = lr
        self.lr_scheduler_factor=lr_scheduler_factor
        self.lr_scheduler_patience=lr_scheduler_patience
        
        # Core conditional diffusion process used by training, validation, and prediction.
        self.model=DenoisingDiffusionConditionalProcess(generated_channels=generated_channels,
                                                        condition_channels=condition_channels,
                                                        schedule=schedule,
                                                        num_timesteps=num_timesteps,
                                                        model_dim=model_dim,
                                                        model_dim_mults=model_dim_mults,
                                                        model_channels=model_channels,
                                                        model_out_dim=model_out_dim)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Lightning inference helper; returns output mapped back to [0, 1]."""
        return self.output_T(self.model(*args, **kwargs))

    def input_T(self, input):
        # Model internally expects values in [-1, 1].
        return (input.clip(0, 1).mul_(2)).sub_(1)

    def output_T(self, input):
        # Inverse mapping from [-1, 1] back to [0, 1] for visualization/metrics.
        return (input.add_(1)).div_(2)
    
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
            self._log_val_reconstruction(input, pred_batch, output)
        
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
        image = tensor.detach().float().cpu().clamp(0, 1)
        # if image.shape[0] >= 3:
            # return image[:3].permute(1, 2, 0).numpy(), None
        return image[0].numpy(), 'gray'

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

    def _log_val_reconstruction(self, input_batch, pred_batch, target_batch):
        """Log a single `x | pred | y` reconstruction panel to W&B."""
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

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(x_img, cmap=x_cmap)
        axes[0].set_title('x')
        axes[0].axis('off')
        axes[1].imshow(pred_img, cmap=pred_cmap)
        axes[1].set_title('pred')
        axes[1].axis('off')
        axes[2].imshow(y_img, cmap=y_cmap)
        axes[2].set_title('y')
        axes[2].axis('off')
        fig.tight_layout()

        self.logger.experiment.log(
            {"val/reconstruction": wandb.Image(fig)},
            step=self.global_step,
        )
        plt.close(fig)
