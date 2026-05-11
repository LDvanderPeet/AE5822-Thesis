import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class _DenseDilatedBlock(nn.Module):
    """Dense + dilated conv block inspired by DC2SCN style connectivity."""

    def __init__(self, channels: int, growth_rate: int = 32, dilations=(1, 2, 3, 2, 1)):
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = channels
        for d in dilations:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch += growth_rate

        self.fuse = nn.Conv2d(in_ch, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, dim=1)))
        return x + self.fuse(torch.cat(feats, dim=1))


class DC2SCNNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int = 64, num_blocks: int = 6, growth_rate: int = 32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_DenseDilatedBlock(width, growth_rate=growth_rate) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.head(x)
        return self.tail(self.blocks(f))


class DC2SCNLightning(pl.LightningModule):
    def __init__(self, condition_channels=2, generated_channels=2, input_condition_labels=None, target_label="FA",
                 width=64, num_blocks=6, growth_rate=32, lr=1e-4, lr_scheduler_factor=0.5,
                 lr_scheduler_patience=10, data_global_max=300.0):
        super().__init__()
        self.model = DC2SCNNet(condition_channels, generated_channels, width, num_blocks, growth_rate)
        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.data_global_max = float(data_global_max)
        self._log_norm_scale = float(np.log1p(max(self.data_global_max, 1e-8)))
        self.input_condition_labels = list(input_condition_labels) if input_condition_labels is not None else None
        self.target_label = str(target_label)
        self.ema = None

    def input_T(self, x):
        return x.clamp(-1, 1)

    def output_T(self, x):
        return x.clamp(-1, 1)

    def forward(self, x):
        return self.output_T(self.model(self.input_T(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.l1_loss(pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.l1_loss(pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx == 0:
            psnr, ssim, l1, phase_coh, phase_err = self._compute_reconstruction_metrics(pred, y)
            self.log('val_recon_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_ssim', ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_l1', l1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_phase_coherence', phase_coh, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_recon_phase_error', phase_err, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx, dataloader_idx
        x, _ = batch
        return self(x)

    def _predict_with_ema_model(self, input_batch):
        del input_batch
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.lr_scheduler_factor,
                                      patience=self.lr_scheduler_patience)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    # metric helpers mirrored from diffusion model
    def _compute_reconstruction_metrics(self, pred_batch, target_batch):
        pred = pred_batch.detach().float().clamp(-1, 1)
        target = target_batch.detach().float().clamp(-1, 1)
        l1 = F.l1_loss(pred, target)
        pred_cplx_norm = self._to_complex_channels(pred)
        target_cplx_norm = self._to_complex_channels(target)
        pred_mag_norm = torch.abs(pred_cplx_norm).cpu().numpy()
        target_mag_norm = torch.abs(target_cplx_norm).cpu().numpy()
        psnr_vals, ssim_vals = [], []
        for i in range(pred_mag_norm.shape[0]):
            psnr_vals.append(peak_signal_noise_ratio(target_mag_norm[i], pred_mag_norm[i], data_range=1.0))
            ssim_vals.append(structural_similarity(target_mag_norm[i], pred_mag_norm[i], data_range=1.0,
                                                   channel_axis=0, win_size=11, gaussian_weights=True, sigma=1.5))
        psnr = torch.tensor(psnr_vals, device=pred.device, dtype=pred.dtype).mean()
        ssim = torch.tensor(ssim_vals, device=pred.device, dtype=pred.dtype).mean()
        pred_phys = self._inverse_signed_log_normalize(pred)
        target_phys = self._inverse_signed_log_normalize(target)
        pred_cplx_phys = self._to_complex_channels(pred_phys)
        target_cplx_phys = self._to_complex_channels(target_phys)
        phase_coh = torch.tensor([self._phase_coherence_mean(pred_cplx_phys[i], target_cplx_phys[i]) for i in range(pred.shape[0])],
                                 device=pred.device, dtype=pred.dtype).mean()
        phase_err = torch.tensor([self._phase_error_mean_abs(pred_cplx_phys[i], target_cplx_phys[i]) for i in range(pred.shape[0])],
                                 device=pred.device, dtype=pred.dtype).mean()
        return psnr, ssim, l1, phase_coh, phase_err

    def _inverse_signed_log_normalize(self, tensor):
        out = torch.zeros_like(tensor)
        if tensor.ndim == 4:
            I_comp, Q_comp = tensor[:, 0::2], tensor[:, 1::2]
        else:
            I_comp, Q_comp = tensor[0::2], tensor[1::2]
        A_comp = torch.sqrt(I_comp ** 2 + Q_comp ** 2)
        A_comp_safe = torch.clamp(A_comp, min=1e-8)
        A_phys = torch.expm1(A_comp * self._log_norm_scale)
        if tensor.ndim == 4:
            out[:, 0::2] = A_phys * (I_comp / A_comp_safe)
            out[:, 1::2] = A_phys * (Q_comp / A_comp_safe)
        else:
            out[0::2] = A_phys * (I_comp / A_comp_safe)
            out[1::2] = A_phys * (Q_comp / A_comp_safe)
        return out

    @staticmethod
    def _to_complex_channels(tensor):
        if torch.is_complex(tensor):
            return tensor
        if tensor.ndim == 4:
            return torch.complex(tensor[:, 0::2], tensor[:, 1::2])
        return torch.complex(tensor[0::2], tensor[1::2])

    @staticmethod
    def _phase_coherence_mean(pred_cplx, target_cplx):
        eps = 1e-8
        interferogram = pred_cplx * torch.conj(target_cplx)
        pred_pow = torch.abs(pred_cplx) ** 2
        target_pow = torch.abs(target_cplx) ** 2
        kernel = torch.ones((1, 1, 5, 5), device=pred_cplx.device, dtype=pred_pow.dtype)

        def _mean_filter(x):
            x_reshaped = x.view(-1, 1, x.shape[-2], x.shape[-1])
            return (F.conv2d(x_reshaped, kernel, padding=2) / 25.0).view(x.shape)

        e_int_re, e_int_im = _mean_filter(interferogram.real), _mean_filter(interferogram.imag)
        e_pred, e_target = _mean_filter(pred_pow), _mean_filter(target_pow)
        return float(torch.mean(torch.sqrt(e_int_re ** 2 + e_int_im ** 2) / torch.sqrt(torch.clamp(e_pred * e_target, min=eps))).item())

    @staticmethod
    def _phase_error_mean_abs(pred_cplx, target_cplx):
        return float(torch.mean(torch.abs(torch.angle(pred_cplx * torch.conj(target_cplx)))).item())
