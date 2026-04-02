# Conditional Pixel Diffusion for SAR Subaperture Reconstruction

This repository contains a PyTorch Lightning training pipeline for **conditional pixel-space diffusion** on SAR subaperture data, using:
- `PixelDiffusion` as the Lightning model
- `DenoisingDiffusionProcess` internals for the diffusion model
- `dataset/` containing the raw full aperture (FA) and subaperture (SA) data
- `train.py` as the Pytorch Lightning training pipeline for conditional pixel-space diffusion on SAR subaperture data

At a high level, the model learns to map: 
- **condition input** `x`: retained subaperture channels
- to **target output** `y`: the original full aperture

using a conditional denoising diffusion process with a configurable U-Net backbone.

--- 
## Current State (April 2026)

### What is implemented:
- End-to-end training with PyTorch Lightning (`train.py`).
- Config-driven setup for data/model/trainer (`configs/config.yaml`).
- Conditional diffusion model (`src/PixelDiffusion.py`) with:
  - configurable noise schedule (`linear` or `cosine`),
  - selectable loss (`mse`, `mae`, or hybrid MS-SSIM + base loss),
  - optional EMA shadow model for evaluation logging.
- SAR data backend via `SafetensorSARDataset` wrapped by `PairedImageDataset`.
- Validation-time reconstruction metrics/logging:
  - `val_loss`,
  - `val_recon_psnr`, `val_recon_ssim`, `val_recon_l1`, `val_recon_phase_coherence`,
  - optional EMA versions of reconstruction metrics when EMA is enabled.
- Separate evaluation script (`evaluate.py`) for SAR-focused analyses (IRF, KDE distance, phase coherence, etc.).

---

## Repository Layout
- `train.py` — training entrypoint.
- `evaluate.py` — offline evaluation script for saved checkpoints.
- `configs/config.yaml` — main runtime configuration.
- `src/PixelDiffusion.py` — LightningModule and training/validation logic.
- `data/datamodule.py` — Lightning DataModule.
- `data/dataset.py` — adapter dataset that selects backend/fallback.
- `data/finaldataset.py` — SAR dataset implementation used in normal runs.

---

## Data Interface
The training/evaluation model expects each batch to be:
```python
return x, y 
```
with shape `[B, C, H, W]` for both tensors.

In the current SAR pipeline:
- channels are built from interleaved magnitude/phase components,
- values are normalised/clamped into `[-1, 1]` inside the dataset backend,
- model `input_T` / `output_T` currently map to `[-1, 1]` (no additional remapping).

---

## Configuration Notes
Main config: `configs/config.yaml`.
Important fields:
- `data.root_dir`: path to patch dataset root.
- `data.subaperture_config`:
  - `input_indices`
  - `output_indices`
  - `active_polarizations`
- `model.in_channels` / `model.out_channels`: must match prepared dataset channels.
- `model.unet.channels`: typically `in_channels + out_channels` for concatenated conditional diffusion input.
- `model.loss_fn`: `mse`, `mae`, or `hybrid`.
- `model.ema.enabled`: enables EMA shadow model + EMA validation metrics.
- `trainer.checkpointing`: controls saved checkpoints (`save_top_k`, `save_last`, etc.).

---


## Install Requirements

Use your preferred environment manager and install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you do not want W&B, you can still run with offline/disabled mode.

---

## Train

```bash
python3 train.py --config configs/config.yaml
```

Resume from a checkpoint:
```bash
python3 train.py --config configs/config.yaml --resume-from-ckpt checkpoints/last.ckpt
```

Main config file: [`configs/config.yaml`](/work/code/dif_img_rec/configs/config.yaml)

### Required fields

```yaml
data:
  loader:
    batch_size: 4
    num_workers: 0
    pin_memory: false

model:
  in_channels: 2
  out_channels: 2
  num_timesteps: 1000
  schedule: linear
  noise_offset: 0.0
  unet:
    dim: 48
    dim_mults: [1, 2, 4, 8]
    channels: 4
    out_dim: 2
```

Notes:
- `unet.channels` should usually be `in_channels + out_channels` for conditional concat input.
- `unet.out_dim` should match `out_channels`.
- `noise_offset` adds an optional low-frequency offset term to training noise (set `0.0` to disable).
- Smaller model: reduce `unet.dim` (for example `48` instead of `64`).
- Current downsized setup reduces the UNet from about `55M` to about `32M` parameters
  (mainly by lowering base width via `unet.dim` and using the configured channel sizes).

### Optimization and scheduler

```yaml
optimization:
  lr: 0.0001
  reduce_lr_on_plateau:
    factor: 0.5
    patience: 10
```

Scheduler monitors `val_loss`.

### Trainer

```yaml
trainer:
  max_epochs: 100
  accelerator: auto
  devices: 1
  precision: 32
  log_every_n_steps: 10
  enable_checkpointing: true
  checkpointing:
    dirpath: checkpoints
    filename: "{epoch:03d}--{val_loss:.4f}"
    monitor: val_loss
    mode: min
    save_top_k: 3
    save_last: true
    every_n_epochs: 1
    auto_inset_metric_name: false
```

---

## Evaluate a Saved Checkpoint

```bash
python3 evaluate.py \
  --config config/config.yaml \
  --checkpoint /path/to/model.ckpt \
  --split test
```

Reported metrics:
- `psf_azimuth_rel_err`: relative 3 dB width error in azimuth.
- `psf_range_rel_err`: relative 3 dB width error in range.
- `enl_rel_err`: relative ENL error on intensity.
- `kde_js_disntance`: Jensen-Shannon distance between KDEs of reconstructed vs target intensity.
- `rel_phase_mae_rad`: mean absolute wrapped error of relative phase (radians).
- `rel_phase_coherence`: circular coherence of relative phase errors (1.0 is best).

Optional JSON output:
```bash
python3 evaluate.py \
  --config config/config.yaml \
  --checkpoint /path/to/model.ckpt \
  --split test \
  --output-json reports/sar_eval.json
```

---
## Logging
Wights & Biases settings are under `loggingin.wandb` in config.

Config section:

```yaml
logging:
  wandb:
    project: dif_img_rec
    name: null
    save_dir: logs
    log_model: false
```

### Option A: online logging

```bash
wandb login
export WANDB_API_KEY=...   # if not already logged in
```

### Option B: offline logging

```bash
export WANDB_MODE=offline
```

### Option C: disable W&B entirely

Set env var before run:

```bash
export WANDB_DISABLED=true
```
---

## Practical Checklist Before Long Training Runs

- Confirm `data.root_dir` exists and contains expected patch structure.
- Verify `model.in_channels` / `model.out_channels` match generated tensors.
- Ensure `model.unet.channels == in_channels + out_channels` (unless intentionally different).
- Run a short sanity run and check validation reconstruction metrics/logged images.
- Validate checkpoint save path and naming in `trainer.checkpointing`.
