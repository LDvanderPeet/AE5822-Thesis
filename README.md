# Conditional Pixel Diffusion (Minimal Scaffold)

This repo is a minimal training scaffold for **conditional pixel-space diffusion** using:
- `PixelDiffusionConditional` as the Lightning model
- `DenoisingDiffusionProcess` internals for the diffusion model
- a placeholder `data/` pipeline that currently returns random tensors

It is intentionally not production-ready yet. This README explains what to change to make it fully functional on real data.

## Current State

What already works:
- Training loop in [`train.py`](/work/code/dif_img_rec/train.py)
- Config-driven setup via [`configs/config.yaml`](/work/code/dif_img_rec/configs/config.yaml)
- Conditional diffusion model wiring
- Validation logging to W&B:
  - `train_loss`, `val_loss`
  - one reconstructed example (`x`, `pred`, `y`) as a matplotlib figure
  - reconstruction metrics on that prediction: `val_recon_psnr`, `val_recon_ssim`, `val_recon_l1`

What is still placeholder:
- Dataset/data module in [`data/`](/work/code/dif_img_rec/data)
    - defaults to random `(x, y)` with shape `(2, 128, 128)` and fixed length `(1000)`
    - can now auto-switch to the SAR loader in `data/sardataset.py` when `data.root_dir` and `data.subaperture_config` are provided in config

## 1. Install Requirements

Use your preferred environment manager and install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you do not want W&B, you can still run with offline/disabled mode (see section 5).

## 2. Implement Real Dataset + DataModule

Edit these files:
- [`data/dataset.py`](/work/code/dif_img_rec/data/dataset.py)
- [`data/datamodule.py`](/work/code/dif_img_rec/data/datamodule.py)

### What to implement

In `PairedImageDataset`:
- load your paired samples `(x, y)` from disk
- return tensors shaped `[C, H, W]`
- keep `__getitem__` output exactly as:

```python
return x, y
```

In `PairedDataModule`:
- replace placeholder dataset construction in `setup()` with your train/val/test datasets
- keep dataloader settings wired from config (`batch_size`, `num_workers`, `pin_memory`)

### Important shape/channel rule

Model input/output channels come from config:
- `model.in_channels` = channels for `x` (condition)
- `model.out_channels` = channels for `y` (target)

Your dataset must match this.

## 3. Implement/Verify Normalization

Current model logic in [`src/PixelDiffusion.py`](/work/code/dif_img_rec/src/PixelDiffusion.py):
- `input_T`: assumes tensors are in `[0, 1]`, then maps to `[-1, 1]`
- `output_T`: maps model output from `[-1, 1]` back to `[0, 1]`

So your dataset should output `x` and `y` in `[0, 1]`.

If your data uses different scaling (for example z-score), you should adapt `input_T`/`output_T` accordingly.

## 4. Edit Config

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
  unet:
    dim: 48
    dim_mults: [1, 2, 4, 8]
    channels: 4
    out_dim: 2
```

Notes:
- `unet.channels` should usually be `in_channels + out_channels` for conditional concat input.
- `unet.out_dim` should match `out_channels`.
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
  enable_checkpointing: false
```

## 5. Optional W&B Setup

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

## 6. Run Training

```bash
python3 train.py --config configs/config.yaml
```

## 7. Sanity Checklist Before Real Training

- dataset returns `(x, y)` tensors, not file paths
- tensors are float and normalized consistently
- channel counts match config
- first validation batch logs reconstruction image and metrics
- loss decreases on a small overfit subset

## Project Files

- Training entrypoint: [`train.py`](/work/code/dif_img_rec/train.py)
- Config: [`configs/config.yaml`](/work/code/dif_img_rec/configs/config.yaml)
- Dataset scaffold: [`data/dataset.py`](/work/code/dif_img_rec/data/dataset.py)
- DataModule scaffold: [`data/datamodule.py`](/work/code/dif_img_rec/data/datamodule.py)
- Lightning model: [`src/PixelDiffusion.py`](/work/code/dif_img_rec/src/PixelDiffusion.py)
- Diffusion internals: [`src/DenoisingDiffusionProcess/`](/work/code/dif_img_rec/src/DenoisingDiffusionProcess)
