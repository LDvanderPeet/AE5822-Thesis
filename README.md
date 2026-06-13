# DiffASR: a Diffusion Mode for Synthetic Aperture Radar Super-Resolution

This repository contains a PyTorch Lightning training pipeline for **conditional pixel-space diffusion** on 
complex-valued SAR data. The azimuth super-resolution task is formulated as:
```text
condition x: reduced-bandwidth complex SAR patch
target y:    higher-bandwidth complex SAR patch, usually the 100% reference
```

using a conditional denoising diffusion process with a configurable U-Net backbone.

The model learns to recover the 100% bandwidth image from lower-bandwidth versions produced by azimuth/Doppler low-pass 
filtering. Complex SLC data is represented with interleaved real and imaginary channels, so both amplitude and phase are
available to the model and to the evaluation metrics.

The active training stack supports two model families:

- `PixelDiffusionConditional`: conditional DDPM-style pixel diffusion with a
  ConvNeXt U-Net backbone.
- `DC2SCNLightning`: deterministic DC2SCN-style baseline with dense dilated
  convolution blocks and a phase-aware physical loss.

The main entry points are:

- `data/patch_generation.py`: convert H5 SLC files into 20-channel
  `.safetensors` patches containing all bandwidth levels.
- `train.py`: train either the diffusion model or the DC2SCN baseline.
- `evaluate.py`: run offline SAR-focused evaluation on a saved checkpoint.
- `configs/config.yaml`: central runtime configuration.
--- 

## Important Terminology Change

Older code and logs still use names such as `subaperture_config`, `FA`, and
`SA1` because the project originally targeted subaperture reconstruction. In the
current super-resolution formulation, those names should be interpreted as
bandwidth levels.

The dataset index mapping is now:

| Logical index | Bandwidth represented | Historical label that may appear in code/logs |
| --- | ---: | --- |
| `0` | `100%` | `FA` |
| `1` | `87.5%` | `SA1` |
| `2` | `75%` | `SA2` |
| `3` | `62.5%` | `SA3` |
| `4` | `50%` | `SA4` |

For example, the active config currently contains:

```yaml
data:
  subaperture_config:
    input_indices: [2]
    output_indices: [0]
    active_polarizations: ["VV"]
```

That means:

```text
input x  = 75% bandwidth VV complex patch
target y = 100% bandwidth VV complex patch
```

If W&B or validation plots label the input as `SA2`, read that label as
`75% bandwidth`. If the target is labeled `FA`, read it as `100% bandwidth`.

---

## Framework Summary

### Data representation

Each model batch is a pair:

```python
x, y = batch
```

Both tensors use shape:

```text
[B, C, H, W]
```

where:

- `B` is the batch size.
- `C` is the number of real-valued channels.
- `H` and `W` are patch height and width.
- Complex values are stored as interleaved real/imaginary channels.

For one polarization and one bandwidth level:

```text
[I, Q] -> 2 channels
```

For two polarizations and one bandwidth level:

```text
[I_VV, Q_VV, I_VH, Q_VH] -> 4 channels
```

For multiple input bandwidth levels, channels are concatenated in the selected
index order.

### Active safetensor channel layout

`data/patch_generation.py` writes each patch as one tensor named `x` with shape:

```text
[20, patch_size, patch_size]
```

The 20 channels are 5 bandwidth levels times 2 polarizations times 2 I/Q
components.

| Bandwidth index | Bandwidth | Channel block | Channels inside block |
| --- | ---: | --- | --- |
| `0` | `100%` | `0:4` | `I_VV, Q_VV, I_VH, Q_VH` |
| `1` | `87.5%` | `4:8` | `I_VV, Q_VV, I_VH, Q_VH` |
| `2` | `75%` | `8:12` | `I_VV, Q_VV, I_VH, Q_VH` |
| `3` | `62.5%` | `12:16` | `I_VV, Q_VV, I_VH, Q_VH` |
| `4` | `50%` | `16:20` | `I_VV, Q_VV, I_VH, Q_VH` |

`data/finaldataset.py` maps logical bandwidth indices to physical channels with:

```python
base_offset = bandwidth_index * 4
```

Then it selects the requested polarizations. The current safetensor dataset
implementation supports `VV` and `VH`.

### Channel count rules

Use this formula when changing `input_indices`, `output_indices`, or
`active_polarizations`:

```text
channels = number_of_bandwidth_indices * number_of_polarizations * 2
```

Examples:

| Task | Config indices | Polarizations | `model.in_channels` | `model.out_channels` |
| --- | --- | --- | ---: | ---: |
| `75% -> 100%` | input `[2]`, output `[0]` | `["VV"]` | `2` | `2` |
| `50% -> 100%` | input `[4]`, output `[0]` | `["VV"]` | `2` | `2` |
| `75%, 62.5%, 50% -> 100%` | input `[2, 3, 4]`, output `[0]` | `["VV"]` | `6` | `2` |
| `75% -> 100%`, dual-pol | input `[2]`, output `[0]` | `["VV", "VH"]` | `4` | `4` |
| Multi-input, dual-pol | input `[2, 3, 4]`, output `[0]` | `["VV", "VH"]` | `12` | `4` |

For the diffusion model, `model.unet.channels` should normally equal:

```text
model.in_channels + model.out_channels
```

because the U-Net receives the noisy target estimate concatenated with the
condition image.

---

## Repository Layout

```text
.
|-- README.md
|-- configs/
|   `-- config.yaml
|-- data/
|   |-- patch_generation.py
|   |-- datamodule.py
|   |-- dataset.py
|   |-- finaldataset.py
|   |-- sardataset.py
|   |-- data_cleaning.py
|   `-- data_inspection.py
|-- src/
|   |-- PixelDiffusion.py
|   |-- DC2SCN.py
|   |-- EMA.py
|   |-- callbacks.py
|   `-- DenoisingDiffusionProcess/
|-- train.py
|-- evaluate.py
|-- requirements.txt
`-- archive/
```

Important files:

- `configs/config.yaml`: main configuration file for data, model, trainer,
  logging, and evaluation.
- `data/patch_generation.py`: generates the current super-resolution
  `.safetensors` dataset from H5 SLC inputs.
- `data/finaldataset.py`: active safetensor SAR dataset backend.
- `data/dataset.py`: adapter returning only `(x, y)` to Lightning.
- `data/datamodule.py`: Lightning `DataModule`, including overfit-mode support.
- `src/PixelDiffusion.py`: conditional diffusion Lightning module, hybrid loss,
  EMA support, validation metrics, and inverse normalization.
- `src/DenoisingDiffusionProcess/`: Gaussian forward process, beta schedules,
  DDPM sampler, DDIM sampler class, and ConvNeXt U-Net backbone.
- `src/DC2SCN.py`: deterministic baseline model and Lightning wrapper.
- `src/callbacks.py`: W&B image plots, KDE plots, histograms, and phase plots.
- `train.py`: builds data, model, callbacks, W&B logger, checkpoints, and trainer.
- `evaluate.py`: checkpoint evaluation script for IRF, KDE, phase coherence, ENL,
  and hypothesis-test summaries.
- `archive/`: older implementation and documentation retained for reference.
  It is not the active training path.

---

## Installation

Use Python 3.10 or newer. The code uses modern type syntax and `zip(...,
strict=False)`, so older Python versions are not recommended.

Create and activate an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the core requirements:

```bash
pip install -r requirements.txt
```

The data generation and evaluation scripts also import packages that may already
be available on the cluster environment but are not all listed in
`requirements.txt`. If your environment is missing them, install:

```bash
pip install safetensors scipy h5py joblib zarr
```

For GPU training, install a PyTorch build that matches the CUDA version on your
machine or cluster.

---

## Data Preparation

### Expected source H5 structure

`data/patch_generation.py` expects H5 files with a group named `bands` by
default. Inside that group it expects I/Q datasets for the supported
polarizations:

```text
bands/i_VV
bands/q_VV
bands/i_VH
bands/q_VH
```

The script recursively searches for `.h5` files unless `--no_recursive` is
provided.


### Generate safetensor patches

Run from the repository root:

```bash
python3 data/patch_generation.py \
  --in_dir /path/to/source_h5_root \
  --out_dir /path/to/safetensors_train \
  --patch_size 256 \
  --stride 256 \
  --block_size 1024 \
  --group bands \
  --ratio_thr 0.5 \
  --zero_thr 0.0
```

The output directory is organized by product:

```text
/path/to/safetensors_train/
|-- PRODUCT_1/
|   |-- tile_name__0_0.safetensors
|   |-- tile_name__0_256.safetensors
|   `-- ...
|-- PRODUCT_2/
|   `-- ...
`-- ...
```

Each `.safetensors` file contains:

```python
{"x": tensor}
```

where `tensor` has shape:

```text
[20, 256, 256]
```

if `patch_size=256`.

### How degraded bandwidth levels are generated

For each source patch, the generator:

1. Reads the complex `VV` and `VH` channels.
2. Stores the original complex patch as logical index `0`, the `100%` bandwidth
   reference.
3. Applies a top-hat low-pass mask in the FFT domain for ratios:

```text
0.875, 0.750, 0.625, 0.500
```

4. Stores those degraded versions as logical indices `1`, `2`, `3`, and `4`.
5. Writes all levels into one 20-channel tensor.

The no-data filtering uses `VV` intensity:

```text
intensity_vv = I_VV^2 + Q_VV^2
```

If the fraction of pixels with intensity less than or equal to `zero_thr` is
greater than `ratio_thr`, the patch is skipped.

### Parallel generation

The patch generator uses `joblib`. By default it uses the available CPU count.
To control the worker count:

```bash
JOBS=8 python3 data/patch_generation.py \
  --in_dir /path/to/source_h5_root \
  --out_dir /path/to/safetensors_train
```

---

## Dataset Loading

The active training path is:

```text
train.py
-> PairedDataModule
-> PairedImageDataset
-> SafetensorSARDataset
```

`SafetensorSARDataset` reads the generated `.safetensors` files from
`data.root_dir`.

### Split behavior

For each product directory, `SafetensorSARDataset` sorts patch filenames and
uses a spatial/geographic split:

```text
first 70%       -> train
middle 15%      -> valid
last 15%        -> test
```

A small filename-index buffer is inserted between train/valid and valid/test.
The implementation currently uses a fixed split in `data/finaldataset.py`; the
`data.split` section in `configs/config.yaml` is present for configuration
clarity but is not currently consumed by the safetensor backend.

Current dataset caps:

```text
train: 100000 patches
valid: 10000 patches
test:  10000 patches
```

### Augmentation

Training uses strip-safe flips only:

- identity
- vertical flip
- horizontal flip
- vertical plus horizontal flip

The code intentionally avoids transposes and 90/270 degree rotations because
those operations can scramble SAR azimuth/range physics.

Validation and test loading do not use augmentation.

### Normalization

The dataset applies phase-preserving complex amplitude compression.

For each I/Q pair:

```text
A      = sqrt(I^2 + Q^2)
A_comp = log1p(A) / log1p(global_max)
I_out  = A_comp * I / max(A, eps)
Q_out  = A_comp * Q / max(A, eps)
```

The resulting values are clamped to:

```text
[-1, 1]
```

This keeps phase direction while compressing the heavy-tailed SAR magnitude
distribution. The model classes implement the inverse transform for physical
metrics and plots.

Set `data.global_max` to a scale appropriate for the generated dataset. If it
is too small, many values will saturate at `[-1, 1]`; if it is too large, most
samples will occupy a narrow range.

---

## Configuration Guide

Main config:

```text
configs/config.yaml
```

### Seed

```yaml
seed: 27
```

Used by Lightning and by dataset shuffling.

### Data

```yaml
data:
  root_dir: "/shared/home/lvanderpeet/AE5822-Thesis/dataset/safetensors_train"
  image_height: 256
  image_width: 256
  global_max: 300
  subaperture_config:
    input_indices: [2]
    output_indices: [0]
    active_polarizations: ["VV"]
  loader:
    batch_size: 16
    num_workers: 8
    pin_memory: true
  split:
    train: 0.7
    valid: 0.2
    test: 0.1
  overfit:
    enabled: false
    source_split: train
    num_patches: 128
```

Important fields:

- `root_dir`: path to generated `.safetensors` dataset root.
- `image_height`, `image_width`: documentation/runtime assumptions for patch
  size. The safetensor backend does not resize; patch size is determined during
  generation.
- `global_max`: normalization scale for phase-preserving signed-log compression.
- `subaperture_config.input_indices`: historical name; now means input
  bandwidth indices.
- `subaperture_config.output_indices`: historical name; now means target
  bandwidth indices.
- `active_polarizations`: currently `VV`, `VH`, or both for the safetensor
  backend.
- `loader.batch_size`: Lightning DataLoader batch size.
- `loader.num_workers`: DataLoader worker count.
- `loader.pin_memory`: usually `true` for CUDA training.
- `overfit`: diagnostic mode that reuses the same subset for train and valid.

### Model selection

The active training path is:

```text
train.py
-> PairedDataModule
-> PairedImageDataset
-> SafetensorSARDataset
```

`SafetensorSARDataset` reads the generated `.safetensors` files from
`data.root_dir`.

### Split behavior

For each product directory, `SafetensorSARDataset` sorts patch filenames and
uses a spatial/geographic split:

```text
first 70%       -> train
middle 15%      -> valid
last 15%        -> test
```

A small filename-index buffer is inserted between train/valid and valid/test.
The implementation currently uses a fixed split in `data/finaldataset.py`; the
`data.split` section in `configs/config.yaml` is present for configuration
clarity but is not currently consumed by the safetensor backend.

Current dataset caps:

```text
train: 100000 patches
valid: 10000 patches
test:  10000 patches
```

### Augmentation

Training uses strip-safe flips only:

- identity
- vertical flip
- horizontal flip
- vertical plus horizontal flip

The code intentionally avoids transposes and 90/270 degree rotations because
those operations can scramble SAR azimuth/range physics.

Validation and test loading do not use augmentation.

### Normalization

The dataset applies phase-preserving complex amplitude compression.

For each I/Q pair:

```text
A      = sqrt(I^2 + Q^2)
A_comp = log1p(A) / log1p(global_max)
I_out  = A_comp * I / max(A, eps)
Q_out  = A_comp * Q / max(A, eps)
```

The resulting values are clamped to:

```text
[-1, 1]
```

This keeps phase direction while compressing the heavy-tailed SAR magnitude
distribution. The model classes implement the inverse transform for physical
metrics and plots.

Set `data.global_max` to a scale appropriate for the generated dataset. If it
is too small, many values will saturate at `[-1, 1]`; if it is too large, most
samples will occupy a narrow range.

---

## Configuration Guide

Main config:

```text
configs/config.yaml
```

### Seed

```yaml
seed: 27
```

Used by Lightning and by dataset shuffling.

### Data

```yaml
data:
  root_dir: "/shared/home/lvanderpeet/AE5822-Thesis/dataset/safetensors_train"
  image_height: 256
  image_width: 256
  global_max: 300
  subaperture_config:
    input_indices: [2]
    output_indices: [0]
    active_polarizations: ["VV"]
  loader:
    batch_size: 16
    num_workers: 8
    pin_memory: true
  split:
    train: 0.7
    valid: 0.2
    test: 0.1
  overfit:
    enabled: false
    source_split: train
    num_patches: 128
```

Important fields:

- `root_dir`: path to generated `.safetensors` dataset root.
- `image_height`, `image_width`: documentation/runtime assumptions for patch
  size. The safetensor backend does not resize; patch size is determined during
  generation.
- `global_max`: normalization scale for phase-preserving signed-log compression.
- `subaperture_config.input_indices`: historical name; now means input
  bandwidth indices.
- `subaperture_config.output_indices`: historical name; now means target
  bandwidth indices.
- `active_polarizations`: currently `VV`, `VH`, or both for the safetensor
  backend.
- `loader.batch_size`: Lightning DataLoader batch size.
- `loader.num_workers`: DataLoader worker count.
- `loader.pin_memory`: usually `true` for CUDA training.
- `overfit`: diagnostic mode that reuses the same subset for train and valid.

### Model selection

```yaml
model:
  type: diffusion
```

Supported values:

- `diffusion`: conditional pixel-space diffusion.
- `dc2scn`: deterministic DC2SCN baseline.

Any value other than `dc2scn` is treated as diffusion by `train.py`.

### Diffusion model

```yaml
model:
  in_channels: 2
  out_channels: 2
  num_timesteps: 1000
  schedule: cosine
  loss_fn: hybrid
  hybrid_base_loss: mse
  hybrid_ms_ssim_weight: 1.0
  hybrid_phase_weight: 2.0
  hybrid_ms_ssim_t_limit: 150
  noise_offset: 0.0
  ema:
    enabled: true
    beta: 0.9999
    update_every: 1
    update_after_step: 10000
  unet:
    dim: 64
    dim_mults: [1, 2, 4, 8]
    channels: 4
    out_dim: 2
    dropout: 0.1
```

Important fields:

- `in_channels`: channels in `x`.
- `out_channels`: channels in `y`.
- `num_timesteps`: number of DDPM denoising steps.
- `schedule`: beta schedule. Implemented schedules include `linear`,
  `cosine`, `quadratic`, and `sigmoid`.
- `loss_fn`: `mse`, `mae`, or `hybrid`.
- `noise_offset`: optional low-frequency noise offset during training. Use
  `0.0` to disable.
- `ema.enabled`: maintain an exponential moving average copy of the diffusion
  process for side-by-side validation.
- `unet.dim`: base U-Net width.
- `unet.dim_mults`: U-Net level multipliers.
- `unet.channels`: input channels to the U-Net. For conditional diffusion this
  should normally be `in_channels + out_channels`.
- `unet.out_dim`: U-Net output channels. This should match `out_channels`.
- `unet.dropout`: dropout inside ConvNeXt blocks and linear attention.

### Hybrid diffusion loss

When `model.loss_fn: hybrid`, training still learns to predict diffusion noise,
but low-noise timesteps can receive additional image-domain physical penalties.

For timesteps:

```text
t <= hybrid_ms_ssim_t_limit
```

the hybrid loss adds:

- MS-SSIM loss on reconstructed real and imaginary components.
- A phase-aware complex penalty in physical scale.

For later timesteps, it falls back to the configured base loss:

```yaml
hybrid_base_loss: mse  # or mae
```

Logged diagnostic terms include:

```text
train_hybrid/base_loss
train_hybrid/ms_ssim_loss
train_hybrid/phase_loss
train_hybrid/advanced_loss
train_hybrid/loss
train_hybrid/advanced_t_fraction
```

with corresponding `val_hybrid/...` metrics.


### DC2SCN Baseline

```yaml
model:
  type: dc2scn
  in_channels: 2
  out_channels: 2
  dc2scn:
    width: 64
    num_blocks: 6
    growth_rate: 32
```

The DC2SCN baseline maps `x` directly to `y`. Its training loss is computed in
physical complex scale:

```text
total_loss = magnitude_L1 + 0.5 * phase_unit_vector_loss
```

This baseline is useful for comparison against the diffusion model because it
does not sample a reverse denoising chain.

### Optimization


Any value other than `dc2scn` is treated as diffusion by `train.py`.

### Diffusion model

```yaml
model:
  in_channels: 2
  out_channels: 2
  num_timesteps: 1000
  schedule: cosine
  loss_fn: hybrid
  hybrid_base_loss: mse
  hybrid_ms_ssim_weight: 1.0
  hybrid_phase_weight: 2.0
  hybrid_ms_ssim_t_limit: 150
  noise_offset: 0.0
  ema:
    enabled: true
    beta: 0.9999
    update_every: 1
    update_after_step: 10000
  unet:
    dim: 64
    dim_mults: [1, 2, 4, 8]
    channels: 4
    out_dim: 2
    dropout: 0.1
```

Important fields:

- `in_channels`: channels in `x`.
- `out_channels`: channels in `y`.
- `num_timesteps`: number of DDPM denoising steps.
- `schedule`: beta schedule. Implemented schedules include `linear`,
  `cosine`, `quadratic`, and `sigmoid`.
- `loss_fn`: `mse`, `mae`, or `hybrid`.
- `noise_offset`: optional low-frequency noise offset during training. Use
  `0.0` to disable.
- `ema.enabled`: maintain an exponential moving average copy of the diffusion
  process for side-by-side validation.
- `unet.dim`: base U-Net width.
- `unet.dim_mults`: U-Net level multipliers.
- `unet.channels`: input channels to the U-Net. For conditional diffusion this
  should normally be `in_channels + out_channels`.
- `unet.out_dim`: U-Net output channels. This should match `out_channels`.
- `unet.dropout`: dropout inside ConvNeXt blocks and linear attention.

### Hybrid diffusion loss

When `model.loss_fn: hybrid`, training still learns to predict diffusion noise,
but low-noise timesteps can receive additional image-domain physical penalties.

For timesteps:

```text
t <= hybrid_ms_ssim_t_limit
```

the hybrid loss adds:

- MS-SSIM loss on reconstructed real and imaginary components.
- A phase-aware complex penalty in physical scale.

For later timesteps, it falls back to the configured base loss:

```yaml
hybrid_base_loss: mse  # or mae
```

Logged diagnostic terms include:

```text
train_hybrid/base_loss
train_hybrid/ms_ssim_loss
train_hybrid/phase_loss
train_hybrid/advanced_loss
train_hybrid/loss
train_hybrid/advanced_t_fraction
```

with corresponding `val_hybrid/...` metrics.

### DC2SCN baseline

```yaml
model:
  type: dc2scn
  in_channels: 2
  out_channels: 2
  dc2scn:
    width: 64
    num_blocks: 6
    growth_rate: 32
```

The DC2SCN baseline maps `x` directly to `y`. Its training loss is computed in
physical complex scale:

```text
total_loss = magnitude_L1 + 0.5 * phase_unit_vector_loss
```

This baseline is useful for comparison against the diffusion model because it
does not sample a reverse denoising chain.

### Optimization

```yaml
optimization:
  lr: 0.00002
  reduce_lr_on_plateau:
    factor: 0.5
    patience: 1
```

Both model families use AdamW. The scheduler is `ReduceLROnPlateau` and monitors
`val_loss`.

### Trainer

```yaml
trainer:
  max_epochs: 100
  accelerator: auto
  devices: 1
  precision: 32
  gradient_clip_val: 1.0
  gradient_clip_algorithm: norm
  log_every_n_steps: 10
  enable_checkpointing: true
  limit_val_batches: 0.8
  checkpointing:
    dirpath: checkpoints
    filename: "refine-{epoch:03d}-{val_recon_phase_coherence:.4f}"
    monitor: "val_recon_phase_coherence"
    mode: "max"
    save_top_k: 3
    save_last: true
    every_n_epochs: 1
    auto_insert_metric_name: false
```

Notes:

- `accelerator: auto` lets Lightning choose CPU/GPU/MPS.
- `devices: 1` uses a single device.
- `limit_val_batches` can be a float fraction or an integer batch count.
- Checkpoints are written under a timestamped run subdirectory:

```text
checkpoints/YYYYMMDD-HHMM-run_name/
```

### Logging

```yaml
logging:
  wandb:
    project: dif_img_rec
    name: diffusion_ASR_refinement
    save_dir: logs
    log_model: false
    save_config_file: true
    config_artifact_name: null
  mode: train
```

`train.py` always creates a W&B logger. Local W&B files are written under:

```text
logs/YYYYMMDD-HHMM-run_name/
```

If `save_config_file: true`, the exact YAML file passed through `--config` is
uploaded to W&B as a run file and as a config artifact.

---


When `model.loss_fn: hybrid`, training still learns to predict diffusion noise,
but low-noise timesteps can receive additional image-domain physical penalties.

For timesteps:

```text
t <= hybrid_ms_ssim_t_limit
```

the hybrid loss adds:

- MS-SSIM loss on reconstructed real and imaginary components.
- A phase-aware complex penalty in physical scale.

For later timesteps, it falls back to the configured base loss:

```yaml
hybrid_base_loss: mse  # or mae
```

Logged diagnostic terms include:

```text
train_hybrid/base_loss
train_hybrid/ms_ssim_loss
train_hybrid/phase_loss
train_hybrid/advanced_loss
train_hybrid/loss
train_hybrid/advanced_t_fraction
```

with corresponding `val_hybrid/...` metrics.

### DC2SCN baseline

```yaml
model:
  type: dc2scn
  in_channels: 2
  out_channels: 2
  dc2scn:
    width: 64
    num_blocks: 6
    growth_rate: 32
```

The DC2SCN baseline maps `x` directly to `y`. Its training loss is computed in
physical complex scale:

```text
total_loss = magnitude_L1 + 0.5 * phase_unit_vector_loss
```

This baseline is useful for comparison against the diffusion model because it
does not sample a reverse denoising chain.

### Optimization

```yaml
optimization:
  lr: 0.00002
  reduce_lr_on_plateau:
    factor: 0.5
    patience: 1
```

Both model families use AdamW. The scheduler is `ReduceLROnPlateau` and monitors
`val_loss`.

### Trainer

```yaml
trainer:
  max_epochs: 100
  accelerator: auto
  devices: 1
  precision: 32
  gradient_clip_val: 1.0
  gradient_clip_algorithm: norm
  log_every_n_steps: 10
  enable_checkpointing: true
  limit_val_batches: 0.8
  checkpointing:
    dirpath: checkpoints
    filename: "refine-{epoch:03d}-{val_recon_phase_coherence:.4f}"
    monitor: "val_recon_phase_coherence"
    mode: "max"
    save_top_k: 3
    save_last: true
    every_n_epochs: 1
    auto_insert_metric_name: false
```

Notes:

- `accelerator: auto` lets Lightning choose CPU/GPU/MPS.
- `devices: 1` uses a single device.
- `limit_val_batches` can be a float fraction or an integer batch count.
- Checkpoints are written under a timestamped run subdirectory:

```text
checkpoints/YYYYMMDD-HHMM-run_name/
```

### Logging

```yaml
logging:
  wandb:
    project: dif_img_rec
    name: diffusion_ASR_refinement
    save_dir: logs
    log_model: false
    save_config_file: true
    config_artifact_name: null
  mode: train
```

`train.py` always creates a W&B logger. Local W&B files are written under:

```text
logs/YYYYMMDD-HHMM-run_name/
```

If `save_config_file: true`, the exact YAML file passed through `--config` is
uploaded to W&B as a run file and as a config artifact.

---

## Training

Run from the repository root.

### Standard training

```bash
python3 train.py --config configs/config.yaml
```

### Limit validation for a quick run

```bash
python3 train.py \
  --config configs/config.yaml \
  --limit-val-batches 0.1
```

This overrides `trainer.limit_val_batches` from the config.

### Load weights from a checkpoint

```bash
python3 train.py \
  --config configs/config.yaml \
  --resume-from-ckpt /path/to/checkpoint.ckpt
```

Important: this code path loads the model `state_dict` only. It does not restore
the full Lightning trainer state, optimizer state, scheduler state, epoch, or
global step. Treat it as weight initialization or fine-tuning, not as a bitwise
training resume.

### W&B modes

Online:

```bash
wandb login
python3 train.py --config configs/config.yaml
```

Offline:

```bash
export WANDB_MODE=offline
python3 train.py --config configs/config.yaml
```

Disabled:

```bash
export WANDB_DISABLED=true
python3 train.py --config configs/config.yaml
```

### Expected training outputs

Training creates:

```text
logs/YYYYMMDD-HHMM-run_name/
checkpoints/YYYYMMDD-HHMM-run_name/
```

Typical checkpoint files include:

```text
last.ckpt
refine-<epoch>-<metric>.ckpt
```

depending on `trainer.checkpointing.filename`.

---

## Validation Metrics and Plots

During training, both model families log:

```text
train_loss
val_loss
val_recon_psnr
val_recon_ssim
val_recon_l1
val_recon_phase_coherence
val_recon_phase_error
```

The diffusion model additionally logs:

```text
val_recon_rmse
val_recon_amp_corr
```

and, when EMA is enabled:

```text
val_recon_psnr_ema
val_recon_ssim_ema
val_recon_l1_ema
val_recon_rmse_ema
val_recon_amp_corr_ema
val_recon_phase_coherence_ema
val_recon_phase_error_ema
```

`WandBPlottingCallback` logs:

- `val/reconstruction`: input bandwidth panels, predicted target, optional EMA
  prediction, and target.
- `val/magnitude_kde_db`: KDE overlay of predicted and target magnitude in dB.
- `val/reconstruction_histograms`: real, imaginary, and magnitude histograms in
  physical scale.
- `val/phase_error_histogram`: wrapped phase error histogram.

The model computes reconstruction metrics after inverse normalization so phase
and magnitude metrics operate in physical SAR scale where applicable.

---

### Load weights from a checkpoint

```bash
python3 train.py \
  --config configs/config.yaml \
  --resume-from-ckpt /path/to/checkpoint.ckpt
```

Important: this code path loads the model `state_dict` only. It does not restore
the full Lightning trainer state, optimizer state, scheduler state, epoch, or
global step. Treat it as weight initialization or fine-tuning, not as a bitwise
training resume.

### W&B modes

Online:

```bash
wandb login
python3 train.py --config configs/config.yaml
```

Offline:

```bash
export WANDB_MODE=offline
python3 train.py --config configs/config.yaml
```

Disabled:

```bash
export WANDB_DISABLED=true
python3 train.py --config configs/config.yaml
```

### Expected training outputs

Training creates:

```text
logs/YYYYMMDD-HHMM-run_name/
checkpoints/YYYYMMDD-HHMM-run_name/
```

Typical checkpoint files include:

```text
last.ckpt
refine-<epoch>-<metric>.ckpt
```

depending on `trainer.checkpointing.filename`.

---

## Validation Metrics and Plots

During training, both model families log:

```text
train_loss
val_loss
val_recon_psnr
val_recon_ssim
val_recon_l1
val_recon_phase_coherence
val_recon_phase_error
```

The diffusion model additionally logs:

```text
val_recon_rmse
val_recon_amp_corr
```

and, when EMA is enabled:

```text
val_recon_psnr_ema
val_recon_ssim_ema
val_recon_l1_ema
val_recon_rmse_ema
val_recon_amp_corr_ema
val_recon_phase_coherence_ema
val_recon_phase_error_ema
```

`WandBPlottingCallback` logs:

- `val/reconstruction`: input bandwidth panels, predicted target, optional EMA
  prediction, and target.
- `val/magnitude_kde_db`: KDE overlay of predicted and target magnitude in dB.
- `val/reconstruction_histograms`: real, imaginary, and magnitude histograms in
  physical scale.
- `val/phase_error_histogram`: wrapped phase error histogram.

The model computes reconstruction metrics after inverse normalization so phase
and magnitude metrics operate in physical SAR scale where applicable.

---

## Overfit Sanity Check

Use overfit mode to verify that the model, normalization, channel mapping, and
loss can memorize a tiny controlled subset.

Set:

```yaml
data:
  overfit:
    enabled: true
    source_split: train
    num_patches: 8
```

Behavior:

- The data module builds one subset from `source_split`.
- The exact same subset is used for train and validation.
- Data augmentation is disabled in this mode.

Expected result:

- `train_loss` should drop quickly.
- Validation reconstruction metrics should improve because validation sees the
  same samples.

If the model cannot overfit this setting, inspect:

- bandwidth index mapping
- I/Q channel order
- `model.in_channels`, `model.out_channels`, `model.unet.channels`,
  `model.unet.out_dim`
- `data.global_max`
- loss scaling and phase loss weighting
- learning rate and gradient clipping
- model capacity

---

## Offline Evaluation

`evaluate.py` reads its checkpoint path and test selection from the
`evaluation` section of the config.

Example:

```yaml
evaluation:
  checkpoint_path: "/shared/home/lvanderpeet/AE5822-Thesis/checkpoints/Complex-090-0.1085.ckpt"
  save_dir: "evaluation_results"
  tests:
    point_target_analysis: false
    kde_distance: true
    phase_coherence: true
    enl_comparison: true
```

Run:

```bash
python3 evaluate.py --config configs/config.yaml
```

The script writes:

```text
evaluation_results/metrics.json
```

and may also write plots such as:

```text
evaluation_results/kde_log_magnitude.png
evaluation_results/irf_profiles_sample_000.png
```

depending on the enabled tests.

### Evaluation tests

#### `point_target_analysis`

For each sample, the script:

- Finds the peak in the target magnitude image.
- Crops a local window around the peak.
- Upsamples the local window.
- Computes azimuth and range impulse response profiles.

Reported IRF metrics:

- `resolution_px`: 3 dB width in pixels.
- `pslr_db`: peak side-lobe ratio.
- `islr_db`: integrated side-lobe ratio.

The output includes both per-sample profiles and aggregate summaries.

#### `kde_distance`

Compares predicted and target log-magnitude distributions.

Reported metrics:

- `wasserstein`: Wasserstein distance between flattened log-magnitude samples.
- `kl_divergence`: KL divergence between KDE-estimated densities.

The script also saves a KDE plot.

#### `phase_coherence`

Computes phase agreement between prediction and target.

Reported metrics include:

- `mean_abs_phase_diff`: mean absolute wrapped phase difference.
- `std_phase_diff_rad`: standard deviation of wrapped phase difference.
- `coherence_mean`: mean local interferometric coherence.
- `coherence_std`: standard deviation of local interferometric coherence.

#### `enl_comparison`

Computes equivalent number of looks from intensity:

```text
ENL = mean(intensity)^2 / var(intensity)
```

Reported metrics:

- target ENL mean/std
- predicted ENL mean/std
- absolute error mean/std
- relative error mean/std
- per-sample values

### Hypothesis testing section

`evaluate.py` also writes a `hypothesis_testing` section when the corresponding
tests are enabled.

Implemented checks:

- `H1_Azimuth_Resolution`: predicted/target azimuth resolution ratio within
  `+/- 5%` at 95% confidence.
- `H2_Range_Resolution`: predicted/target range resolution ratio within
  `+/- 5%` at 95% confidence.
- `H3_ENL_Margin`: predicted/target ENL ratio within `+/- 5%` at 95%
  confidence.
- `H4_Phase_Consistency`: upper 95% confidence bound of phase standard
  deviation in degrees below the implemented threshold.

---

## Common Experiment Recipes

### Train `75% -> 100%` with VV only

```yaml
data:
  subaperture_config:
    input_indices: [2]
    output_indices: [0]
    active_polarizations: ["VV"]

model:
  in_channels: 2
  out_channels: 2
  unet:
    channels: 4
    out_dim: 2
```

### Train `50% -> 100%` with VV only

```yaml
data:
  subaperture_config:
    input_indices: [4]
    output_indices: [0]
    active_polarizations: ["VV"]

model:
  in_channels: 2
  out_channels: 2
  unet:
    channels: 4
    out_dim: 2
```

### Train multi-input `75%, 62.5%, 50% -> 100%`

```yaml
data:
  subaperture_config:
    input_indices: [2, 3, 4]
    output_indices: [0]
    active_polarizations: ["VV"]

model:
  in_channels: 6
  out_channels: 2
  unet:
    channels: 8
    out_dim: 2
```

### Train dual-polarization `75% -> 100%`

```yaml
data:
  subaperture_config:
    input_indices: [2]
    output_indices: [0]
    active_polarizations: ["VV", "VH"]

model:
  in_channels: 4
  out_channels: 4
  unet:
    channels: 8
    out_dim: 4
```

### Switch to DC2SCN baseline

```yaml
model:
  type: dc2scn
  in_channels: 2
  out_channels: 2
  dc2scn:
    width: 64
    num_blocks: 6
    growth_rate: 32
```

`unet` settings are ignored when `model.type: dc2scn`.

---

## Practical Checklist Before Long Runs

1. Confirm `data.root_dir` exists and contains product directories with
   `.safetensors` files.
2. Confirm each generated patch contains a tensor named `x` with 20 channels.
3. Confirm the bandwidth index mapping:

```text
0=100%, 1=87.5%, 2=75%, 3=62.5%, 4=50%
```

4. Confirm `active_polarizations` matches the generated channels and current
   dataset support.
5. Recalculate `model.in_channels` and `model.out_channels`.
6. For diffusion, set `model.unet.channels = in_channels + out_channels`.
7. Set `model.unet.out_dim = out_channels`.
8. Confirm `data.global_max` is suitable for the dataset.
9. Run a small overfit sanity check.
10. Inspect W&B reconstruction panels and phase/magnitude histograms.
11. Confirm checkpoint monitor and mode match the metric objective.

---

    active_polarizations: ["VV"]

model:
  in_channels: 6
  out_channels: 2
  unet:
    channels: 8
    out_dim: 2
```

### Train dual-polarization `75% -> 100%`

```yaml
data:
  subaperture_config:
    input_indices: [2]
    output_indices: [0]
    active_polarizations: ["VV", "VH"]

model:
  in_channels: 4
  out_channels: 4
  unet:
    channels: 8
    out_dim: 4
```

### Switch to DC2SCN baseline

```yaml
model:
  type: dc2scn
  in_channels: 2
  out_channels: 2
  dc2scn:
    width: 64
    num_blocks: 6
    growth_rate: 32
```

`unet` settings are ignored when `model.type: dc2scn`.

---

## Practical Checklist Before Long Runs

1. Confirm `data.root_dir` exists and contains product directories with
   `.safetensors` files.
2. Confirm each generated patch contains a tensor named `x` with 20 channels.
3. Confirm the bandwidth index mapping:

```text
0=100%, 1=87.5%, 2=75%, 3=62.5%, 4=50%
```

4. Confirm `active_polarizations` matches the generated channels and current
   dataset support.
5. Recalculate `model.in_channels` and `model.out_channels`.
6. For diffusion, set `model.unet.channels = in_channels + out_channels`.
7. Set `model.unet.out_dim = out_channels`.
8. Confirm `data.global_max` is suitable for the dataset.
9. Run a small overfit sanity check.
10. Inspect W&B reconstruction panels and phase/magnitude histograms.
11. Confirm checkpoint monitor and mode match the metric objective.

---

## Troubleshooting

### Shape mismatch in the model

Most shape errors come from inconsistent channel counts.

Check:

```yaml
data.subaperture_config.input_indices
data.subaperture_config.output_indices
data.subaperture_config.active_polarizations
model.in_channels
model.out_channels
model.unet.channels
model.unet.out_dim
```

For diffusion:

```text
model.unet.channels = model.in_channels + model.out_channels
model.unet.out_dim  = model.out_channels
```

### Reconstruction labels still say `FA` or `SA`

This is expected until the historical labels are renamed in the code. Interpret
them through the bandwidth mapping table:

```text
FA  -> index 0 -> 100%
SA1 -> index 1 -> 87.5%
SA2 -> index 2 -> 75%
SA3 -> index 3 -> 62.5%
SA4 -> index 4 -> 50%
```

### Validation is slow

Diffusion validation can be expensive because prediction runs a full reverse
denoising chain. To shorten test runs:

```bash
python3 train.py --config configs/config.yaml --limit-val-batches 0.1
```

or lower `trainer.limit_val_batches` in the config.

### W&B is not desired

Use:

```bash
export WANDB_DISABLED=true
```

or:

```bash
export WANDB_MODE=offline
```

### Generated values look saturated

Check `data.global_max`. Saturation usually means the value is too small for
the dataset. Inspect the physical magnitude distribution and choose a scale that
compresses outliers without forcing most high-intensity values to `1.0`.

### Phase metrics are poor even when magnitude improves

Check:

- I/Q order in the generated safetensors.
- Whether both input and target use the same polarization order.
- Whether the inverse normalization uses the same `global_max` as training.
- Hybrid loss phase weight.
- Whether the task is too underdetermined for the selected input bandwidth.