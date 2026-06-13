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

# SAR Bandwidth Super-Resolution Framework

This repository contains a PyTorch Lightning framework for complex-valued SAR
bandwidth super-resolution. The current task is no longer a full-aperture /
subaperture reconstruction problem. It is now formulated as:

```text
condition x: reduced-bandwidth complex SAR patch
target y:    higher-bandwidth complex SAR patch, usually the 100% reference
```

The model learns to recover the 100% bandwidth image from lower-bandwidth
versions produced by azimuth/Doppler low-pass filtering. Complex SLC data is
represented with interleaved real and imaginary channels, so both amplitude and
phase are available to the model and to the evaluation metrics.

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

---

