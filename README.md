# AE5822-Thesis

# Complex SAR Subaperture Reconstruction Framework

A PyTorch-based deep learning pipeline designed for Single Look Complex (SLC) SAR data processing. This framework facilitates the reconstruction of target subapertures from a subset of input subapertures using either standard regression or Denoising Diffusion Probabilistic Models (DDPM).

## Features
* **Complex Component Preservation**: Explicitly handles SLC data by interleaving Real ($Re$) and Imaginary ($Im$) components as separate channels to maintain phase information.
* **Dual-Mode Architecture**: Toggle between standard **U-Net Regression** and **Diffusion (DDPM)** via configuration.
* **SAR-Specific Augmentation**: `tfm_dihedral` transformations are restricted to "strip-safe" operations (flips and 180° rotations) to preserve the 1024x240 aspect ratio.
* **Advanced Loss Functions**: Includes a Magnitude-Weighted Loss that balances pixel-wise complex error with physical magnitude consistency.

---

## System Architecture

The core model is a flexible U-Net that adapts its input dimensionality based on the task.



### Complex Data Representation
To satisfy the requirement of preserving both real and imaginary components without using standard complex-valued layers, data is transformed as follows:
* **Interleaving**: A complex array of shape $(N, H, W)$ is converted into a real-valued tensor of shape $(2N, H, W)$.
* **Channel Order**: $[Re_1, Im_1, Re_2, Im_2, \dots, Re_N, Im_N]$.
* **Reconstruction**: The model predicts $2N$ channels, which are then used to calculate physical magnitude: $\text{Mag} = \sqrt{Re^2 + Im^2}$.

---

## Project Structure

* **`dataset.py`**: Handles Zarr-backed SLC data loading, normalization, and "strip-safe" cropping/padding for SAR geometries.
* **`models.py`**: Contains the `UNet` and `TimeEmbedding` classes. Supports dynamic depth and time-injection for diffusion conditioning.
* **`diff_utils.py`**: The `DiffusionEngine` manages noise schedules ($\beta_t$), the forward diffusion process, and reverse sampling steps. Includes magnitude visualization tools.
* **`train.py`**: The main entry point. Orchestrates training loops, validation with early stopping, and automatic experiment logging.

---

## Getting Started

### 1. Configuration
Experiments are controlled via `config.yaml`. Key parameters include:
* `model.diffusion`: Boolean to switch between Regression and DDPM.
* `data.subaperture_config.is_complex`: Ensures 2x channel multiplier for SLC data.
* `training.loss_function`: Choose between `mse`, `mae`, or `complex_magnitude`.

### 2. Execution
Run the training pipeline:
```bash
python train.py