from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data import PairedDataModule
from src.PixelDiffusion import PixelDiffusionConditional


def load_config(config_path: str) -> dict:
    """Load the YAML config file used to build data, model, and trainer."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def main() -> None:
    """Entry point for training with PyTorch Lightning.

    Lightning flow in this script:
    1. Build data module (encapsulates DataLoaders).
    2. Build LightningModule (`PixelDiffusionConditional`).
    3. Build `pl.Trainer` with runtime options.
    4. Call `trainer.fit(model, datamodule=datamodule)` to start the full train/val loop.
    """
    parser = argparse.ArgumentParser()
    # Path to the YAML config used for all runtime settings.
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    # Optional override for Lightning's validation batch fraction/count.
    parser.add_argument("--limit-val-batches", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    # Ensures deterministic random behavior where possible.
    pl.seed_everything(config.get("seed", 42), workers=True)
    # Controls float32 matmul precision tradeoff: "medium" (faster) or "high" (more accurate).
    torch.set_float32_matmul_precision(
        config.get("trainer", {}).get("float32_matmul_precision", "high")
    )

    # DataModule centralizes loader construction and setup for Lightning.
    datamodule = PairedDataModule.from_config(config)
    # Split config sections for clarity.
    model_cfg = config.get("model", {})
    opt_cfg = config.get("optimization", {})
    lr_sched_cfg = opt_cfg.get("reduce_lr_on_plateau", {})
    unet_cfg = model_cfg.get("unet", {})
    wandb_cfg = config.get("logging", {}).get("wandb", {})

    # This is the Lightning model used for training and validation.
    model = PixelDiffusionConditional(
        condition_channels=model_cfg.get("in_channels", 2),
        generated_channels=model_cfg.get("out_channels", 2),
        num_timesteps=model_cfg.get("num_timesteps", 1000),
        schedule=model_cfg.get("schedule", "linear"),
        noise_offset=model_cfg.get("noise_offset", 0),
        loss_fn=model_cfg.get("loss_fn", 'mse'),
        model_dim=unet_cfg.get("dim", 64),
        model_dim_mults=tuple(unet_cfg.get("dim_mults", [1, 2, 4, 8])),
        model_channels=unet_cfg.get("channels"),
        model_out_dim=unet_cfg.get("out_dim"),
        lr=opt_cfg.get("lr", 1e-3),
        lr_scheduler_factor=lr_sched_cfg.get("factor", 0.5),
        lr_scheduler_patience=lr_sched_cfg.get("patience", 10),
    )

    trainer_cfg = config.get("trainer", {})
    # CLI override has priority over config file value.
    limit_val_batches = (
        args.limit_val_batches
        if args.limit_val_batches is not None
        else trainer_cfg.get("limit_val_batches", 1.0)
    )
    # Lightning logger wrapper for Weights & Biases.
    wandb_logger = WandbLogger(
        project=wandb_cfg.get("project", "dif_img_rec"),
        name=wandb_cfg.get("name"),
        save_dir=wandb_cfg.get("save_dir", "logs"),
        log_model=wandb_cfg.get("log_model", False),
    )
    # Trainer controls loop behavior, device placement, precision, and logging cadence.
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 1),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", 32),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        enable_checkpointing=trainer_cfg.get("enable_checkpointing", False),
        limit_val_batches=limit_val_batches,
        logger=wandb_logger,
        callbacks=[lr_monitor],
    )

    # Starts the training/validation loop.
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
