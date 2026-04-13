from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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
    parser.add_argument("--resume-from-ckpt", type=str, default=None)
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
    ema_cfg = model_cfg.get("ema", {})
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    raw_run_name = wandb_cfg.get("name") or "run"
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_run_name).strip("._-") or "run"
    timestamp = datetime.now().strftime("%Y%d%m-%H%M")
    run_folder_name = f"{timestamp}-{run_name}"
    wandb_base_dir = Path(wandb_cfg.get("save_dir", "logs")).expanduser()
    wandb_local_dir = wandb_base_dir / run_folder_name
    wandb_local_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config.get("data", {})
    sa_cfg = data_cfg.get("subaperture_config", {}) if isinstance(data_cfg, dict) else {}
    input_indices = sa_cfg.get("input_indices", [])
    input_condition_labels = [f"SA{int(idx)}" if int(idx) > 0 else "FA" for idx in input_indices]

    # This is the Lightning model used for training and validation.
    model = PixelDiffusionConditional(
        condition_channels=model_cfg.get("in_channels", 2),
        generated_channels=model_cfg.get("out_channels", 2),
        input_condition_labels=input_condition_labels if input_condition_labels else None,
        target_label="FA",
        num_timesteps=model_cfg.get("num_timesteps", 1000),
        schedule=model_cfg.get("schedule", "linear"),
        noise_offset=model_cfg.get("noise_offset", 0),
        loss_fn=model_cfg.get("loss_fn", 'mse'),
        hybrid_base_loss=model_cfg.get("hybrid_base_loss", "mse"),
        hybrid_ms_ssim_t_limt=model_cfg.get("hybrid_ms_ssim_t_limit", 10),
        model_dim=unet_cfg.get("dim", 64),
        model_dim_mults=tuple(unet_cfg.get("dim_mults", [1, 2, 4, 8])),
        model_channels=unet_cfg.get("channels"),
        model_out_dim=unet_cfg.get("out_dim"),
        lr=opt_cfg.get("lr", 1e-3),
        lr_scheduler_factor=lr_sched_cfg.get("factor", 0.5),
        lr_scheduler_patience=lr_sched_cfg.get("patience", 10),
        ema_enabled=ema_cfg.get("enabled", False),
        ema_beta=ema_cfg.get("beta", 0.9999),
        ema_update_every=ema_cfg.get("update_every", 1),
        ema_update_after_step=ema_cfg.get("update_after_step", 0),
        phase_hist_max_batches=model_cfg.get("phase_hist_max_batches", 8),
        data_global_max=config.get("data", {}).get("global_max", 4257.0),
        wandb_save_config_file=wandb_cfg.get("save_config_file", True),
        config_path=args.config,
        wandb_config_artifact_name=wandb_cfg.get("config_artifact_name"),
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
        save_dir=str(wandb_local_dir),
        log_model=wandb_cfg.get("log_model", False),
    )
    # Trainer controls loop behavior, device placement, precision, and logging cadence.
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cfg = trainer_cfg.get("checkpointing", {})
    checkpoint_base_dir = Path(checkpoint_cfg.get("dirpath", "checkpoints")).expanduser()
    checkpoint_dir = checkpoint_base_dir / run_folder_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=checkpoint_cfg.get("filename", "{epoch:03d}-{val_loss:.4f}"),
        monitor=checkpoint_cfg.get("monitor", "val_loss"),
        mode=checkpoint_cfg.get("mode", "min"),
        save_top_k=checkpoint_cfg.get("save_top_k", 3),
        save_last=checkpoint_cfg.get("save_last", True),
        every_n_epochs=checkpoint_cfg.get("every_n_epochs", 1),
        auto_insert_metric_name=checkpoint_cfg.get("auto_insert_metric_name", False),
    )
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 1),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", 32),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        enable_checkpointing=trainer_cfg.get("enable_checkpointing", False),
        limit_val_batches=limit_val_batches,
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=trainer_cfg.get("gradient_clip_algorithm", "norm"),
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
    )

    # Starts the training/validation loop.
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_ckpt)


if __name__ == "__main__":
    main()
