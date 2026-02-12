import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from diff_utils import visualize_reconstruction, setup_run_directory, log_to_csv, generate_final_plots
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from dataset import build_datasets_from_config
from models import UNet

def train():
    config_path = "/shared/home/lvanderpeet/AE5822-Thesis/config.yaml"

    run_dir, viz_dir = setup_run_directory(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds = build_datasets_from_config(config_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["train"]["num_workers"]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["train"]["num_workers"]
    )

    in_ch = (len(cfg["data"]["subaperture_config"]["input_indices"]) *
             cfg["data"]["subaperture_config"]["polarizations"] *
             (2 if cfg["data"]["subaperture_config"]["is_complex"] else 1))
    out_ch = (len(cfg["data"]["subaperture_config"]["output_indices"]) *
              cfg["data"]["subaperture_config"]["polarizations"] *
              (2 if cfg["data"]["subaperture_config"]["is_complex"] else 1))

    model = UNet(
        in_channels=in_ch,
        out_channels= out_ch,
        base_channels=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=float(cfg["training"]["scheduler"]["factor"]),
    #     patience=int(cfg["training"]["scheduler"]["patience"]),
    #     min_lr=float(cfg["training"]["scheduler"]["min_lr"])
    # )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["training"]["epochs"]),
        eta_min=float(cfg["training"]["scheduler"]["min_lr"]),
    )

    es_counter = 0
    best_val_loss = float('inf')
    epochs = cfg["training"]["epochs"]
    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            loop = tqdm(train_loader, desc=f"Epoch[{epoch+1}/{epochs}]")
            for i, (x, y, _) in enumerate(loop):
                x, y = x.to(device), y.to(device)

                ## ---- Forward Pass ---- ##
                outputs = model(x)
                loss = criterion(outputs, y)

                ## ---- Backward Pass ---- ##
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            model.eval()
            val_loss = 0.0
            val_loop = tqdm(val_loader, desc="Validating")
            with torch.no_grad():
                for i, (x, y, _) in enumerate(val_loop):
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # scheduler.step(avg_val_loss)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
                "es_counter": es_counter,
            }

            log_to_csv(run_dir, metrics)

            print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, LR: {current_lr}")

            if avg_val_loss < (best_val_loss - float(cfg["training"]["early_stopping"]["min_delta"])):
                best_val_loss = avg_val_loss
                es_counter = 0

                best_path = os.path.join(run_dir, "unet_best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"--> New best model saved with Val Loss: {avg_val_loss:.4f}")
            else:
                es_counter += 1
                print(f"Early Stopping counter: {es_counter} out of {cfg["training"]["early_stopping"]["patience"]}")

            visualize_reconstruction(model, val_loader, device, epoch + 1, viz_dir=viz_dir)

            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)

            if es_counter >= cfg["training"]["early_stopping"]["patience"]:
                print(f"Early stopping triggered! No improvement for {cfg['training']['early_stopping']['patience']} epochs")
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving current progress...")

    generate_final_plots(run_dir)


if __name__ == "__main__":
    train()