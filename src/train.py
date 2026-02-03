import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets_from_config
from models import UNet

def train():
    config_path = "/shared/home/lvanderpeet/AE5822-Thesis/config.yaml"
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

    in_ch = len(cfg["data"]["subaperture_config"]["input_indices"]) * 2
    out_ch = len(cfg["data"]["subaperture_config"]["output_indices"]) * 2

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

    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch[{epoch+1}/{epochs}]")
        for x, y, _ in loop:
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
        with torch.nograd():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"unet_checkpoint_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()