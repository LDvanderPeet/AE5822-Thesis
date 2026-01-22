import yaml
import torch
import os


def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # train_loader, val_loader = get_dataloaders(config)
    #
    # model = build_sar_model(config)



if __name__ == "__main__":
    print("Hello, World!")