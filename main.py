# import yaml
import torch
import os
import numpy as np

print(torch.__version__)
if torch.cuda.is_available():
    # is_available() takes no arguments
    print(f"CUDA is generally available: {torch.cuda.is_available()}")

    # To check how many GPUs you have
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs found: {device_count}")

    # To get the name of your A6000 (index 0)
    print(f"GPU 0 Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available")

# def main(config_path="config.yaml"):
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#
#     # train_loader, val_loader = get_dataloaders(config)
#     #
#     # model = build_sar_model(config)
#
#
#
# if __name__ == "__main__":
#     print("Hello, World!")