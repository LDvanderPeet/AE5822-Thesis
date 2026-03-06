from __future__ import annotations

import torch
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    """Hardwired random paired dataset returning (x, y)."""

    LENGTH = 1000
    CHANNELS = 2
    IMAGE_SIZE = 128

    def __len__(self) -> int:
        return self.LENGTH

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        del idx
        x = torch.rand(self.CHANNELS, self.IMAGE_SIZE, self.IMAGE_SIZE)
        y = torch.rand(self.CHANNELS, self.IMAGE_SIZE, self.IMAGE_SIZE)
        return x, y


if __name__ == "__main__":
    # 1. Instantiate the dataset
    dataset = PairedImageDataset()

    # 2. Try to get the first item
    try:
        x, y = dataset[0]
        print("Success!")
        print(f"Input shape (x): {x.shape}")  # Should be [2, 128, 128]
        print(f"Target shape (y): {y.shape}")  # Should be [2, 128, 128]
        print(f"Device: {x.device}")
    except Exception as e:
        print(f"Failed: {e}")