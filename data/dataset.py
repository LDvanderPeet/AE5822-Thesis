from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from .finaldataset import SafetensorSARDataset


class _RandomPairedImageDataset(Dataset):
    """Fallback random paired dataset returning (x, y)."""

    LENGTH = 100
    IMAGE_SIZE = 128

    def __init__(self, in_channels: int = 2, out_channels: int = 2) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __len__(self) -> int:
        return self.LENGTH

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        del idx
        x = torch.rand(self.in_channels, self.IMAGE_SIZE, self.IMAGE_SIZE) # change to self.CHANNELS
        y = torch.rand(self.out_channels, self.IMAGE_SIZE, self.IMAGE_SIZE) # change to self.CHANNELS
        return x, y

class PairedImageDataset(Dataset):
    """Adapter dataset returning exactly `(x, y)` for Lightning.

    If SAR config keys are present this wraps `ComplexSARDataset`; otherwise it falls back to a random placeholder
    dataset.
    """
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        split: str = "train",
        da: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.config = config or {}
        self.split = split
        self.da = da

        data_cfg = self.config.get("data", {})
        root_dir = data_cfg.get("root_dir") if isinstance(data_cfg, dict) else None

        if root_dir and os.path.exists(root_dir):
            print(f"--- Loading Safetensor Backend for {split} Split ---")
            self._backend = SafetensorSARDataset(cfg=self.config, split=split, da=da)
        else:
            print("--- Root directory not found. Falling back to Random Data ---")
            model_cfg = self.config.get("model", {})
            in_channels = int(model_cfg.get("in_channels", 12))  # Match your SAR count
            out_channels = int(model_cfg.get("out_channels", 4))
            self._backend = _RandomPairedImageDataset(in_channels=in_channels, out_channels=out_channels)


    def __len__(self) -> int:
        return len(self._backend)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # SafetensorSARDataset returns (x, y, meta)
        # Lightning usually only wants (x, y)
        sample = self._backend[idx]

        # Ensure we only return the first two elements (x, y) to the model
        return sample[0], sample[1]


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
