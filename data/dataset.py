from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class _RandomPairedImageDataset(Dataset):
    """Fallback random paired dataset returning (x, y)."""

    LENGTH = 1000
    # CHANNELS = 2
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
        self._is_sar_backend = False

        data_cfg = self.config.get("data", {})
        sar_cfg = data_cfg.get("subaperture_config", {}) if isinstance(data_cfg, dict) else {}
        root_dir = data_cfg.get("root_dir") if isinstance(data_cfg, dict) else None

        has_required_sar_fields = (
            isinstance(sar_cfg, dict)
            and bool(root_dir)
            and isinstance(sar_cfg.get("input_indices"), list)
            and isinstance(sar_cfg.get("output_indices"), list)
            and isinstance(sar_cfg.get("active_polarizations"), list)
        )

        if has_required_sar_fields:
            from .sardataset import ComplexSARDataset
            self._backend: Dataset = ComplexSARDataset(cfg=self.config, split=split, da=da)
            self._is_sar_backend = True
        else:
            model_cfg = self.config.get("model", {}) if isinstance(self.config, dict) else {}
            in_channels = int(model_cfg.get("in_channels", 2))
            out_channels = int(model_cfg.get("out_channels", 2))
            self._backend = _RandomPairedImageDataset(in_channels=in_channels, out_channels=out_channels)

    def __len__(self) -> int:
        return len(self._backend)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._backend[idx]
        if not (isinstance(sample, tuple) and len(sample) >= 2):
            raise ValueError("Dataset backend must return at least (x, y).")

        x, y = sample[0], sample[1]

        # SAR backend currently emits values in [-1, 1].
        # The Lightning model expects data in [0, 1] before applying `input_T`.
        if self._is_sar_backend:
            x = x.add(1.0).div(2.0).clamp(0.0, 1.0)
            y = y.add(1.0).div(2.0).clamp(0.0, 1.0)

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
