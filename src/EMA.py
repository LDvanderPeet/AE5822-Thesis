from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class EMA:
    """EMA helper inspired by DiffusionFastForward's callback approach.

    This version keeps an explicit EMA model for easy side-by-side inference
    (regular vs EMA) during the same validation step.
    """

    def __init__(self, decay: float, apply_every_n_steps: int = 1, start_step: int = 0) -> None:
        if not (0.0 <= decay <= 1.0):
            raise ValueError("EMA decay must be within [0, 1].")
        self.decay = float(decay)
        self.apply_every_n_steps = int(apply_every_n_steps)
        self.start_step = int(start_step)
        self._last_applied_step = -1

    def should_apply(self, step: int) -> bool:
        return (
            step != self._last_applied_step
            and step >= self.start_step
            and step % self.apply_every_n_steps == 0
        )

    @torch.no_grad()
    def update_model_average(self, ema_model: nn.Module, current_model: nn.Module) -> None:
        ema_state = ema_model.state_dict()
        current_state = current_model.state_dict()

        for name, ema_value in ema_state.items():
            current_value = current_state.get(name)
            if current_value is None or ema_value.shape != current_value.shape:
                # Keep behavior similar to DiffusionFastForward:
                # if shapes mismatch, skip that tensor.
                continue
            ema_value.mul_(self.decay).add_(current_value, alpha=1.0 - self.decay)

    @torch.no_grad()
    def update(self, ema_model: nn.Module, current_model: nn.Module, step: int) -> None:
        if self.should_apply(step):
            self._last_applied_step = step
            self.update_model_average(ema_model, current_model)


class EMAWrapper(nn.Module):
    """Owns an EMA shadow model and updates it with the configured schedule."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        apply_every_n_steps: int = 1,
        start_step: int = 0,
    ) -> None:
        super().__init__()
        self.ema = EMA(
            decay=decay,
            apply_every_n_steps=apply_every_n_steps,
            start_step=start_step,
        )
        self.ema_model = deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        # Before scheduled EMA starts, keep exact copy to avoid stale weights.
        if step < self.ema.start_step:
            self.ema_model.load_state_dict(model.state_dict())
            return
        self.ema.update(self.ema_model, model, step)
