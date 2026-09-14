"""DINOv3-only LIBERO RLDS dataset with suite-specific state/action stats."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .libero_rlds import (
    LiberoRLDSDataset as _BaseLiberoRLDSDataset,
    vla_collate_fn,
)


def _select_stats(payload: dict, stats_key: str | None) -> dict:
    if stats_key:
        if stats_key not in payload:
            raise KeyError(f"stats_key={stats_key!r} not found in stats payload")
        selected = payload[stats_key]
    else:
        keys = [key for key in payload.keys() if key != "metadata"]
        if len(keys) != 1:
            raise KeyError(f"stats_key is required; available keys: {keys}")
        selected = payload[keys[0]]
    if not isinstance(selected, dict):
        raise ValueError("selected stats entry must be a dict")
    return selected


def _vector(stats: dict, section: str, name: str) -> torch.Tensor:
    if section not in stats:
        raise KeyError(f"stats payload missing section {section!r}")
    value = stats[section].get(name)
    if value is None:
        raise KeyError(f"stats payload missing {section}.{name}")
    return torch.tensor(value, dtype=torch.float32)


class LiberoRLDSDataset(_BaseLiberoRLDSDataset):
    def __init__(
        self,
        *args,
        stats_path: str | os.PathLike[str],
        stats_key: str | None = None,
        normalize_binary_gripper: str = "auto",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stats_path = str(stats_path)
        self.stats_key = stats_key
        self.normalize_binary_gripper = normalize_binary_gripper

        payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        stats = _select_stats(payload, stats_key)
        state_section = "proprio" if "proprio" in stats else "state"

        self.proprio_mean = _vector(stats, state_section, "mean")
        self.proprio_std = _vector(stats, state_section, "std")
        self.action_min = _vector(stats, "action", "min")
        self.action_max = _vector(stats, "action", "max")

        gripper_min = float(self.action_min[6].item())
        gripper_max = float(self.action_max[6].item())
        if normalize_binary_gripper == "auto":
            self._normalize_binary_gripper = gripper_min >= -1e-6 and gripper_max <= 1.0 + 1e-6
        elif normalize_binary_gripper in {"1", "true", "yes"}:
            self._normalize_binary_gripper = True
        elif normalize_binary_gripper in {"0", "false", "no"}:
            self._normalize_binary_gripper = False
        else:
            raise ValueError("normalize_binary_gripper must be auto, true, or false")

        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "suite_stats="
                f"path={self.stats_path}, key={self.stats_key}, "
                f"state_dim={self.proprio_mean.numel()}, action_dim={self.action_min.numel()}, "
                f"normalize_binary_gripper={self._normalize_binary_gripper}",
                flush=True,
            )

    def _normalize_action_chunk(self, action_chunk):
        action_chunk = super()._normalize_action_chunk(action_chunk)
        if self._normalize_binary_gripper:
            action_chunk[:, 6] = action_chunk[:, 6] * 2.0 - 1.0
        return action_chunk


__all__ = ["LiberoRLDSDataset", "vla_collate_fn"]
