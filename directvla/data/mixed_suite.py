"""DINOv3-only LIBERO RLDS dataset mixed across multiple TFDS suites."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import get_worker_info

from .libero_rlds import (
    LiberoRLDSDataset as _BaseLiberoRLDSDataset,
    vla_collate_fn,
)
from .suite_stats import (
    _select_stats,
    _vector,
)


try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


def _split_dataset_dirs(dataset_dirs: str | os.PathLike[str] | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(dataset_dirs, (list, tuple)):
        parts = [str(item).strip() for item in dataset_dirs]
    else:
        raw = str(dataset_dirs)
        normalized = raw.replace("\n", ",").replace(";", ",")
        parts = [item.strip() for item in normalized.split(",")]
    parts = [item for item in parts if item]
    if not parts:
        raise ValueError("dataset_dirs must contain at least one TFDS builder directory")
    return parts


class LiberoMixedRLDSDataset(_BaseLiberoRLDSDataset):
    def __init__(
        self,
        dataset_dir,
        *args,
        dataset_dirs: str | os.PathLike[str] | list[str] | tuple[str, ...],
        stats_path: str | os.PathLike[str],
        stats_key: str | None = None,
        normalize_binary_gripper: str = "auto",
        **kwargs,
    ):
        kwargs.pop("dataset_dir", None)
        self.dataset_dirs = _split_dataset_dirs(dataset_dirs)
        super().__init__(self.dataset_dirs[0], *args, **kwargs)

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
                "mixed_suite_stats="
                f"dirs={self.dataset_dirs}, path={self.stats_path}, key={self.stats_key}, "
                f"state_dim={self.proprio_mean.numel()}, action_dim={self.action_min.numel()}, "
                f"normalize_binary_gripper={self._normalize_binary_gripper}",
                flush=True,
            )

    def _normalize_action_chunk(self, action_chunk):
        action_chunk = super()._normalize_action_chunk(action_chunk)
        if self._normalize_binary_gripper:
            action_chunk[:, 6] = action_chunk[:, 6] * 2.0 - 1.0
        return action_chunk

    def _iter_mixed_episodes(self, epoch_seed, rank, world_size, worker_info):
        builders = [tfds.builder_from_directory(builder_dir=path) for path in self.dataset_dirs]
        dataset = None
        for builder in builders:
            current = builder.as_dataset(split=self.split)
            if world_size > 1:
                current = current.shard(num_shards=world_size, index=rank)
            if worker_info is not None:
                current = current.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            dataset = current if dataset is None else dataset.concatenate(current)
        dataset = dataset.shuffle(
            buffer_size=self.shuffle_buffer,
            seed=epoch_seed,
            reshuffle_each_iteration=False,
        )
        for episode in tfds.as_numpy(dataset):
            yield episode

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        base_seed = self.seed + 1009 * self.rank + 9176 * worker_id
        epoch = 0
        while True:
            epoch_seed = base_seed + epoch
            rng = np.random.default_rng(epoch_seed)
            step_buffer = []

            for episode in self._iter_mixed_episodes(epoch_seed, self.rank, self.world_size, worker_info):
                steps = list(episode["steps"])
                episode_len = len(steps)
                step_indices = list(range(episode_len))

                if self.shuffle_steps_within_episode:
                    rng.shuffle(step_indices)

                if self.step_mix_buffer_size > 0:
                    for t in step_indices:
                        step_buffer.append(self._build_step_sample(steps, t, episode_len))
                        if len(step_buffer) >= self.step_mix_buffer_size:
                            out_idx = int(rng.integers(0, len(step_buffer)))
                            yield step_buffer.pop(out_idx)
                else:
                    for t in step_indices:
                        yield self._build_step_sample(steps, t, episode_len)

            while len(step_buffer) > 0:
                out_idx = int(rng.integers(0, len(step_buffer)))
                yield step_buffer.pop(out_idx)

            epoch += 1


__all__ = ["LiberoMixedRLDSDataset", "vla_collate_fn"]
