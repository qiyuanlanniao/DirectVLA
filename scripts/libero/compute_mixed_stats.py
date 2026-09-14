#!/usr/bin/env python3
"""Compute LIBERO state/action statistics across multiple TFDS suites."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", action="append", required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--stats_key", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def _init_accum(dim: int) -> dict[str, np.ndarray | int]:
    return {
        "count": 0,
        "total": np.zeros(dim, dtype=np.float64),
        "total_sq": np.zeros(dim, dtype=np.float64),
        "min": np.full(dim, np.inf, dtype=np.float64),
        "max": np.full(dim, -np.inf, dtype=np.float64),
    }


def _update_accum(accum: dict[str, np.ndarray | int] | None, arr: np.ndarray) -> dict[str, np.ndarray | int]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if accum is None:
        accum = _init_accum(arr.shape[1])
    if arr.shape[1] != np.asarray(accum["total"]).shape[0]:
        raise ValueError(f"dimension changed from {np.asarray(accum['total']).shape[0]} to {arr.shape[1]}")
    accum["count"] = int(accum["count"]) + int(arr.shape[0])
    accum["total"] = np.asarray(accum["total"]) + arr.sum(axis=0)
    accum["total_sq"] = np.asarray(accum["total_sq"]) + np.square(arr).sum(axis=0)
    accum["min"] = np.minimum(np.asarray(accum["min"]), arr.min(axis=0))
    accum["max"] = np.maximum(np.asarray(accum["max"]), arr.max(axis=0))
    return accum


def _finalize(accum: dict[str, np.ndarray | int]) -> dict[str, list[float]]:
    count = int(accum["count"])
    if count <= 0:
        raise RuntimeError("cannot finalize empty accumulator")
    total = np.asarray(accum["total"], dtype=np.float64)
    total_sq = np.asarray(accum["total_sq"], dtype=np.float64)
    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-6)
    return {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "min": np.asarray(accum["min"], dtype=np.float64).astype(float).tolist(),
        "max": np.asarray(accum["max"], dtype=np.float64).astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output} (use --overwrite to replace it)")

    state_accum = None
    action_accum = None
    suite_summaries = []
    total_episodes = 0

    for dataset_dir in args.dataset_dir:
        builder = tfds.builder_from_directory(builder_dir=dataset_dir)
        dataset = builder.as_dataset(split=args.split)
        episode_lengths: list[int] = []

        for episode in tfds.as_numpy(dataset):
            states = []
            actions = []
            for step in episode["steps"]:
                states.append(np.asarray(step["observation"]["state"], dtype=np.float64).reshape(-1))
                actions.append(np.asarray(step["action"], dtype=np.float64).reshape(-1))
            if not actions:
                continue
            state_arr = np.stack(states, axis=0)
            action_arr = np.stack(actions, axis=0)
            state_accum = _update_accum(state_accum, state_arr)
            action_accum = _update_accum(action_accum, action_arr)
            episode_lengths.append(int(action_arr.shape[0]))
            total_episodes += 1
            if args.log_every > 0 and total_episodes % args.log_every == 0:
                print(f"processed total_episodes={total_episodes} actions={int(action_accum['count'])}", flush=True)

        if not episode_lengths:
            raise RuntimeError(f"no episodes found in dataset={dataset_dir} split={args.split}")
        suite_summaries.append(
            {
                "dataset_dir": os.path.abspath(dataset_dir),
                "num_episodes": int(len(episode_lengths)),
                "num_actions": int(sum(episode_lengths)),
                "episode_length_min": int(min(episode_lengths)),
                "episode_length_max": int(max(episode_lengths)),
                "episode_length_mean": float(np.mean(episode_lengths)),
            }
        )

    if state_accum is None or action_accum is None:
        raise RuntimeError("no samples found")

    payload = {
        args.stats_key: {
            "state": _finalize(state_accum),
            "proprio": _finalize(state_accum),
            "action": _finalize(action_accum),
        },
        "metadata": {
            "dataset_dirs": [os.path.abspath(path) for path in args.dataset_dir],
            "split": args.split,
            "stats_key": args.stats_key,
            "num_actions": int(action_accum["count"]),
            "num_states": int(state_accum["count"]),
            "num_episodes": int(total_episodes),
            "suite_summaries": suite_summaries,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, output)
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
