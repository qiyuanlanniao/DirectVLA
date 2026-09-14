#!/usr/bin/env python3
"""Mixed-suite ViT-B online-BERT full-unfreeze trainer with pi0.5 knobs."""

from __future__ import annotations

import argparse
import sys

from . import pi05, trainer
from ..data.mixed_suite import (
    LiberoMixedRLDSDataset,
)


def parse_args_with_mixed_suite_stats():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dinov3_precision",
        type=str,
        default="bf16_autocast",
        choices=["fp32", "bf16", "bf16_autocast"],
    )
    parser.add_argument("--dataset_dirs", type=str, required=True)
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--stats_key", type=str, required=True)
    parser.add_argument(
        "--normalize_binary_gripper",
        type=str,
        default="auto",
        choices=["auto", "true", "false", "1", "0", "yes", "no"],
    )
    known, remaining = parser.parse_known_args()

    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        args = pi05._ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    args.dinov3_precision = known.dinov3_precision
    args.dataset_dirs = known.dataset_dirs
    args.stats_path = known.stats_path
    args.stats_key = known.stats_key
    args.normalize_binary_gripper = known.normalize_binary_gripper
    return args


class MixedSuiteStatsLiberoRLDSDataset(LiberoMixedRLDSDataset):
    def __init__(self, *args, **kwargs):
        active_args = trainer._ACTIVE_TRAIN_ARGS
        super().__init__(
            *args,
            dataset_dirs=active_args.dataset_dirs,
            stats_path=active_args.stats_path,
            stats_key=active_args.stats_key,
            normalize_binary_gripper=active_args.normalize_binary_gripper,
            **kwargs,
        )


trainer._ACTIVE_TRAIN_ARGS = None


def parse_args_and_record():
    args = parse_args_with_mixed_suite_stats()
    trainer._ACTIVE_TRAIN_ARGS = args
    return args


trainer.parse_args = parse_args_and_record
trainer.LiberoRLDSDataset = MixedSuiteStatsLiberoRLDSDataset


def main():
    trainer.train_model()


if __name__ == "__main__":
    main()
