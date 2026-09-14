"""
Centralized registry that auto-discovers benchmark-specific data configs from
``experiments/*/data_registry/`` and merges them with the base
registries defined in this package.

Three registries are maintained:

* ``DATASET_NAMED_MIXTURES``       – mixture_name → [(dataset, weight, robot_type)]
* ``ROBOT_TYPE_CONFIG_MAP``       – robot_type → DataConfig instance
* ``ROBOT_TYPE_TO_EMBODIMENT_TAG`` – robot_type → EmbodimentTag


Usage::

    from starVLA.dataloader.gr00t_lerobot.registry import (
        ROBOT_TYPE_CONFIG_MAP,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
        DATASET_NAMED_MIXTURES,
    )
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

# Base registries (kept as fallback / seed values)
from starVLA.dataloader.gr00t_lerobot.data_config import (
    ROBOT_TYPE_CONFIG_MAP as _BASE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import (
    ROBOT_TYPE_TO_EMBODIMENT_TAG as _BASE_EMBODIMENT_MAP,
    EmbodimentTag,  # noqa: F401  – re-export for convenience
)
from starVLA.dataloader.gr00t_lerobot.mixtures import (
    DATASET_NAMED_MIXTURES as _BASE_MIXTURES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mutable copies – will be extended by discovered modules
# ---------------------------------------------------------------------------
ROBOT_TYPE_CONFIG_MAP: dict = dict(_BASE_CONFIG_MAP)
ROBOT_TYPE_TO_EMBODIMENT_TAG: dict = dict(_BASE_EMBODIMENT_MAP)
DATASET_NAMED_MIXTURES: dict = dict(_BASE_MIXTURES)

# ---------------------------------------------------------------------------
# Discovery logic
# ---------------------------------------------------------------------------
_REGISTRY_DIR_NAME = "data_registry"
_DISCOVERED = False


def _find_registry_dirs() -> list[Path]:
    """Return all ``experiments/*/data_registry/`` directories."""
    repo_root = next(
        (
            candidate
            for candidate in Path(__file__).resolve().parents
            if (candidate / "pyproject.toml").is_file()
            and (candidate / "experiments").is_dir()
        ),
        None,
    )
    if repo_root is None:
        return []
    experiments_dir = repo_root / "experiments"
    dirs: list[Path] = []
    for bench_dir in sorted(experiments_dir.iterdir()):
        registry_dir = bench_dir / _REGISTRY_DIR_NAME
        if registry_dir.is_dir():
            dirs.append(registry_dir)
    return dirs


def _load_module_from_path(module_name: str, file_path: Path):
    """Import a Python file as a module with the given name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_and_merge() -> None:
    """Scan ``experiments/*/data_registry/`` and merge into global registries."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    for registry_dir in _find_registry_dirs():
        bench_name = registry_dir.parent.name
        prefix = f"_data_registry_{bench_name}"

        # --- data_config.py (may contain all three registries) ---
        cfg_file = registry_dir / "data_config.py"
        if cfg_file.is_file():
            mod = _load_module_from_path(f"{prefix}.data_config", cfg_file)
            if mod:
                if hasattr(mod, "ROBOT_TYPE_CONFIG_MAP"):
                    ROBOT_TYPE_CONFIG_MAP.update(mod.ROBOT_TYPE_CONFIG_MAP)
                    logger.info(f"[registry] Loaded data_config from {bench_name}: {list(mod.ROBOT_TYPE_CONFIG_MAP.keys())}")
                if hasattr(mod, "ROBOT_TYPE_TO_EMBODIMENT_TAG"):
                    ROBOT_TYPE_TO_EMBODIMENT_TAG.update(mod.ROBOT_TYPE_TO_EMBODIMENT_TAG)
                    logger.info(f"[registry] Loaded embodiment_tags from {bench_name} (data_config): {list(mod.ROBOT_TYPE_TO_EMBODIMENT_TAG.keys())}")
                if hasattr(mod, "DATASET_NAMED_MIXTURES"):
                    DATASET_NAMED_MIXTURES.update(mod.DATASET_NAMED_MIXTURES)
                    logger.info(f"[registry] Loaded mixtures from {bench_name} (data_config): {list(mod.DATASET_NAMED_MIXTURES.keys())}")


# Run discovery on first import
discover_and_merge()
