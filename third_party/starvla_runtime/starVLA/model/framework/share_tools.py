"""Configuration and checkpoint helpers for the StarVLA runtime adapter."""

import dataclasses
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def dict_to_namespace(value):
    """Return an OmegaConf config with attribute and ``get`` access."""
    return OmegaConf.create(value)


def merge_framework_config(default_config_cls, cfg):
    """Merge dataclass framework defaults with the user/checkpoint config."""
    defaults = OmegaConf.create(dataclasses.asdict(default_config_cls()))

    if hasattr(cfg, "framework"):
        incoming = cfg.framework
        if hasattr(incoming, "_cfg"):
            incoming = incoming._cfg
        if not isinstance(incoming, DictConfig):
            incoming = OmegaConf.create(incoming if isinstance(incoming, dict) else {})
    else:
        incoming = OmegaConf.create({})

    merged = OmegaConf.merge(defaults, incoming)

    if hasattr(cfg, "_cfg") and isinstance(cfg._cfg, DictConfig):
        cfg._cfg.framework = merged
        if hasattr(cfg, "_children") and "framework" in cfg._children:
            accessed = cfg._children["framework"]._local_accessed.copy()
            del cfg._children["framework"]
            cfg.framework._local_accessed.update(accessed)
    elif isinstance(cfg, DictConfig):
        cfg.framework = merged
    else:
        cfg.framework = merged
    return cfg


def apply_config_compat(cfg, *, strict: bool = False):
    """Normalize the only legacy action-window alias still needed by checkpoints."""
    del strict
    if cfg is None:
        return cfg

    action_path = "framework.action"
    action_model = OmegaConf.select(cfg, action_path, default=None)
    if action_model is None:
        action_path = "framework.action_model"
        action_model = OmegaConf.select(cfg, action_path, default=None)
    if action_model is None:
        return cfg

    horizon = OmegaConf.select(action_model, "horizon", default=None)
    if horizon is None:
        horizon = OmegaConf.select(action_model, "action_horizon", default=None)
    future_window = OmegaConf.select(action_model, "future_action_window_size", default=None)
    if horizon is None and future_window is not None:
        OmegaConf.update(
            cfg,
            f"{action_path}.horizon",
            int(future_window) + 1,
            force_add=True,
        )
    return cfg


def read_mode_config(pretrained_checkpoint):
    """Load checkpoint metadata from the nearest containing parent directory."""
    checkpoint_path = Path(pretrained_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.suffix not in {".pt", ".safetensors"}:
        raise ValueError(f"Unsupported checkpoint suffix: {checkpoint_path.suffix}")

    run_dir = next(
        (
            parent
            for parent in checkpoint_path.parents
            if (parent / "config.yaml").is_file() and (parent / "dataset_statistics.json").is_file()
        ),
        None,
    )
    if run_dir is None:
        searched = ", ".join(str(parent) for parent in checkpoint_path.parents)
        raise FileNotFoundError(
            "Could not find checkpoint metadata files `config.yaml` and "
            f"`dataset_statistics.json` in any parent directory of {checkpoint_path}. "
            f"Searched: {searched}"
        )

    config_path = run_dir / "config.yaml"
    statistics_path = run_dir / "dataset_statistics.json"
    logger.info(f"Loading checkpoint metadata from `{run_dir}`")
    config = OmegaConf.load(config_path)
    apply_config_compat(config)
    with statistics_path.open("r", encoding="utf-8") as handle:
        statistics = json.load(handle)
    return OmegaConf.to_container(config, resolve=True), statistics
