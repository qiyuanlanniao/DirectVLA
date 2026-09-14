from pathlib import Path

import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

def build_dataloader(cfg, dataset_py="lerobot_datasets"):
    if dataset_py != "lerobot_datasets":
        raise ValueError(
            "This open-source subset only supports the RoboTwin LeRobot loader; "
            f"got dataset_py={dataset_py!r}."
        )

    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    num_workers = int(vla_dataset_cfg.get("num_workers", 8))
    pin_memory = _as_bool(vla_dataset_cfg.get("pin_memory", True))
    persistent_workers = _as_bool(vla_dataset_cfg.get("persistent_workers", True))
    prefetch_factor = int(vla_dataset_cfg.get("prefetch_factor", 4))

    dataloader_kwargs = {
        "batch_size": vla_dataset_cfg.per_device_batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    if dist.get_rank() == 0:
        logger.info(
            "VLA DataLoader config: "
            f"num_workers={num_workers}, pin_memory={pin_memory}, "
            f"persistent_workers={persistent_workers if num_workers > 0 else False}, "
            f"prefetch_factor={prefetch_factor if num_workers > 0 else None}"
        )

    vla_train_dataloader = DataLoader(vla_dataset, **dataloader_kwargs)
    if dist.get_rank() == 0:
        output_dir = Path(cfg.output_dir)
        vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
    return vla_train_dataloader
