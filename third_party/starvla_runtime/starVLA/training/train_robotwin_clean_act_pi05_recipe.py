"""Train ACT on RoboTwin with a pi0.5-style optimizer recipe and EMA checkpoints."""

import argparse
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training import train_starvla as base_train
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


class EMAVLATrainer(base_train.VLATrainer):
    def prepare_training(self):
        super().prepare_training()
        self._init_ema()

    def _init_ema(self):
        self.ema_decay = float(getattr(self.config.trainer, "ema_decay", 0.0) or 0.0)
        self.ema_state = None
        if self.ema_decay <= 0.0:
            return

        ema_device = str(getattr(self.config.trainer, "ema_device", "cuda")).lower()
        module = self.accelerator.unwrap_model(self.model)
        self.ema_state = {}
        with torch.no_grad():
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                value = param.detach().float().clone()
                if ema_device == "cpu":
                    value = value.cpu()
                self.ema_state[name] = value

        if self.accelerator.is_main_process:
            base_train.logger.info(
                f"EMA enabled: decay={self.ema_decay}, device={ema_device}, tensors={len(self.ema_state)}"
            )

    def _update_ema(self):
        if not self.ema_state:
            return

        decay = self.ema_decay
        module = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            for name, param in module.named_parameters():
                shadow = self.ema_state.get(name)
                if shadow is None:
                    continue
                current = param.detach().float()
                if shadow.device.type == "cpu":
                    current = current.cpu()
                shadow.mul_(decay).add_(current, alpha=1.0 - decay)

    def _train_step(self, batch_vla, batch_vlm=None):
        metrics = super()._train_step(batch_vla, batch_vlm=batch_vlm)
        if self.accelerator.sync_gradients:
            self._update_ema()
        return metrics

    def _ema_state_dict(self):
        if not self.ema_state:
            return None

        state_dict = self.accelerator.get_state_dict(self.model)
        for name, shadow in self.ema_state.items():
            if name in state_dict:
                state_dict[name] = shadow.detach().to(device="cpu", dtype=state_dict[name].dtype)
        return state_dict

    def _save_ema_checkpoint(self, checkpoint_stem: str):
        if not self.ema_state:
            return

        state_dict = self._ema_state_dict()
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_stem + "_ema_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_stem + "_ema_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`.")
            self.accelerator.print(f"EMA checkpoint saved at {checkpoint_stem}")
        del state_dict
        self.accelerator.wait_for_everyone()

    def _save_checkpoint(self):
        super()._save_checkpoint()
        checkpoint_stem = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        self._save_ema_checkpoint(checkpoint_stem)

    def _finalize_training(self):
        super()._finalize_training()
        final_stem = os.path.join(self.config.output_dir, "final_model", "ema")
        self._save_ema_checkpoint(final_stem)


def main():
    base_train.VLATrainer = EMAVLATrainer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="experiments/robotwin/configs/clean50.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    accelerator = base_train.build_accelerator(cfg)
    base_train.main(cfg, accelerator)


if __name__ == "__main__":
    main()
