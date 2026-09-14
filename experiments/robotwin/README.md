# RoboTwin 2.0 clean50

This directory contains the released RoboTwin 2.0 clean50 recipe and public entry points. Both benchmarks use the shared model in `../../turbovla/models/`. The code under `third_party/starvla_runtime/` only adapts RoboTwin batches, training checkpoints, and the policy server to that model.

The released model uses three DINOv3 camera views, online BERT instruction encoding, bidirectional vision-language interaction, a 50-step ACT head, and 14-D bimanual actions.

## Layout

```text
configs/                         clean50, modality, and DeepSpeed configuration
data_registry/                   50-task mixture and embodiment contract
evaluation/                      policy client and simulator adapter
train.py                         public Python training entry point
evaluate.py                      public Python evaluation entry point
../../scripts/robotwin/          data, training, serving, and evaluation scripts
../../third_party/               StarVLA-compatible runtime and licenses
```

## Installation

Python 3.10 or newer is required.

```bash
pip install -e ".[robotwin]"
```

Install a CUDA-compatible PyTorch build and FlashAttention 2 separately when needed by the selected environment.

## Required assets

```bash
export ROBOTWIN_DATA_ROOT=/path/to/converted/RoboTwin
export BERT_MODEL_PATH=/path/to/bert-base-uncased
export TURBOVLA_INIT_CKPT=/path/to/groundingdino_swint_ogc.pth
export DINOV3_MODEL_PATH=/path/to/dinov3
```

`ROBOTWIN_DATA_ROOT` must contain the 50 datasets under `Clean/<task_name>`.

## Training

The default recipe uses four GPUs, global batch 192, 100k optimizer steps, learning rate `5e-5`, 1k warmup steps, and EMA decay `0.999`.

```bash
export CUDA_VISIBLE_DEVICES=<gpu_ids>
bash scripts/robotwin/train.sh
```

Override `NUM_PROCESSES`, `PER_DEVICE_BATCH_SIZE`, `MAX_TRAIN_STEPS`, `LEARNING_RATE`, `RUN_ROOT_DIR`, or `RUN_ID` as needed.

## Evaluation

Install RoboTwin separately and set:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
export STARVLA_PYTHON=/path/to/policy-env/bin/python
export ROBOTWIN_PYTHON=/path/to/robotwin-env/bin/python
```

Evaluate all clean50 tasks:

```bash
export CUDA_VISIBLE_DEVICES=<gpu_ids>
bash scripts/robotwin/evaluate.sh \
  pretrained/TurboVLA/checkpoints/robotwin/steps_55000_ema_model.safetensors
```

Append task names to evaluate a subset. The released result is 60.2% on clean50 using one shared step-55k EMA checkpoint and 100 trials per task.

The compatibility runtime retains the `starVLA` and `deployment` package names used by released checkpoints. See `third_party/licenses/StarVLA-MIT.txt`, `third_party/licenses/Apache-2.0.txt`, and the source-file copyright notices.
