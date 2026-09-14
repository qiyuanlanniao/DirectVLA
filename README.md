# DirectVLA: Development of a High-Frequency, Low-Memory, Real-Time VLA Policy Based on a Minimalist Direct-Mapping Architecture

## 📄 Abstract

To address the bottlenecks of traditional VLAs—which rely on Large Language Models (LLMs) with billions of parameters, resulting in high VRAM usage (>10 GB) and high latency (~11 Hz)—we developed DirectVLA, a real-time policy model featuring an ultra-simple direct mapping architecture. Moving away from the standard V→L→A pipeline, the project constructs a V+L→A model based on DINOv3 and BERT, integrating bidirectional vision-language interaction with non-autoregressive continuous action block decoding. Running on a single RTX 4090, the model achieves ultra-low latency of ~30 ms with only 0.2 billion parameters and 0.9 GB of VRAM. Its high robustness in closed-loop control has been validated across LIBERO single-arm (97.7%), RoboTwin dual-arm (60.2%), and xMate SR3 real-world robot (up to 92.5%) benchmarks.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/qiyuanlanniao/DirectVLA.git
cd DirectVLA
```

LIBERO and RoboTwin use different simulator and data stacks. We recommend separate Python 3.10 environments.

### LIBERO Environment

```bash
conda create -n direct-libero python=3.10 -y
conda activate direct-libero

# Install the CUDA-compatible PyTorch build for your system first.
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[libero]"
```

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) separately in the same environment.

### RoboTwin Environment

```bash
conda create -n direct-robotwin python=3.10 -y
conda activate direct-robotwin

# Install a CUDA-compatible PyTorch build before the project dependencies.
pip install -e ".[robotwin]"
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Install the [RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin) simulator in a separate evaluation environment when required by your setup.

---

## 📦 Dataset and Model Preparation

Model weights and benchmark datasets are external assets and are not committed to this repository.

### Required Models

| Asset | Source | Used by |
| --- | --- | --- |
| DINOv3 ViT-B | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | LIBERO |
| DINOv3 ViT-L | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | RoboTwin |
| BERT base uncased | [google-research/bert](https://github.com/google-research/bert) | Both |
| GroundingDINO Swin-T OGC | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Both |

### LIBERO Data

direct expects the four modified no-noops suites in TFDS/RLDS format:

```text
data/libero/
|-- libero_10_no_noops/1.0.0/
|-- libero_goal_no_noops/1.0.0/
|-- libero_object_no_noops/1.0.0/
`-- libero_spatial_no_noops/1.0.0/
```

The repository provides no-op removal and mixed-suite statistics utilities:

```bash
python scripts/libero/regenerate_libero_no_noops.py --help
python scripts/libero/compute_mixed_stats.py --help
```

Released normalization statistics are stored in `experiments/libero/configs/libero_all4_stats.json`. BERT is part of the model and runs online during both training and evaluation; no text-feature cache is required. See [experiments/libero/README.md](experiments/libero/README.md) for details.

### RoboTwin Data

Download the clean LeRobot dataset and create the expected local link:

```bash
bash scripts/robotwin/prepare_data.sh /path/to/storage
export ROBOTWIN_DATA_ROOT="$PWD/playground/Datasets/RoboTwin"
```

The default downloader uses [StarVLA/RoboTwin-Clean](https://huggingface.co/datasets/StarVLA/RoboTwin-Clean). The training registry expects all 50 datasets under `Clean/<task_name>`.

---

## 🏋️ Training and Evaluation

### LIBERO Training

The paper recipe uses DINOv3 ViT-B, two camera views, 7-D actions, a 12-step action chunk, 80k optimizer steps, 10k warmup steps, and global batch size 256 on four GPUs.

```bash
torchrun --nproc_per_node=4 experiments/libero/train.py \
  --dataset_dir data/libero/libero_10_no_noops/1.0.0 \
  --dataset_dirs "data/libero/libero_10_no_noops/1.0.0,data/libero/libero_goal_no_noops/1.0.0,data/libero/libero_object_no_noops/1.0.0,data/libero/libero_spatial_no_noops/1.0.0" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --dinov3_path facebook/dinov3-vitb16-pretrain-lvd1689m \
  --bert_path google-bert/bert-base-uncased \
  --allow_hf_download \
  --pretrained_init_ckpt /path/to/groundingdino_swint_ogc.pth \
  --checkpoint_dir outputs/libero
```

### LIBERO Evaluation

One command evaluates one checkpoint on one suite.

```bash
python experiments/libero/evaluate.py \
  --ckpt_path pretrained/direct/checkpoints/libero/libero_object.pth \
  --dinov3_path /path/to/dinov3-vitb \
  --bert_path /path/to/bert-base-uncased \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --task_suite_name libero_object \
  --num_trials_per_task 50 \
  --chunk_size 12 \
  --num_open_loop_steps 12 \
  --seed 7 \
  --precision bf16 \
  --result_json_path outputs/evaluation/libero_object.json
```

Valid suite names are `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`.

### RoboTwin Training

The paper recipe uses DINOv3 ViT-L, three camera views, 14-D absolute joint-position actions, a 50-step ACT head, global batch size 192, and 55k optimizer steps on four GPUs.

```bash
export ROBOTWIN_DATA_ROOT="$PWD/playground/Datasets/RoboTwin"
export BERT_MODEL_PATH=/path/to/bert-base-uncased
export direct_INIT_CKPT=/path/to/groundingdino_swint_ogc.pth
export DINOV3_MODEL_PATH=/path/to/dinov3-vitl
export CUDA_VISIBLE_DEVICES=0,1,2,3

MAX_TRAIN_STEPS=55000 \
RUN_ID=direct_robotwin_clean50_55k \
bash scripts/robotwin/train.sh
```

### RoboTwin Evaluation

The policy server and RoboTwin simulator can run in separate Python environments:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
export STARVLA_PYTHON=/path/to/policy-env/bin/python
export ROBOTWIN_PYTHON=/path/to/robotwin-env/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3

ROBOTWIN_TEST_NUM=100 \
bash scripts/robotwin/evaluate.sh \
  pretrained/direct/checkpoints/robotwin/steps_55000_ema_model.safetensors
```

Append task names to the evaluation command to run a subset. Omitting them evaluates all 50 clean tasks.

---

## 👍 Acknowledgement

direct builds upon the following projects and resources:

- [DINOv3](https://github.com/facebookresearch/dinov3) for visual representations.
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for bidirectional vision-language interaction components and initialization.
- [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) for the LIBERO task and episode rollout protocol.
- [StarVLA](https://github.com/StarVLA/StarVLA) for the RoboTwin-compatible training and evaluation runtime.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin) for simulation benchmarks.

---