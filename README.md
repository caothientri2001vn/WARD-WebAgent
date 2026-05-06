# WARD

## Features

- Base webpage sample generation with visual overlays
- Guard-targeted adversarial dataset construction
- Reasoning generation for malicious and benign samples
- SFT dataset preparation for multimodal guard training
- Guard inference utilities for Hugging Face checkpoints

## Repository Structure

```text
.
├── generate_guard_dataset.py
├── requirements-core.txt
├── source/
│   ├── overlay_attack.py
│   ├── randomness.py
│   └── sample_overlay_image.py
├── malicious_goals/
│   ├── malicious_goals.json
│   └── malicious_goals_use.json
├── reasoning/
│   ├── first_ground.py
│   ├── generate_reasoning.py
│   └── second_ground.py
├── adversarial_attack_data/
│   ├── build_guard_attack_dataset.py
│   ├── build_guard_attack_dataset_malicious_reasoned.py
│   └── guard_attack_utils.py
├── adversarial_attack/
│   ├── attack_guard_ref.py
│   ├── infer_guard_groundtruth_spam.py
│   └── infer_guard_groundtruth_visual.py
└── train_guard/llamafactory_guard/
    ├── infer_guard_hf.py
    ├── merge_guard_train_sft_with_generated_only.py
    ├── prepare_guard_sft_dataset.py
    ├── prepare_guard_train_SFT_ground2.py
    └── render_llamafactory_config.py
```

## Installation

Create a Conda environment:

```bash
conda create -n WARD python=3.11 -y
conda activate WARD
python -m pip install --upgrade pip
```

Install PyTorch separately using the command that matches the local CUDA runtime from the official PyTorch installation guide.

Install the remaining dependencies:

```bash
pip install -r requirements-core.txt
playwright install chromium
```

## Main Workflow

### 1. Generate guarded webpage samples

`generate_guard_dataset.py` builds rendered guard examples from webpage inputs and overlay templates.

```bash
python generate_guard_dataset.py --help
```

Supporting rendering utilities are located in `source/`.

### 2. Build guard-targeted attack datasets

The `adversarial_attack_data/` directory contains scripts for transforming source samples into attack-oriented datasets.

```bash
python adversarial_attack_data/build_guard_attack_dataset.py --help
python adversarial_attack_data/build_guard_attack_dataset_malicious_reasoned.py --help
```

Attack-goal definitions are stored in `malicious_goals/`.

### 3. Generate reasoning labels

The `reasoning/` directory contains scripts for generating or refining reasoning traces used in guard training.

```bash
python reasoning/first_ground.py --help
python reasoning/second_ground.py --help
python reasoning/generate_reasoning.py --help
```

### 4. Prepare SFT training data

The `train_guard/llamafactory_guard/` directory contains the dataset preparation utilities used for multimodal SFT training.

```bash
python train_guard/llamafactory_guard/prepare_guard_sft_dataset.py --help
python train_guard/llamafactory_guard/prepare_guard_train_SFT_ground2.py --help
python train_guard/llamafactory_guard/merge_guard_train_sft_with_generated_only.py --help
```

`render_llamafactory_config.py` is included as a minimal config-generation helper for training setup.

### 5. Run guard inference

The main Hugging Face inference entrypoint is:

```bash
python train_guard/llamafactory_guard/infer_guard_hf.py --help
```

The `adversarial_attack/` directory contains additional probing scripts:

```bash
python adversarial_attack/infer_guard_groundtruth_spam.py --help
python adversarial_attack/infer_guard_groundtruth_visual.py --help
python adversarial_attack/attack_guard_ref.py --help
```

## Smoke Checks

The following commands are useful for verifying that the environment and entrypoints are wired correctly:

```bash
python generate_guard_dataset.py --help
python adversarial_attack_data/build_guard_attack_dataset.py --help
python reasoning/generate_reasoning.py --help
python train_guard/llamafactory_guard/prepare_guard_sft_dataset.py --help
python train_guard/llamafactory_guard/infer_guard_hf.py --help
```
