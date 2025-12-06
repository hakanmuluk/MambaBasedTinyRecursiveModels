# 🧩 Mamba-Based Tiny Recursive Models — Sudoku Pretraining Guide

This README is a **complete, copy-and-use cheat sheet** for running the Mamba-based Tiny Recursive Model (TRM) on RunPod using the image:

```
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
```

Project path (persistent volume):

```
/workspace/MambaBasedTinyRecursiveModels
```

Training target:

- Architecture: TRM (Tiny Recursive Model)
- Task: Sudoku Extreme (~1k puzzles × 1000 augmentations)
- Global batch size: `256`

This document includes:

1. 🧱 One-time environment setup  
2. 🧩 Fixing & building `mamba-ssm` from source  
3. 🧮 Dataset creation  
4. 🚀 Training with `tmux`  
5. 🔁 Reconnecting to the session  
6. 🧷 Ultra-short quick reference  

---

# 🧱 1. One-Time Environment Setup

Perform this once per **new RunPod instance** or **new virtual environment**.

### Clone project & create venv

```bash
cd /workspace

git clone https://github.com/hakanmuluk/MambaBasedTinyRecursiveModels.git
cd MambaBasedTinyRecursiveModels

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
```

### Install PyTorch 2.7.0 (CUDA 12.6 wheels)

```bash
pip install torch==2.7.0+cu126 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

```

### Install project dependencies

```bash
pip install -r specific_requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2

```

### (Optional) Login to W&B once

```bash
wandb login
```

Your environment is now ready — except for the Mamba CUDA extension, which must be built from source.

---

# 🧩 2. Fixing `mamba-ssm` (Build from Source)

Run this **once per venv**.

### Uninstall any prebuilt wheels

```bash
cd /workspace/MambaBasedTinyRecursiveModels
source .venv/bin/activate

pip uninstall -y mamba-ssm selective-scan selective_scan_cuda
```

### Clone the official Mamba repo & build

```bash
cd /workspace
git clone https://github.com/state-spaces/mamba.git mamba_ssm_src
cd mamba_ssm_src

pip install --upgrade pip wheel setuptools
pip install ninja cmake

pip install --no-build-isolation .
```

### Verify

```bash
python -c "from mamba_ssm import Mamba; print('Mamba OK')"
```

If it prints **Mamba OK** → you're ready.

---

# 🧮 3. Build the Sudoku Dataset

```bash
cd /workspace/MambaBasedTinyRecursiveModels
source .venv/bin/activate

mkdir -p data

python dataset/build_sudoku_dataset.py \
--output-dir data/sudoku-extreme-1k-aug-1000 \
--subsample-size 1000 \
--test-subsample-size 20000 \
--num-aug 1000 \
--seed 42
```

Check output:

```bash
ls data/sudoku-extreme-1k-aug-1000
# train/  test/  identifiers.json
```

Dataset is ready.

---

# 🚀 4. Start Training (with tmux)

### Enter project & activate venv

```bash
cd /workspace/MambaBasedTinyRecursiveModels
source .venv/bin/activate
```

### Start a persistent tmux session

```bash
tmux new -s trm_mamba
```

### Run pretraining

Inside tmux:

```bash
export run_name="pretrain_mamba_sudoku_smallbs"

python pretrain.py   arch=trm   data_paths="[data/sudoku-extreme-1k-aug-1000]"   evaluators="[]"   epochs=50000 eval_interval=5000   lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0   arch.L_layers=2   arch.H_cycles=3 arch.L_cycles=6   global_batch_size=256   +run_name=${run_name}   ema=True
```

### Detach tmux

```
Ctrl + B, then D
```

---

# 🔁 5. Reconnect Later

```bash
cd /workspace/MambaBasedTinyRecursiveModels
source .venv/bin/activate

tmux ls
tmux attach -t trm_mamba
```

---

# 🧷 6. Ultra-Short Quick Reference

```bash
# Setup
cd /workspace
git clone https://github.com/hakanmuluk/MambaBasedTinyRecursiveModels.git
cd MambaBasedTinyRecursiveModels
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r specific_requirements.txt

cd /workspace
git clone https://github.com/state-spaces/mamba.git mamba_ssm_src
cd mamba_ssm_src
pip install ninja cmake
pip install --no-build-isolation .

cd /workspace/MambaBasedTinyRecursiveModels
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Training
cd /workspace/MambaBasedTinyRecursiveModels
source .venv/bin/activate
tmux new -s trm_mamba
export run_name="pretrain_mamba_sudoku_smallbs"
python pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" evaluators="[]" epochs=50000 eval_interval=5000   lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0   arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 global_batch_size=256 +run_name=${run_name} ema=True
```

