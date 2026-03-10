# Runbook: Local Dev Setup

## Prerequisites

### macOS
```bash
# Install Conda (if not already installed)
brew install --cask miniconda
# OR download from https://docs.conda.io/en/latest/miniconda.html

# Install Git
brew install git
```

### Create environment
```bash
conda create -n nanogpt python=3.10 -y
conda activate nanogpt

# Install PyTorch with CUDA (adjust cuda version as needed)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU-only (Mac / no GPU):
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install numpy transformers tiktoken wandb tqdm
```

## Verifying Your Install
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Should print torch version and True if CUDA is available
```

## Running Locally (Quick Test — Shakespeare)

```bash
cd /Users/nancheng/Projects/nanoGPT

# 1. Prepare the dataset
python data/shakespeare_char/prepare.py

# 2. Train a tiny model (runs on CPU or MPS in ~minutes)
python train.py config/train_shakespeare_char.py \
  --device=cpu --compile=False --eval_iters=20 \
  --log_interval=1 --block_size=64 --batch_size=12 \
  --n_layer=4 --n_head=4 --n_embd=128 \
  --max_iters=2000 --lr_decay_iters=2000

# 3. Sample from the trained model
python sample.py --out_dir=out-shakespeare-char \
  --device=cpu --compile=False
```

## Useful Commands

```bash
# Train with full config file
python train.py config/train_shakespeare_char.py

# Override individual params
python train.py config/train_gpt2.py --batch_size=4 --compile=False

# Resume from checkpoint
python train.py --init_from=resume --out_dir=out

# Sample from GPT-2 pretrained (no training required)
python sample.py --init_from=gpt2 --device=cpu

# Benchmark throughput
python bench.py
```

## Connecting to GPU (Remote)

Set `device=cuda` (default) and ensure CUDA toolkit matches your PyTorch version. For multi-GPU:

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py
```
