# nanoGPT — Project Intelligence

> Root memory file. Claude reads this first every session.
> Last updated: 2026-03-10

---

## What This Project Does

nanoGPT is a minimal, clean PyTorch reimplementation of GPT-2 for training and fine-tuning language models. It accepts tokenised text datasets (binary `.bin` files), trains a decoder-only Transformer, and supports both single-GPU and multi-GPU (DDP/`torchrun`) runs. The entire model fits in one file (`model.py`); training fits in another (`train.py`).

---

## Navigation — Where to Go for What

| I want to understand… | Read this |
|-----------------------|-----------|
| Full architecture & data flows | [`docs/architecture.md`](docs/architecture.md) |
| Why key decisions were made | [`docs/decisions/`](docs/decisions/) |
| Ops procedures | [`docs/runbooks/`](docs/runbooks/) |
| Model internals (GPT, Block, Attention, MLP) | [`model.py`](model.py) |
| Training loop, DDP, checkpointing | [`train.py`](train.py) |
| Sampling / inference | [`sample.py`](sample.py) |
| Config override system | [`configurator.py`](configurator.py) |
| Pre-made training configs | [`config/`](config/) |
| Skills / task recipes | [`.claude/skills/`](.claude/skills/) |
| Agent settings | [`.claude/settings.json`](.claude/settings.json) |

---

## Project Identity

| Property | Value |
|----------|-------|
| Name | `nanoGPT` |
| Language / Runtime | Python 3.10+ |
| Framework | Raw PyTorch (no Trainer wrapper) |
| Key Libraries | `torch`, `numpy`, `tiktoken`, `transformers` (weight loading only) |
| Model Family | GPT-2 (decoder-only Transformer) |
| Repo | https://github.com/karpathy/nanoGPT |

---

## Architecture (3-Sentence Summary)

nanoGPT is a decoder-only Transformer language model that takes a batch of token sequences and is trained to predict the next token at every position using cross-entropy loss. Data flows as: raw text → tokeniser → `.bin` file → `get_batch()` memmap loader → `GPT.forward()` → softmax logits → cross-entropy loss → AdamW with cosine LR decay. All business logic (architecture, weight init, MFU estimation, generation) lives in `model.py`; all training orchestration (DDP, checkpointing, gradient accumulation, evaluation) lives in `train.py`.

---

## Key Source Files

| File | Role |
|------|------|
| `model.py` | Full GPT model: `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, `GPTConfig`, `GPT` |
| `train.py` | Training loop: DDP setup, data loading, optimiser, LR schedule, checkpointing, W&B logging |
| `sample.py` | Inference: loads checkpoint or pretrained GPT-2, runs autoregressive generation |
| `configurator.py` | Config override: reads `config/*.py` files and `--key=value` CLI args into globals |
| `config/` | Pre-made configs: `train_gpt2.py`, `train_shakespeare_char.py`, `finetune_shakespeare.py`, `eval_gpt2*.py` |
| `data/` | Dataset preparation scripts (e.g. `data/shakespeare_char/prepare.py`) |
| `bench.py` | Benchmarking: measures forward/backward throughput and MFU |

---

## Source Layout

```
nanoGPT/
├── model.py           # Model definition only — no I/O, no training logic
├── train.py           # Training entrypoint — wires everything together
├── sample.py          # Inference entrypoint
├── configurator.py    # Config override utility (exec'd by train.py / sample.py)
├── bench.py           # Throughput benchmark
├── config/            # Config override files (not modules — exec'd at runtime)
│   ├── train_gpt2.py
│   ├── train_shakespeare_char.py
│   ├── finetune_shakespeare.py
│   └── eval_gpt2*.py
└── data/              # Dataset prep scripts (one subfolder per dataset)
```

---

## Non-Negotiable Rules

1. **`model.py` has no I/O side effects** — no `print` calls in forward pass, no file reads/writes; it is a pure nn.Module file.
2. **Config via globals** — all hyperparameters in `train.py`/`sample.py` are module-level globals, overridden via `configurator.py`; do not use argparse or introduce a config class in training scripts.
3. **No trainer framework** — use raw PyTorch loops; do not import HuggingFace `Trainer`, Lightning, or similar. `transformers` is allowed only for weight loading in `GPT.from_pretrained`.
4. **Weight tying** — `transformer.wte.weight` and `lm_head.weight` must remain shared (`lm_head.weight = wte.weight`). Do not break this.
5. **`torch.compile` compatible** — avoid Python control flow inside `nn.Module.forward()` that would break `torch.compile`; keep tensor operations symbolic.
6. **Checkpoint format** — checkpoints are dicts with keys `model`, `optimizer`, `model_args`, `iter_num`, `best_val_loss`, `config`. Do not alter this schema without updating load logic.

---

## Common Commands

```bash
# Prepare a dataset (example: Shakespeare char-level)
python data/shakespeare_char/prepare.py

# Train from scratch (single GPU)
python train.py config/train_shakespeare_char.py

# Train GPT-2 (124M) on OpenWebText (single GPU, override batch size)
python train.py config/train_gpt2.py --batch_size=8

# Resume training from checkpoint
python train.py config/train_gpt2.py --init_from=resume

# Multi-GPU training (4 GPUs, 1 node)
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py

# Fine-tune from GPT-2 pretrained weights
python train.py config/finetune_shakespeare.py

# Sample from a trained checkpoint
python sample.py --out_dir=out-shakespeare-char

# Sample from OpenAI GPT-2
python sample.py --init_from=gpt2

# Benchmark throughput
python bench.py

# Evaluate GPT-2 perplexity
python train.py config/eval_gpt2.py
```

---

## Environment Variables / Key Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `out_dir` | `out` | Directory for checkpoints |
| `dataset` | `openwebtext` | Dataset subfolder under `data/` |
| `device` | `cuda` | `cpu`, `cuda`, `cuda:0`, `mps` |
| `dtype` | `bfloat16` | `float32`, `bfloat16`, `float16` |
| `compile` | `True` | Enable `torch.compile` (PyTorch ≥ 2.0) |
| `init_from` | `scratch` | `scratch`, `resume`, `gpt2`, `gpt2-medium`, etc. |
| `wandb_log` | `False` | Enable W&B experiment tracking |
| `wandb_project` | `owt` | W&B project name |
| `batch_size` | `12` | Micro-batch size per GPU |
| `gradient_accumulation_steps` | `40` | Steps to accumulate before optimizer step |
| `max_iters` | `600000` | Total training iterations |
| `learning_rate` | `6e-4` | Peak learning rate |
| `block_size` | `1024` | Context window (sequence length) |

---

## Known TODOs

| Location | Description |
|----------|-------------|
| `model.py:137` | Investigate `UserWarning: functional_call was passed multiple values for tied weights` in `torch.compile` |
| `sample.py:65` | Make tokeniser loading more general to support arbitrary encoder/decoder schemes (not just char-level or GPT-2 tiktoken) |

---

## Recent Changes

| Date | Summary |
|------|---------|
| 2026-03-10 | Added Claude Code project intelligence structure (this file + `docs/`, `.claude/`) |
