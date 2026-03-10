# Architecture — nanoGPT

> Last updated: 2026-03-10

## Overview

nanoGPT is a single-file, decoder-only Transformer (GPT-2 family) trained with raw PyTorch. It is a batch training system: data arrives as memory-mapped binary token files, passes through a causal self-attention stack, and produces next-token predictions via a weight-tied language-model head. It depends on no training framework — only PyTorch, NumPy, and optionally tiktoken (tokenisation) and HuggingFace `transformers` (pretrained weight loading).

---

## High-Level Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                            train.py                              │
│                                                                  │
│  data/*.bin  ──► get_batch() ──► GPT.forward(X, Y)              │
│  (memmap)        (random crop)    │                              │
│                                   ├── logits ──► cross_entropy   │
│                                   │              (loss)          │
│                                   └── loss ──► scaler.backward() │
│                                                ──► AdamW.step()  │
│                                                                  │
│  Checkpoints: out/ckpt.pt  (saved every eval_interval steps)    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                            model.py                              │
│                                                                  │
│  GPTConfig                                                       │
│    └── GPT (nn.Module)                                           │
│          ├── transformer.wte  (token embeddings)                 │
│          ├── transformer.wpe  (position embeddings)              │
│          ├── transformer.drop (embedding dropout)                │
│          ├── transformer.h    (N × Block)                        │
│          │     ├── ln_1  ──► CausalSelfAttention                 │
│          │     └── ln_2  ──► MLP                                 │
│          ├── transformer.ln_f (final LayerNorm)                  │
│          └── lm_head          (weight-tied to wte)               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Architectural Pattern

**Single-responsibility flat scripts** — no layered architecture. Each script has one job:

- `model.py` — model *definition only* (no I/O, no training)
- `train.py` — full training orchestration (no model definition)
- `sample.py` — inference only
- `configurator.py` — config override (exec'd, not imported as module)

**Rules:**
- `model.py` must never import from `train.py`
- `train.py`/`sample.py` import from `model.py` only
- No circular imports
- Config is via globals, not a config class or argparse

---

## Data Flow — Training

1. **Dataset prep** (one-time): `data/<dataset>/prepare.py` tokenises raw text → `train.bin` / `val.bin` (uint16 NumPy arrays, memory-mapped)
2. **Config loading**: `train.py` defines default hyperparameters as module globals → `exec(configurator.py)` overrides from CLI args and/or a `config/*.py` file
3. **DDP init** (if multi-GPU): `torchrun` sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE` env vars → `init_process_group(backend='nccl')` → each rank gets its own CUDA device
4. **Data loading**: `get_batch(split)` creates a fresh `np.memmap` each call (avoids memory leak), randomly crops `batch_size` windows of `block_size` tokens, pins memory, and ships to GPU asynchronously
5. **Model init**: one of three paths — `scratch` (random init), `resume` (load `out/ckpt.pt`), `gpt2*` (load HuggingFace weights + transpose Conv1D → Linear)
6. **Compilation**: `torch.compile(model)` (optional, PyTorch ≥ 2.0) — wraps model before DDP wrapping
7. **DDP wrap**: `DDP(model, device_ids=[ddp_local_rank])` — gradient sync is disabled for all but the last micro-step
8. **Training loop**:
   - Set LR via cosine schedule with warmup (`get_lr(iter_num)`)
   - Every `eval_interval` steps: `estimate_loss()` → print stats → optional W&B log → save checkpoint
   - Gradient accumulation: run `gradient_accumulation_steps` micro-steps, scaling loss; prefetch next batch asynchronously
   - `GradScaler.scale(loss).backward()` → `clip_grad_norm_` → `scaler.step(optimizer)` → `optimizer.zero_grad(set_to_none=True)`
   - Log MFU (model flops utilisation vs. A100 bfloat16 peak)

---

## Data Flow — Inference (sample.py)

1. Load model from `out/ckpt.pt` **or** `GPT.from_pretrained('gpt2*')`
2. Set model to `eval()` mode
3. Detect tokeniser: if dataset `meta.pkl` has `stoi`/`itos` (char-level), use it; else default to `tiktoken` GPT-2 encoding
4. Encode prompt string → token ID tensor
5. Autoregressive loop in `GPT.generate()`: forward only last position's logits → temperature scale → top-k filter → softmax → multinomial sample → append → repeat
6. Decode output token IDs → print

---

## Model Internals

### GPTConfig (dataclass)

| Field | Default | Meaning |
|-------|---------|---------|
| `block_size` | 1024 | Context window (max sequence length) |
| `vocab_size` | 50304 | Padded to nearest multiple of 64 for efficiency |
| `n_layer` | 12 | Number of Transformer blocks |
| `n_head` | 12 | Attention heads per block |
| `n_embd` | 768 | Embedding dimension |
| `dropout` | 0.0 | Dropout rate (0 for pretraining, 0.1+ for fine-tuning) |
| `bias` | True | Bias in Linear and LayerNorm |

### GPT-2 Size Variants

| Variant | n_layer | n_head | n_embd | Params |
|---------|---------|--------|--------|--------|
| gpt2 | 12 | 12 | 768 | 124M |
| gpt2-medium | 24 | 16 | 1024 | 350M |
| gpt2-large | 36 | 20 | 1280 | 774M |
| gpt2-xl | 48 | 25 | 1600 | 1558M |

### CausalSelfAttention

- Single `c_attn` Linear projects input to Q, K, V for all heads at once (dim = `3 * n_embd`)
- Reshapes to `(B, n_head, T, head_size)` per head
- **Flash Attention** (`scaled_dot_product_attention`) used when PyTorch ≥ 2.0; falls back to manual masked attention with causal bias buffer
- Output projected back with `c_proj` Linear + residual dropout

### MLP

- 4× expansion: `n_embd` → `4 * n_embd` → `n_embd`
- Activation: GELU
- No gating (unlike LLaMA SwiGLU)

### Block

- Pre-norm: LayerNorm applied *before* attention and MLP (not after, unlike original GPT paper)
- Residual connections around both attention and MLP sub-layers

### Weight Initialisation

- Linears: `N(0, 0.02)`; biases: zeros
- Embeddings: `N(0, 0.02)`
- Residual projections (`c_proj`): scaled by `1/sqrt(2 * n_layer)` per GPT-2 paper to control residual stream growth

### Weight Tying

`transformer.wte.weight = lm_head.weight` — the token embedding matrix is reused as the output projection. This halves vocabulary parameter count and is critical for GPT-2 compatibility.

---

## Optimiser & LR Schedule

- **AdamW** with fused CUDA kernel (if available): `lr=6e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.1`
- Weight decay applied to all 2D parameters (weight matrices, embeddings); biases and LayerNorm params are exempt
- **Cosine decay with linear warmup**: warmup for `warmup_iters` steps, cosine decay to `min_lr = lr/10` over `lr_decay_iters` steps
- **Gradient clipping**: `clip_grad_norm_(model.parameters(), 1.0)`
- **GradScaler**: enabled only for `float16` dtype (bfloat16 does not need it)

---

## Checkpoint Format

Saved to `out/ckpt.pt` as a Python dict:

```python
{
    'model': raw_model.state_dict(),      # model weights
    'optimizer': optimizer.state_dict(),  # optimizer state
    'model_args': model_args,             # GPTConfig kwargs dict
    'iter_num': iter_num,                 # current iteration
    'best_val_loss': best_val_loss,       # best validation loss seen
    'config': config,                     # full config dict snapshot
}
```

> **Do not change this schema** without updating the resume loading logic in `train.py` (lines ~158–180) and `sample.py` (lines ~35–46).

---

## Observability

- **Console**: loss, LR, MFU printed every `log_interval` steps (default: every step)
- **W&B**: optional, enabled via `wandb_log=True`; logs `train/loss`, `val/loss`, `lr`, `mfu`
- **MFU**: estimated as `achieved_FLOPS / A100_bfloat16_peak_FLOPS` — useful for hardware utilisation tracking

---

## Deployment

nanoGPT is a research/educational codebase — not a production service. Run locally or on a GPU cluster via `torchrun`. There is no HTTP API, container image, or health probe. Deployment means running `train.py` and loading the resulting `out/ckpt.pt` with `sample.py`.
