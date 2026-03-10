# Runbook: Training a Model

## Pre-flight Checklist
- [ ] Dataset prepared (`data/<dataset>/train.bin` and `val.bin` exist)
- [ ] Environment activated (`conda activate nanogpt`)
- [ ] GPU available or `device=cpu`/`device=mps` set for Mac
- [ ] `out_dir` set (default: `out`)

---

## Training from Scratch

```bash
# Minimal example: Shakespeare character-level (~2 mins on CPU)
python train.py config/train_shakespeare_char.py

# GPT-2 (124M) on OpenWebText (needs A100, ~4 days)
python train.py config/train_gpt2.py

# Override any param inline
python train.py config/train_gpt2.py \
  --batch_size=8 \
  --grad_accum=40 \
  --out_dir=out-my-run
```

## Fine-tuning from GPT-2 Pretrained Weights

```bash
# Fine-tune GPT-2 on Shakespeare
python train.py config/finetune_shakespeare.py
```

## Resuming an Interrupted Run

```bash
python train.py config/train_gpt2.py --init_from=resume --out_dir=out
```

## Multi-GPU Training (DDP via torchrun)

```bash
# 4 GPUs, 1 node
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py

# 8 GPUs across 2 nodes (run on each node)
# Node 0 (master, IP: 123.456.123.456):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=123.456.123.456 --master_port=1234 \
  train.py config/train_gpt2.py

# Node 1 (worker):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=123.456.123.456 --master_port=1234 \
  train.py config/train_gpt2.py
```

> **Note:** With DDP, `gradient_accumulation_steps` is divided automatically across workers.

## Enabling W&B Logging

```bash
wandb login  # one-time
python train.py config/train_gpt2.py --wandb_log=True --wandb_project=my-project
```

## Understanding the Output

```
tokens per iteration will be: 491,520
iter 0: loss 4.2341, time 1234.56ms, mfu 0.00%
step 2000: train loss 2.1234, val loss 2.3456
saving checkpoint to out
```

- **MFU**: Model Flops Utilisation vs. A100 bfloat16 peak. Target ≥ 40% on A100.
- **Checkpoint**: saved to `<out_dir>/ckpt.pt` every `eval_interval` steps (and always if `always_save_checkpoint=True`)

## Key Hyperparameters Reference

| Param | Default | Notes |
|-------|---------|-------|
| `learning_rate` | `6e-4` | For GPT-2 scale; reduce for fine-tuning (~1e-4) |
| `dropout` | `0.0` | 0 for pretraining; `0.1`–`0.2` for fine-tuning |
| `bias` | `False` | Slightly better and faster without bias |
| `gradient_accumulation_steps` | `40` | Increase to simulate larger batch on fewer GPUs |
| `decay_lr` | `True` | Cosine decay; disable for quick experiments |
