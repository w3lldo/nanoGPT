# Runbook: Sampling / Inference

## From a Trained Checkpoint

```bash
# Sample 10 completions from a checkpoint in out-shakespeare-char/
python sample.py --out_dir=out-shakespeare-char

# Control generation quality
python sample.py \
  --out_dir=out \
  --num_samples=5 \
  --max_new_tokens=200 \
  --temperature=0.8 \
  --top_k=200 \
  --device=cpu
```

## From OpenAI GPT-2 Pretrained Weights

```bash
# No training required — downloads weights from HuggingFace
python sample.py --init_from=gpt2             # 124M params
python sample.py --init_from=gpt2-medium      # 350M params
python sample.py --init_from=gpt2-large       # 774M params
python sample.py --init_from=gpt2-xl          # 1558M params
```

## With a Custom Starting Prompt

```bash
# Inline string prompt
python sample.py --out_dir=out --start="Once upon a time"

# From a file
python sample.py --out_dir=out --start="FILE:my_prompt.txt"
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | 10 | Number of independent completions |
| `max_new_tokens` | 500 | Tokens to generate per sample |
| `temperature` | 0.8 | < 1.0 = more focused; > 1.0 = more random |
| `top_k` | 200 | Clamp to top-k logits before sampling |
| `seed` | 1337 | Random seed for reproducibility |
| `compile` | False | `torch.compile` for faster inference (optional) |

## Tokeniser Auto-Detection

- If `data/<dataset>/meta.pkl` has `stoi`/`itos` (char-level) — uses it
- Otherwise defaults to GPT-2 tiktoken encoding (`cl100k_base`-like)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: ckpt.pt` | No checkpoint in `out_dir` | Train first, or use `--init_from=gpt2` |
| Repetitive / degenerate output | Temperature too low or top_k too small | Increase `temperature` or `top_k` |
| CUDA OOM on large model | Not enough GPU memory for inference | Use `--device=cpu` |
| `_orig_mod.` prefix in state dict | Loaded a `torch.compile`d checkpoint | Automatically stripped — no action needed |
