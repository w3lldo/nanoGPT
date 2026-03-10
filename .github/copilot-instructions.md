# GitHub Copilot Instructions — nanoGPT

This file is automatically read by GitHub Copilot as repository-level context every session.

---

## Extended Documentation

| File | Contents |
|------|----------|
| [`CLAUDE.md`](../CLAUDE.md) | Root project memory — start here |
| [`docs/architecture.md`](../docs/architecture.md) | Full architecture & data flows |
| [`docs/decisions/`](../docs/decisions/) | Architectural Decision Records |
| [`docs/runbooks/`](../docs/runbooks/) | Setup, training, and inference runbooks |

---

## Skill Prompts (attach with `#` in Copilot Chat)

| Prompt | When to use |
|--------|-------------|
| [`.claude/skills/code-review/SKILL.md`](../.claude/skills/code-review/SKILL.md) | Review a PR or diff |
| [`.claude/skills/refactor/SKILL.md`](../.claude/skills/refactor/SKILL.md) | Safe refactor |
| [`.claude/skills/release/SKILL.md`](../.claude/skills/release/SKILL.md) | Prepare a release |
| [`tools/prompts/add-model-component.md`](../tools/prompts/add-model-component.md) | Add a new model component |
| [`tools/prompts/train-new-dataset.md`](../tools/prompts/train-new-dataset.md) | Add and train on a new dataset |
| [`tools/prompts/write-tests.md`](../tools/prompts/write-tests.md) | Write tests |

---

## Non-Negotiable Rules

1. **`model.py` has no I/O side effects** — no `print` in forward pass, no file access. Pure `nn.Module`.
2. **Config via globals** — all hyperparameters are module-level globals in `train.py`/`sample.py`, overridden via `configurator.py`. Do not use argparse.
3. **No trainer framework** — no HuggingFace `Trainer`, PyTorch Lightning, or Accelerate. Raw PyTorch loops only.
4. **Weight tying** — `transformer.wte.weight` and `lm_head.weight` must remain the same tensor.
5. **`torch.compile` compatible** — no Python-level control flow inside `forward()` that changes tensor outputs.
6. **Checkpoint schema** — `{'model', 'optimizer', 'model_args', 'iter_num', 'best_val_loss', 'config'}`. Do not change.

---

## Quick Command Reference

```bash
# Prepare data (example)
python data/shakespeare_char/prepare.py

# Train (single GPU)
python train.py config/train_shakespeare_char.py

# Multi-GPU
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py

# Sample
python sample.py --out_dir=out-shakespeare-char

# Smoke test
python -c "from model import GPT, GPTConfig; GPT(GPTConfig(n_layer=2,n_head=2,n_embd=64))"
```
