# Pre-Edit Hook

> Read this before making any code changes.

## Checklist

1. **Identify the file** being edited — is it `model.py`, `train.py`, `sample.py`, or `configurator.py`?
2. **Re-read Non-Negotiable Rules** in root `CLAUDE.md`
3. **Check the relevant ADR** in `docs/decisions/` if touching a design decision area
4. **Pick the right task prompt** from `tools/prompts/` for this type of change

## File → Responsibility Map

| Editing... | Is responsible for | Must NOT touch |
|---|---|---|
| `model.py` | GPT architecture, weight init, generation | Training loops, file I/O, checkpointing |
| `train.py` | Training orchestration, DDP, data loading, checkpointing | Model architecture |
| `sample.py` | Inference and text generation only | Training logic |
| `configurator.py` | CLI arg / config file override system | Nothing else |
| `config/*.py` | Hyperparameter presets | Model architecture |

## Before Touching Model Architecture

- Does this change break weight tying? (`lm_head.weight = wte.weight`)
- Does this change break checkpoint compatibility? (schema: `model`, `optimizer`, `model_args`, `iter_num`, `best_val_loss`, `config`)
- Does this break `torch.compile` compatibility? (avoid Python-level branching inside `forward()`)
- Does the `crop_block_size()` surgery still work after your change?

## Before Touching Training Logic

- Does the gradient accumulation pattern still correctly scale the loss?
- Does DDP gradient sync still only happen on the last micro-step?
- Is the new config parameter added as a module-level global (not argparse)?
- Is the new config parameter included in `config_keys`?
