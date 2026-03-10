# SKILL: Code Review

> Use when asked to review a PR, diff, or code block in this repository.

## Review Checklist

### Safety & Correctness
- [ ] No silent tensor shape mismatches — verify `(B, T, C)` / `(B, nh, T, hs)` dims at each step
- [ ] No `.view()` or `.reshape()` on non-contiguous tensors without `.contiguous()` first
- [ ] `torch.no_grad()` applied to inference and evaluation code (`estimate_loss`, `generate`)
- [ ] No in-place operations on tensors that require gradients
- [ ] No hardcoded device (`'cuda'`) — should use the `device` variable

### Architecture (model.py)
- [ ] `model.py` contains no I/O, no file access, no `print` in forward pass
- [ ] Weight tying intact after any change: `model.transformer.wte.weight is model.lm_head.weight`
- [ ] `torch.compile` compatible — no Python-level branching that changes output inside `forward()`
- [ ] `crop_block_size()` still works if block_size was changed

### Training (train.py)
- [ ] New hyperparameter is a module-level global (not argparse), included in `config_keys`
- [ ] Checkpoint schema unchanged: `{'model', 'optimizer', 'model_args', 'iter_num', 'best_val_loss', 'config'}`
- [ ] Loss is divided by `gradient_accumulation_steps` before `.backward()`
- [ ] DDP gradient sync only on last micro-step (`model.require_backward_grad_sync` toggled)
- [ ] `optimizer.zero_grad(set_to_none=True)` used (not `zero_grad()`)

### Models & Data
- [ ] Token IDs are `int64` (not `int32` or `float`) when passed to `GPT.forward()`
- [ ] Sequence length ≤ `block_size` enforced before forwarding
- [ ] `np.memmap` recreated each batch call (avoids memory leak — see train.py comment)

### Observability
- [ ] `print()` used for training console output only (not inside nn.Module)
- [ ] New metrics logged via `wandb.log()` if `wandb_log` is True
- [ ] No secrets or API keys hardcoded

### Tests & Validation
- [ ] New model component passes the smoke test:
  `python -c "from model import GPT, GPTConfig; GPT(GPTConfig(n_layer=2,n_head=2,n_embd=64))"`
- [ ] New training logic validated with quick CPU run (see post-edit hook)

### Documentation
- [ ] New config param added to `CLAUDE.md` Environment Variables table
- [ ] Architecture change reflected in `docs/architecture.md`
- [ ] Breaking change documented in a new ADR

## Common Issues in nanoGPT

| Pattern seen | Problem | Correct approach |
|-------------|---------|-----------------|
| `model.forward(x)` instead of `model(x, targets)` | Missing loss computation | Pass `targets` to get `(logits, loss)` |
| Accessing `block.attn.bias` unconditionally | `bias` buffer only exists without Flash Attention | Gate on `not block.attn.flash` |
| `torch.save(model, ...)` | Saves entire model object, not state dict | `torch.save({'model': model.state_dict(), ...})` |
| `import argparse` in train.py | Breaks the globals config pattern | Use module-level global + `config_keys` |
| `loss.backward()` without scaling | Float16 underflow | `scaler.scale(loss).backward()` |
| `optimizer.zero_grad()` | Slower than `set_to_none` | `optimizer.zero_grad(set_to_none=True)` |
