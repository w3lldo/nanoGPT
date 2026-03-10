# SKILL: Refactor

> Use when asked to refactor, restructure, or improve existing code.

## Principles

1. **Preserve behaviour first** — run the smoke tests before and after; no behaviour changes in a refactor PR
2. **Respect file responsibilities** — do not move training logic into `model.py` or vice versa
3. **One concern at a time** — structural changes separate from behavioural changes
4. **Follow existing patterns** — refactor toward patterns already in use, not new abstractions

## Refactor Safety Checklist

- [ ] Smoke test passes before starting (see post-edit hook)
- [ ] Smoke test passes after finishing
- [ ] `python -m py_compile model.py train.py sample.py configurator.py` — no syntax errors
- [ ] No forbidden imports introduced
- [ ] Weight tying still holds
- [ ] Checkpoint schema unchanged

## Common Safe Refactors for nanoGPT

- **Extract a repeated block_size assertion** into a helper — safe if it's pure assertion logic
- **Rename local variables** for clarity (e.g. `B`, `T`, `C` comment annotations)
- **Add type hints** to function signatures in `model.py` — `import torch; Tensor = torch.Tensor`
- **Extract `get_lr` or `estimate_loss`** from global scope to clearly labelled function — already done; keep them

## What NOT to Refactor

> These look improvable but must not be changed.

| Thing | Why it must stay |
|-------|-----------------|
| `exec(open('configurator.py').read())` | This is the intentional config override mechanism — do not replace with argparse or a config class |
| `self.transformer.wte.weight = self.lm_head.weight` | Load-bearing weight tying — breaks GPT-2 checkpoint compatibility if removed |
| `np.memmap` recreated inside `get_batch()` each call | Intentional — avoids a well-documented memory leak (see code comment) |
| `gradient_accumulation_steps //= ddp_world_size` in-place | Required for correct DDP scaling — not a bug |
| `model.require_backward_grad_sync` toggle | Necessary DDP optimisation — removing it causes unnecessary gradient syncs on every micro-step |
| `_orig_mod.` prefix stripping in checkpoint loading | Handles `torch.compile` checkpoint artefact — removing it breaks resume from compiled checkpoints |
