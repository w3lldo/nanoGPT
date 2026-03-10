# SKILL: Release

> Use when preparing a version bump or publishing changes.

## When to Release

nanoGPT does not have a traditional release cadence — it is a research/educational codebase. "Releasing" typically means: tagging a stable commit after a meaningful change, for citation or reproducibility purposes.

## Version Bump Decision

| Change type | Bump |
|-------------|------|
| Bug fix, documentation, performance tuning | Patch (`x.y.Z`) |
| New feature (new dataset support, new config, new script) | Minor (`x.Y.0`) |
| Breaking change (checkpoint schema, API, removed file) | Major (`X.0.0`) |

## Step-by-Step

1. **Verify everything passes:**
   ```bash
   # Syntax check
   python -m py_compile model.py train.py sample.py configurator.py
   
   # Smoke test with CPU
   python train.py config/train_shakespeare_char.py \
     --device=cpu --compile=False --eval_iters=2 --max_iters=10 \
     --n_layer=2 --n_head=2 --n_embd=64 --batch_size=2 --block_size=32
   ```

2. **Update `CLAUDE.md` Recent Changes table** with a summary of what changed

3. **Commit:**
   ```bash
   git add -A
   git commit -m "chore: release vx.y.z — <one-line summary>"
   ```

4. **Tag:**
   ```bash
   git tag -a vx.y.z -m "Release vx.y.z — <summary>"
   git push origin main --tags
   ```

## Breaking Change Protocol

1. If the checkpoint schema changes: update loading code in **both** `train.py` and `sample.py`
2. Create a new ADR in `docs/decisions/`
3. Update `docs/architecture.md` checkpoint format section
4. Update `CLAUDE.md` with the new schema

## Post-Release Verification

- [ ] `git tag` shows the new tag
- [ ] `python sample.py --init_from=gpt2 --device=cpu --num_samples=1 --max_new_tokens=5` still works
- [ ] README.md still matches the actual commands
