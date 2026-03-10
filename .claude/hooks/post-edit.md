# Post-Edit Hook

> Run these checks after every code change.

## Verification Checklist

- [ ] **Syntax check:** `python -m py_compile model.py train.py sample.py configurator.py`
- [ ] **Quick smoke test (model.py changes):**
  ```bash
  python -c "from model import GPT, GPTConfig; m = GPT(GPTConfig(n_layer=2,n_head=2,n_embd=64)); print('OK:', m.get_num_params(), 'params')"
  ```
- [ ] **Quick smoke test (train.py changes):**
  ```bash
  python train.py config/train_shakespeare_char.py \
    --device=cpu --compile=False --eval_iters=2 --max_iters=3 \
    --n_layer=2 --n_head=2 --n_embd=64 --batch_size=2 --block_size=32
  ```
- [ ] **Sampling still works:**
  ```bash
  python sample.py --init_from=gpt2 --device=cpu --num_samples=1 --max_new_tokens=10 --compile=False
  ```

## Documentation Checklist

- [ ] New config parameter? → Add to `CLAUDE.md` Environment Variables table
- [ ] New model component? → Update `docs/architecture.md` model internals section
- [ ] Changed checkpoint schema? → Update `docs/architecture.md` checkpoint format + `docs/decisions/` if significant
- [ ] Breaking change? → Create or update ADR in `docs/decisions/`

## Non-Negotiable Verification

- [ ] `model.py` has no I/O: no `open()`, no `torch.save()`, no `wandb` calls
- [ ] Weight tying intact: `model.transformer.wte.weight is model.lm_head.weight`
- [ ] Config globals still work: `python train.py --n_layer=2 --compile=False --max_iters=1 --device=cpu`
- [ ] No forbidden imports introduced (`argparse`, `Trainer`, `lightning`, `accelerate`)
