# Prompt: Train on a New Dataset

## When to Use
When adding support for a new text dataset to nanoGPT.

## Files to Create/Edit

| File | Change |
|------|--------|
| `data/<dataset>/prepare.py` | [NEW] tokenisation prep script |
| `config/train_<dataset>.py` | [NEW] hyperparameter config for this dataset |
| `CLAUDE.md` | Add dataset to common commands if notable |

## Steps

### 1. Create the dataset directory and prepare.py

```
data/
└── <dataset>/
    └── prepare.py   ← you create this
```

The prep script must produce:
- `data/<dataset>/train.bin` — NumPy uint16 array (token IDs)
- `data/<dataset>/val.bin`   — NumPy uint16 array (token IDs)
- `data/<dataset>/meta.pkl`  — (optional) for char-level: `{'vocab_size': N, 'stoi': {...}, 'itos': {...}}`

### 2. Tokenisation options

**Option A: Character-level (small datasets)**
```python
import numpy as np, pickle, os

text = open('input.txt', 'r').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]

n = int(0.9 * len(text))
train_ids = np.array(encode(text[:n]), dtype=np.uint16)
val_ids   = np.array(encode(text[n:]), dtype=np.uint16)

os.makedirs(os.path.dirname(__file__), exist_ok=True)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}, f)
```

**Option B: BPE via tiktoken (larger datasets)**
```python
import tiktoken, numpy as np, os
enc = tiktoken.get_encoding('gpt2')  # 50257-token vocabulary

# tokenise and save
train_ids = np.array(enc.encode_ordinary(train_text), dtype=np.uint16)
val_ids   = np.array(enc.encode_ordinary(val_text),   dtype=np.uint16)
train_ids.tofile('data/<dataset>/train.bin')
val_ids.tofile('data/<dataset>/val.bin')
# no meta.pkl needed — sample.py falls back to tiktoken automatically
```

### 3. Create a config file

```python
# config/train_<dataset>.py
out_dir = 'out-<dataset>'
dataset = '<dataset>'          # must match data/<dataset>/ folder name

# Model size — adjust for dataset size
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256

# Training
max_iters = 5000
lr_decay_iters = 5000
eval_interval = 250
eval_iters = 200
log_interval = 10

# Optimiser
learning_rate = 1e-3
min_lr = 1e-4
batch_size = 64
dropout = 0.2      # fine-tuning: 0.1–0.2; pretraining: 0.0
```

### 4. Run preparation and training

```bash
python data/<dataset>/prepare.py
python train.py config/train_<dataset>.py
```

## Rules
- `train.bin` / `val.bin` must be `dtype=np.uint16` — uint16 is assumed in `get_batch()`
- Character-level datasets need `meta.pkl` with `stoi`/`itos`; BPE datasets don't
- The `dataset` variable in the config must exactly match the subfolder name under `data/`
- Reduce `block_size` and model dims for small datasets to avoid overfitting
