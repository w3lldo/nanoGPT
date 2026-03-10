# Prompt: Add a New Model Component

## When to Use
When adding a new architectural component to `model.py` (e.g. a new attention variant, normalisation layer, or block modification).

## Files to Edit

| File | Change |
|------|--------|
| `model.py` | Add new `nn.Module` class; update `Block` or `GPT` to use it |
| `CLAUDE.md` | Update source layout if significant |
| `docs/architecture.md` | Document the new component in model internals |
| `docs/decisions/` | Create new ADR if this is a significant architectural decision |

## Steps

### 1. Define the component as an `nn.Module`
```python
class MyComponent(nn.Module):
    def __init__(self, config):
        super().__init__()
        # initialise sub-layers using config fields (n_embd, n_head, dropout, bias)
        self.linear = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x shape: (B, T, C) — always document tensor shapes
        return self.dropout(self.linear(x))
```

### 2. Add config parameters if needed
In `GPTConfig` dataclass at the top of `model.py`:
```python
@dataclass
class GPTConfig:
    # ... existing fields ...
    my_new_param: float = 0.0  # description of what this does
```

### 3. Wire into Block or GPT
```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.my_component = MyComponent(config)  # <-- add here
```

### 4. Update weight init if needed
If your component has Linear layers, the default `_init_weights` already handles them (`N(0, 0.02)` for weights, zeros for biases). For residual output projections, add scaled init:
```python
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):  # or your output projection name
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

### 5. Smoke test
```bash
python -c "from model import GPT, GPTConfig; m = GPT(GPTConfig(n_layer=2,n_head=2,n_embd=64)); print('params:', m.get_num_params())"
```

## Rules
- All new `nn.Module` classes go in `model.py` — do not create new files
- Accept `config` (a `GPTConfig` instance) as the only constructor argument
- Document every tensor's shape with a comment in `forward()`
- Do not add `print()` statements in `forward()` — they cause performance issues
- Check that `torch.compile` compatibility is maintained (no Python branching on tensor values in forward)
