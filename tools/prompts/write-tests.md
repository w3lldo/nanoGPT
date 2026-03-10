# Prompt: Write Tests

## When to Use
When adding unit tests or validation scripts for model components or training utilities.

> **Note:** nanoGPT has no test suite by default. Tests are standalone scripts placed in `tools/scripts/` or a `tests/` folder.

## Unit Test Pattern (Model Component)

```python
# tests/test_model.py
import torch
from model import GPT, GPTConfig, CausalSelfAttention, MLP, Block

def get_small_config():
    return GPTConfig(block_size=32, vocab_size=64, n_layer=2, n_head=2, n_embd=64, dropout=0.0, bias=True)

def test_causal_self_attention_shape():
    cfg = get_small_config()
    attn = CausalSelfAttention(cfg)
    B, T, C = 2, 16, cfg.n_embd
    x = torch.randn(B, T, C)
    y = attn(x)
    assert y.shape == (B, T, C), f"Expected {(B, T, C)}, got {y.shape}"
    print("PASS: CausalSelfAttention shape")

def test_mlp_shape():
    cfg = get_small_config()
    mlp = MLP(cfg)
    B, T, C = 2, 16, cfg.n_embd
    x = torch.randn(B, T, C)
    y = mlp(x)
    assert y.shape == (B, T, C), f"Expected {(B, T, C)}, got {y.shape}"
    print("PASS: MLP shape")

def test_gpt_forward():
    cfg = get_small_config()
    model = GPT(cfg)
    model.eval()
    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is not None
    print(f"PASS: GPT forward — loss={loss.item():.4f}")

def test_weight_tying():
    cfg = get_small_config()
    model = GPT(cfg)
    assert model.transformer.wte.weight is model.lm_head.weight, "Weight tying broken!"
    print("PASS: weight tying")

def test_generate():
    cfg = get_small_config()
    model = GPT(cfg)
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=10)
    assert out.shape == (1, 11), f"Expected (1,11), got {out.shape}"
    print("PASS: generate")

if __name__ == '__main__':
    test_causal_self_attention_shape()
    test_mlp_shape()
    test_gpt_forward()
    test_weight_tying()
    test_generate()
    print("All tests passed!")
```

## Run Tests
```bash
python tests/test_model.py
```

## Where Tests Live
`tests/` folder at repo root (create if needed), or `tools/scripts/` for one-off validation scripts.

## Naming Convention
`test_<component>_<scenario>` — e.g. `test_causal_self_attention_shape`, `test_gpt_forward_no_targets`

## Rules
- Always use `model.eval()` and `torch.no_grad()` for inference-only tests
- Use small configs (`n_layer=2`, `n_head=2`, `n_embd=64`) for fast tests
- Assert tensor shapes explicitly — shape bugs are the #1 source of errors
- Do not depend on CUDA in tests — use CPU (`device='cpu'`)
- Test weight tying in any test that calls `GPT.__init__`
