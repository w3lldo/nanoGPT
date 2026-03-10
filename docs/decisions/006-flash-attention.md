# ADR-006: Flash Attention via scaled_dot_product_attention

- **Status:** Accepted
- **Date:** 2023-01-01

## Context

Standard attention computes the full `(T, T)` attention matrix in GPU global memory, which is O(T²) in memory and slow. Flash Attention recomputes attention scores in tiles using CUDA kernels, achieving O(T) memory and significant speed gains.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| `torch.nn.functional.scaled_dot_product_attention` | Built into PyTorch ≥ 2.0; uses Flash Attention CUDA kernels automatically | Requires PyTorch ≥ 2.0 |
| `flash-attn` package | Maximum performance | External C++ dependency; complex install |
| Manual attention (tril mask + softmax) | No dependency; always works | Slow for long sequences; O(T²) memory |

## Decision

Use `torch.nn.functional.scaled_dot_product_attention` when available (`hasattr` check at init time). Fall back to the manual causal-mask attention implementation for older PyTorch versions.

## Consequences

- PyTorch ≥ 2.0 is strongly recommended for training performance — the fallback is functionally correct but slower
- `is_causal=True` is passed to `scaled_dot_product_attention` — no explicit causal mask buffer is allocated when Flash Attention is active
- The fallback path registers a `bias` buffer (causal mask) in `CausalSelfAttention.__init__` — this only exists when Flash Attention is *not* available; do not access `block.attn.bias` unconditionally in weight manipulation code
