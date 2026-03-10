# ADR-005: Weight Tying Between Token Embeddings and LM Head

- **Status:** Accepted
- **Date:** 2022-01-01

## Context

The output linear layer (`lm_head`) maps from `n_embd` to `vocab_size`. The token embedding matrix (`transformer.wte`) also has shape `(vocab_size, n_embd)`. These two matrices can share weights, halving the vocabulary parameter count.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Tied weights (`lm_head.weight = wte.weight`) | Fewer parameters; matches GPT-2 paper; better generalisation | Slightly quirky with `torch.compile` |
| Separate weights | Simpler code | More parameters; doesn't match original GPT-2 |

## Decision

`self.lm_head.weight = self.transformer.wte.weight` — the two are the same tensor. This matches the original GPT-2 implementation.

## Consequences

- **Do not untie these weights** — loading OpenAI GPT-2 checkpoints (`GPT.from_pretrained`) depends on this
- `torch.compile` generates a harmless warning about functional_call being passed multiple values for tied weights — this is a known PyTorch issue and can be ignored (TODO: investigate)
- Parameter counts reported by `get_num_params()` exclude position embeddings but include the shared token embedding/LM head weight (since it contributes to both)
