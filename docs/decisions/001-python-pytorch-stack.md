# ADR-001: Python + PyTorch as the Implementation Stack

- **Status:** Accepted
- **Date:** 2022-01-01

## Context

nanoGPT aims to be the simplest, cleanest implementation of GPT-2 for educational and research purposes. The choice of language and ML framework was central to achieving this goal.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Python + PyTorch | Widest ML researcher adoption, excellent GPU support, `torch.compile`, Flash Attention | Slower than C++/CUDA for production |
| Python + JAX/Flax | Functional style, XLA compilation | Smaller community, less familiar to most |
| C++/CUDA | Maximum speed | Prohibitive implementation complexity for educational use |

## Decision

Python 3.10+ with PyTorch. PyTorch is the de facto framework for ML research and has excellent tooling (DDP, `torch.compile`, CUDA kernels via `scaled_dot_product_attention`).

## Consequences

- All model and training code uses PyTorch APIs exclusively
- Training scripts require Python ≥ 3.10 and PyTorch ≥ 2.0 (for `torch.compile` and Flash Attention)
- The codebase is immediately accessible to the broad ML research community
