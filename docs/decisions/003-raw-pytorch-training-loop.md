# ADR-003: Raw PyTorch Training Loop (No Trainer Abstraction)

- **Status:** Accepted
- **Date:** 2022-01-01

## Context

HuggingFace `Trainer`, PyTorch Lightning, and similar frameworks abstract the training loop behind high-level APIs. While convenient, they hide the details that educators and researchers need to understand.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Raw PyTorch loop | Fully transparent; every step is explicit | More code to write |
| HuggingFace Trainer | Less boilerplate | Hides gradient accumulation, mixed precision, DDP details |
| PyTorch Lightning | Modular, tested | Significant framework overhead, not educational |

## Decision

Write the training loop explicitly in `train.py`. Every step — DDP wrapping, GradScaler, gradient accumulation, gradient clipping, LR scheduling, checkpoint saving — is visible and annotated.

## Consequences

- `transformers` library is permitted **only** for loading pretrained GPT-2 weights (`GPT.from_pretrained`) — never for training
- No Lightning, no HuggingFace Trainer, no Accelerate
- Gradient accumulation, mixed precision, and DDP are all handled manually with explicit PyTorch APIs
