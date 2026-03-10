# ADR-002: Single-File Architecture (No Framework)

- **Status:** Accepted
- **Date:** 2022-01-01

## Context

Many GPT implementations are spread across dozens of files, require config systems, and have significant indirection. This makes them hard to read and understand as a learning resource.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Single file per concern (model.py, train.py, sample.py) | Easy to read end-to-end; self-contained | Less modular for large-scale production |
| Full project structure (src/, lib/, etc.) | Standard software engineering | Obscures the core logic with plumbing |
| Monolithic single file | Maximally simple to share | Too large to navigate |

## Decision

Two primary files: `model.py` (all model code) and `train.py` (all training code), with minimal supporting scripts. Each file can be read top-to-bottom and understood completely.

## Consequences

- `model.py` must never contain training logic; `train.py` must never contain model definitions
- Any new component goes into the most appropriate existing file — do not create new modules without strong justification
- The codebase prioritises readability over extensibility
