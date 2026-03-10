# ADR-004: Globals-Based Config System

- **Status:** Accepted
- **Date:** 2022-01-01

## Context

Configuration management is one of the main sources of complexity in ML training code. Standard approaches (argparse, dataclasses, YAML/hydra) all add indirection between the config declaration and its use.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Module globals + `exec(configurator.py)` | Zero indirection; variables are just in scope | Non-standard; `exec` is unusual |
| argparse | Familiar to Python developers | Verbose; no file-based config |
| Hydra / OmegaConf | Powerful composition | Heavy dependency; obscures simple cases |
| Dataclass config | Type-safe | Requires `config.` prefix everywhere |

## Decision

All hyperparameters are module-level globals in `train.py` / `sample.py`. `configurator.py` is `exec`'d at runtime to override these globals via a config file path and/or `--key=value` CLI arguments.

## Consequences

- No `config.` prefix needed anywhere — variables are used directly
- Config files in `config/` are plain Python scripts (not YAML/JSON) — they can contain expressions
- This pattern is unconventional and fragile (type checking is best-effort via `literal_eval`) but maximally readable
- Do **not** convert to argparse or a config dataclass without strong justification — the readability benefit is intentional
