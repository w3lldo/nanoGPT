---
mode: agent
description: Bootstrap a full Claude Code-style project intelligence structure in ANY repository using GitHub Copilot with Claude models. Language and framework agnostic. Includes step-by-step reasoning AND ready-to-fill file templates.
---

# Skill: Bootstrap Claude Code Project Structure (General)

Use this prompt on **any repository** to give Claude persistent memory, navigable context, reusable skills, task recipes, architectural decisions, and runbooks — without needing the Claude Code CLI.

**How to use:** Attach this file in Copilot Chat and say:
> "Set up the Claude Code project structure for this repository."

---

## What This Creates

```
<repo-root>/
├── CLAUDE.md                            ← Root memory — Claude reads this every session
├── docs/
│   ├── architecture.md                  ← Full architecture + data flows
│   ├── decisions/
│   │   ├── README.md                    ← ADR index
│   │   ├── 001-<first-decision>.md
│   │   └── 00N-<nth-decision>.md
│   └── runbooks/
│       ├── README.md                    ← Runbook index
│       ├── local-dev-setup.md
│       ├── deploying-a-release.md
│       └── <infra-component>-ops.md     ← One per major infrastructure component
├── .claude/
│   ├── settings.json
│   ├── hooks/
│   │   ├── pre-edit.md                  ← What Claude checks before editing
│   │   └── post-edit.md                 ← Verification checklist after editing
│   └── skills/
│       ├── code-review/SKILL.md         ← Project-specific PR review checklist
│       ├── refactor/SKILL.md            ← Safe refactor patterns + what NOT to change
│       └── release/SKILL.md             ← Version bump → changelog → tag → deploy
├── tools/
│   ├── prompts/
│   │   ├── add-<input-type>.md          ← e.g. add-kafka-consumer.md, add-webhook.md
│   │   ├── add-<output-type>.md         ← e.g. add-publisher.md, add-db-query.md
│   │   ├── add-api-endpoint.md
│   │   ├── add-domain-model.md
│   │   └── write-tests.md
│   └── scripts/
│       └── README.md
└── src/                                 ← Mirror the major source modules
    ├── <module-1>/CLAUDE.md
    ├── <module-2>/CLAUDE.md
    └── <module-N>/CLAUDE.md
```

---

## Phase 1 — Discover Before You Write

> **Do not create any files yet.** Read the project first.

Read in this order:
1. The main entrypoint (`main.rs`, `main.py`, `index.ts`, `Program.cs`, `app.py`, etc.)
2. The dependency manifest (`Cargo.toml`, `package.json`, `pyproject.toml`, `pom.xml`, `*.csproj`, `go.mod`, etc.)
3. Any existing `README.md`, `CHANGELOG.md`, or `docs/`
4. The top-level `src/` (or equivalent) directory — one level deep
5. Any CI config (`.github/workflows/`, `Makefile`, etc.) for build/test/deploy commands
6. Any lint/formatter config (`.eslintrc`, `clippy.toml`, `.flake8`, etc.) for enforced rules

Answer these before writing anything:

| Question | Answer |
|----------|--------|
| What does this service do? (1–3 sentences) | |
| What is the stack? (language, framework, key libraries) | |
| What are the data flows? (inputs → processing → outputs) | |
| What are the main modules / layers? | |
| What env vars / config does it need? | |
| What rules are already enforced? (linting, error handling, naming) | |
| What are the common dev commands? (build, test, lint, run) | |
| What is the release history? (CHANGELOG or git tags) | |

---

## Phase 2 — Create `CLAUDE.md` (Root)

> The single most important file. A **navigation hub** — short summaries with links, not walls of text.
> Claude reads this on every session to orient itself.

```markdown
# <ProjectName> — Project Intelligence

> Root memory file. Claude reads this first every session.
> Sub-modules have their own CLAUDE.md files linked below.
> Last updated: YYYY-MM-DD

---

## What This Service Does

<!-- 2–4 sentences: what problem it solves, who calls it, what it produces -->

---

## Navigation — Where to Go for What

| I want to understand… | Read this |
|-----------------------|-----------|
| Full architecture & data flows | [`docs/architecture.md`](docs/architecture.md) |
| Why key decisions were made | [`docs/decisions/`](docs/decisions/) |
| Ops procedures | [`docs/runbooks/`](docs/runbooks/) |
| <Module 1> layer | [`src/<module-1>/CLAUDE.md`](src/<module-1>/CLAUDE.md) |
| <Module 2> layer | [`src/<module-2>/CLAUDE.md`](src/<module-2>/CLAUDE.md) |
| Coding skills / task recipes | [`.claude/skills/`](.claude/skills/) |
| Agent settings | [`.claude/settings.json`](.claude/settings.json) |

---

## Project Identity

| Property | Value |
|----------|-------|
| Name | `<project-name>` |
| Version | `<x.y.z>` |
| Language / Runtime | `<language> <version>` |
| Framework | `<framework>` |
| Ticket prefix | `<JIRA-prefix>-*` |
| Staging URL | `<url>` |
| Production URL | `<url>` |

---

## Architecture (3-Sentence Summary)

<!-- Sentence 1: what the service is and does -->
<!-- Sentence 2: how data flows through it -->
<!-- Sentence 3: where business logic lives vs transport/plumbing -->

---

## Key Inputs / Outputs

| Name | Direction | Type | Description |
|------|-----------|------|-------------|
| `<topic/endpoint/queue>` | Consume/Publish/Request/Response | Kafka/HTTP/DB/Queue | Description |

---

## Source Layout

```
src/
├── <entrypoint>      # Wiring only — spawns tasks/servers, no logic
├── <config>          # Configuration / env var parsing
├── <module-1>/       # <one-line description> — see src/<module-1>/CLAUDE.md
└── <module-N>/       # <one-line description> — see src/<module-N>/CLAUDE.md
```

---

## Non-Negotiable Rules

<!-- Enforced by linting / CI — violations fail the build -->

1. **<Rule 1>** — `<forbidden pattern>` is forbidden. Use `<correct pattern>` instead.
2. **<Rule 2>** — explanation + correct pattern.
3. **<Rule N>** — explanation + correct pattern.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `<VAR_NAME>` | Yes/No | `<value>` | Description |

---

## Common Commands

```bash
# Run locally
<command>
# Run tests
<command>
# Lint
<command>
# Format
<command>
# Build for production
<command>
```

---

## Known TODOs

| Location | Description |
|----------|-------------|
| `<file>` | `<todo description>` |

---

## Recent Changes

| Version | Date | Summary |
|---------|------|---------|
| `x.y.z` | YYYY-MM-DD | Description |
```

---

## Phase 3 — Create `docs/architecture.md`

> Cover the system deeply enough that someone new can understand every data flow
> without having to read source code.

```markdown
# Architecture — <ProjectName>

> Last updated: YYYY-MM-DD

## Overview

<!-- 1 paragraph: service type (event-driven / request-response / batch),
     production data path, what it depends on -->

## High-Level Diagram

```
┌──────────────────────────────────────────────────┐
│                  <ServiceName>                    │
│                                                   │
│  ┌──────────┐   ┌───────────────┐                │
│  │ Inbound  │──►│   Domain /    │──► Outbound    │
│  │ Adapter  │   │   Processing  │    Adapters    │
│  └──────────┘   └───────────────┘                │
└──────────────────────────────────────────────────┘
```

## Architectural Pattern

<!-- e.g. Hexagonal / Layered / CQRS / Event-driven.
     State the rules: what layer can call what, what is forbidden. -->

## Data Flow — <Flow 1 Name>

<!-- Step-by-step, tied to actual function/file names -->

## Data Flow — <Flow 2 Name>

## State & Singleton Lifecycle

<!-- How expensive resources (DB connections, HTTP clients, message producers)
     are initialised and shared across requests/messages -->

## Infrastructure Detail

### <Component 1 — e.g. Database>
<!-- Schema, access patterns, partitioning, TTL, connection config -->

### <Component 2 — e.g. Message Broker>
<!-- Topics/queues, consumer groups, publisher config, delivery guarantees -->

## Observability

<!-- Metric names + labels, log format, tracing, health endpoints -->

## Deployment

<!-- Container image, env config source, health probe paths, release profile -->
```

---

## Phase 4 — Create ADRs in `docs/decisions/`

> One file per major technical decision. Write from the perspective of the person
> who made the decision, not a retrospective justification.

### `docs/decisions/README.md`

```markdown
# ADR Index

| # | Title | Status | Date |
|---|-------|--------|------|
| [001](001-<slug>.md) | <Title> | Accepted | YYYY-MM-DD |
```

### `docs/decisions/00N-<slug>.md` (template)

```markdown
# ADR-00N: <Title>

- **Status:** Accepted / Superseded / Deprecated
- **Date:** YYYY-MM-DD
- **Ticket:** <TICKET-NNN> (if applicable)

## Context

<!-- Why did this decision need to be made? What forces / constraints existed? -->

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Option A | | |
| Option B | | |

## Decision

<!-- What was chosen and the single most important reason why -->

## Consequences

<!-- Positive outcomes, negative trade-offs, constraints this imposes on future work -->
```

**Write ADRs for:**
- Language / runtime choice
- Database / persistence technology
- Architecture pattern (hexagonal, layered, etc.)
- Message broker / queue technology
- Any major infrastructure component
- Any breaking change that affected downstream consumers

---

## Phase 5 — Create Runbooks in `docs/runbooks/`

> Every runbook must be self-contained — commands must work copy-paste without extra context.

### `docs/runbooks/README.md`

```markdown
# Runbook Index

| Runbook | When to use |
|---------|-------------|
| [local-dev-setup.md](local-dev-setup.md) | First-time local environment setup |
| [deploying-a-release.md](deploying-a-release.md) | Step-by-step release process |
| [<component>-ops.md](<component>-ops.md) | <Component> operations and troubleshooting |
```

### `docs/runbooks/local-dev-setup.md` (template)

```markdown
# Runbook: Local Dev Setup

## Prerequisites
<!-- Per OS: list every tool with install commands -->

## Running Locally
<!-- Exact commands. What env vars to set. What defaults are assumed. -->

## Connecting to Staging / Dev Environment
<!-- How to override env vars to point at remote infrastructure -->

## Useful Commands
<!-- Build, test, lint, format — copy-paste ready -->

## Verifying the Service is Up
<!-- curl commands to health endpoints with expected responses -->
```

### `docs/runbooks/deploying-a-release.md` (template)

```markdown
# Runbook: Deploying a Release

## Pre-flight Checklist
- [ ] All CI checks pass
- [ ] CHANGELOG.md updated
- [ ] Version bumped in manifest
- [ ] Breaking changes communicated to downstream teams

## Versioning Convention

| Change type | Bump |
|-------------|------|
| Bug fix, non-breaking internal change | Patch (`x.y.Z`) |
| New feature, non-breaking | Minor (`x.Y.0`) |
| Breaking change (contract / schema change) | **Major** (`X.0.0`) |

## Release Steps
1. Bump version in `<manifest file>`
2. Update `CHANGELOG.md` — add section at top
3. Verify: `<build>`, `<test>`, `<lint>`
4. Commit: `git commit -m "chore: release vx.y.z"`
5. Tag: `git tag -a vx.y.z -m "Release vx.y.z"`
6. Push: `git push origin <branch> --tags`
7. CI/CD: `<what the pipeline does on tag push>`

## Rollback
<!-- How to revert; any data migration concerns -->

## Post-Release Verification
- [ ] Health probes respond OK
- [ ] Primary metric is incrementing
- [ ] No error spike in logs for first 15 minutes
```

### `docs/runbooks/<component>-ops.md` (template — one per infra component)

```markdown
# Runbook: <Component> Operations

## Common Queries / Commands
<!-- Copy-paste ready commands for the most frequent tasks -->

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|

## Schema / Config Changes
<!-- How to apply changes safely across environments -->
```

---

## Phase 6 — Create `.claude/settings.json`

```json
{
  "_comment": "Claude Code / Copilot-with-Claude agent settings for <ProjectName>",

  "project": {
    "name": "<ProjectName>",
    "language": "<language>",
    "version": "<x.y.z>",
    "rootMemoryFile": "CLAUDE.md"
  },

  "context": {
    "alwaysRead": [
      "CLAUDE.md",
      "src/<module-1>/CLAUDE.md",
      "src/<module-2>/CLAUDE.md"
    ],
    "referenceOnDemand": [
      "docs/architecture.md",
      "docs/decisions/",
      "docs/runbooks/",
      ".github/instructions/"
    ],
    "ignore": [
      "target/", "node_modules/", "dist/", "build/",
      "__pycache__/", "*.lock", ".idea/", ".vscode/"
    ]
  },

  "rules": {
    "forbid": [
      "<forbidden pattern — e.g. .unwrap(), console.log, eval()>"
    ],
    "require": [
      "<required convention — e.g. structured logging via tracing/winston/structlog>",
      "<required convention — e.g. Result<T,E> for error handling>"
    ],
    "codeStyle": {
      "errorHandling": "<pattern>",
      "logging": "<library + functions to use>",
      "asyncRuntime": "<tokio / asyncio / Node event loop / etc.>"
    }
  },

  "skills": {
    "directory": ".claude/skills/",
    "available": [
      "code-review/SKILL.md",
      "refactor/SKILL.md",
      "release/SKILL.md"
    ]
  },

  "hooks": {
    "directory": ".claude/hooks/",
    "beforeEdit": "hooks/pre-edit.md",
    "afterEdit": "hooks/post-edit.md"
  },

  "tools": {
    "scripts": "tools/scripts/",
    "prompts": "tools/prompts/"
  }
}
```

---

## Phase 7 — Create `.claude/hooks/`

### `pre-edit.md` (template)

```markdown
# Pre-Edit Hook

> Read this before making any code changes.

## Checklist

1. **Identify the module** being edited → read its `src/<module>/CLAUDE.md`
2. **Re-read the Non-Negotiable Rules** in root `CLAUDE.md`
3. **Search for existing patterns** before writing new ones — the codebase is intentionally consistent
4. **Pick the right task prompt** from `tools/prompts/` for this type of change

## Module → CLAUDE.md Map

| Editing files in… | Read this first |
|-------------------|-----------------|
| `src/<module-1>/` | [`src/<module-1>/CLAUDE.md`](../../src/<module-1>/CLAUDE.md) |
| `src/<module-2>/` | [`src/<module-2>/CLAUDE.md`](../../src/<module-2>/CLAUDE.md) |
```

### `post-edit.md` (template)

```markdown
# Post-Edit Hook

> Run these checks after every code change.

## Verification Checklist

- [ ] **Build:** `<build command>` — fix all errors
- [ ] **Lint:** `<lint command>` — resolve all warnings
- [ ] **Format:** `<format command>`
- [ ] **Tests:** `<test command>` — all pass; new logic has new tests
- [ ] **Docs updated?**
  - New endpoint → update `src/<api-module>/CLAUDE.md` route table
  - New env var → update `CLAUDE.md` env var table
  - New input/output → update `CLAUDE.md` + `docs/architecture.md`
  - Breaking change → update `CHANGELOG.md` + create a new ADR
```

---

## Phase 8 — Create `.claude/skills/`

### `code-review/SKILL.md` (template)

```markdown
# SKILL: Code Review

> Use when asked to review a PR, diff, or code block in this repository.

## Review Checklist

### Safety & Correctness
- [ ] <Language-specific: e.g. no .unwrap(), no null dereference, no panic>
- [ ] <e.g. no blocking calls on async executor>
- [ ] All errors handled — no silent discards
- [ ] No hardcoded secrets or credentials

### Architecture
- [ ] Business logic only in `<domain/core/service>` layer
- [ ] Layer boundaries respected: <list the specific rules for this project>
- [ ] No circular imports between modules

### Models & Serialisation
- [ ] Naming conventions followed
- [ ] Optional / nullable fields handled correctly
- [ ] New API response types included in OpenAPI / schema

### Infrastructure
- [ ] New input sources wired into the main consumer/handler loop
- [ ] New output destinations use the established publish/write helper pattern

### Observability
- [ ] Structured logging used — no `<print/console.log/println>`
- [ ] Errors logged with context (IDs, relevant state)
- [ ] New processing paths emit metrics

### Tests
- [ ] New logic has unit tests
- [ ] Async code tested with the async test runner
- [ ] Serialisation tested in both directions

### Documentation
- [ ] Complex logic has inline *why* comments
- [ ] New endpoints/topics/queues added to `CLAUDE.md` + `docs/architecture.md`
- [ ] Breaking changes in `CHANGELOG.md`

## Common Issues

| Pattern seen | Problem | Correct approach |
|-------------|---------|-----------------|
| `<bad pattern 1>` | `<why it's wrong>` | `<correct pattern>` |
| `<bad pattern 2>` | `<why it's wrong>` | `<correct pattern>` |
| `<bad pattern 3>` | `<why it's wrong>` | `<correct pattern>` |
| `<bad pattern 4>` | `<why it's wrong>` | `<correct pattern>` |
| `<bad pattern 5>` | `<why it's wrong>` | `<correct pattern>` |
<!-- Fill in with real patterns from this codebase — this is the highest-value section -->
```

### `refactor/SKILL.md` (template)

```markdown
# SKILL: Refactor

> Use when asked to refactor, restructure, or improve existing code.

## Principles

1. **Preserve behaviour first** — run tests before and after; no behaviour changes in a refactor PR
2. **Respect layer boundaries** — do not move logic between layers
3. **One concern at a time** — structural changes separate from behavioural changes
4. **Follow existing patterns** — refactor toward patterns already in use, not new ones

## Common Refactor Patterns for This Codebase

<!-- Fill in 3–5 patterns specific to this project. Examples:
- Extract long functions into named sub-steps
- Replace nested conditionals with early returns
- Consolidate duplicate error-mapping into a shared helper
- Replace manual null/Option chains with .and_then() / flatMap() / ?. -->

## Refactor Safety Checklist
- [ ] `<test command>` passes before starting
- [ ] `<test command>` passes after finishing
- [ ] `<lint command>` — no new warnings
- [ ] No forbidden patterns introduced
- [ ] Layer boundaries still respected

## What NOT to Refactor

> These look improvable but must not be changed.

| Thing | Why it must stay |
|-------|-----------------|
| `<identifier or pattern 1>` | `<reason — e.g. downstream services depend on this format>` |
| `<identifier or pattern 2>` | `<reason — e.g. changing this orphans DB rows>` |
| `<identifier or pattern 3>` | `<reason — e.g. init order is load-bearing for thread safety>` |
<!-- This is the most valuable part — fill in real examples from the codebase -->
```

### `release/SKILL.md` (template)

```markdown
# SKILL: Release

> Use when preparing or executing a release.

## When to Release
<!-- Describe the release cadence / trigger for this project -->

## Version Bump Decision

| Change type | Bump |
|-------------|------|
| Bug fix, performance, non-breaking internal change | Patch (`x.y.Z`) |
| New feature, new endpoint/topic (non-breaking) | Minor (`x.Y.0`) |
| Breaking change (removed/renamed field, changed contract) | **Major** (`X.0.0`) |

## Step-by-Step

1. Bump version in `<manifest file>`
2. Update `CHANGELOG.md` — add new section at top
3. Verify: `<build>`, `<test>`, `<lint>` all pass
4. Commit: `git commit -m "chore: release vx.y.z"`
5. Tag: `git tag -a vx.y.z -m "Release vx.y.z"`
6. Push: `git push origin <branch> --tags`
7. CI/CD: `<what the pipeline does on tag push>`

## Breaking Change Protocol

1. Notify owners of all downstream consumers **before** merging
2. Consider a parallel period where old and new interfaces both work
3. Create a new ADR in `docs/decisions/`
4. Update `CLAUDE.md` and `docs/architecture.md`

## Post-Release Verification
- [ ] Health probes respond OK
- [ ] Primary metric incrementing normally
- [ ] No error spike in logs for first 15 minutes
- [ ] Docs URL shows updated version
```

---

## Phase 9 — Create `tools/prompts/`

> One `.md` file per common development task. Each file:
> 1. States when to use it
> 2. Lists which files to edit
> 3. Provides a step-by-step guide with language-appropriate code templates
> 4. Ends with a Rules section

### `add-domain-model.md` (template)

```markdown
# Prompt: Add Domain Model

## Where Does It Live?
<!-- Convention for this project: e.g. src/domain/models/, src/types/ -->

## Struct / Class Template
\`\`\`<language>
// Following project naming + serialisation conventions
\`\`\`

## Enum Template
\`\`\`<language>
// Integer-backed or string-backed depending on external source
\`\`\`

## Register the Model
<!-- mod.rs / index.ts barrel / __init__.py / etc. -->

## Serialisation Tests
\`\`\`<language>
// Deserialise from external format — assert fields
// Serialise to expected format — assert output
\`\`\`

## Rules
- Follow `<naming convention>` for JSON field names
- Use `<timestamp type>` for timestamps
- Use `<money type>` for monetary values
```

### `write-tests.md` (template)

```markdown
# Prompt: Write Tests

## Unit Test
\`\`\`<language>
// test_<unit>_<scenario>_<expected_outcome>
\`\`\`

## Async Test
\`\`\`<language>
// async test with the project's test runner / decorator
\`\`\`

## Serialisation Test
\`\`\`<language>
// Deserialise from JSON — assert fields
// Serialise to JSON — assert format
\`\`\`

## Where Tests Live
<!-- Same file / tests/ / __tests__/ / spec/ -->

## Naming Convention
`test_<unit>_<scenario>_<expected_outcome>`

## Rules
- Never use `<forbidden pattern>` in test code — use `<alternative>`
- All async tests use `<async test decorator/attribute>`
- Test both serialisation directions for every new model
```

### Additional prompts — one per task type in this repo

For each other common task (new input source, new output destination, new endpoint, etc.), create a `tools/prompts/add-<task>.md` following this skeleton:

```markdown
# Prompt: <Task Title>

## When to Use
<!-- "When adding a new X to the codebase" -->

## Files to Edit
| File | Change |
|------|--------|

## Steps
1. <Step with code template>
2. <Step with code template>

## Rules
- Rule 1
- Rule 2
```

---

## Phase 10 — Create Sub-Module `CLAUDE.md` Files

> One per major module in `src/` (or equivalent).
> More detailed than the root, scoped to one module's concerns.

```markdown
# src/<module> — <Purpose>

> Parent: [CLAUDE.md](../../CLAUDE.md)

---

## Purpose

<!-- What this module does.
     Equally important: what it does NOT do.
     Be explicit: "no business logic here", "no DB calls here", etc. -->

---

## Files

| File | Role |
|------|------|
| `<file1>` | `<one-line description>` |
| `<file2>` | `<one-line description>` |

---

## Key Functions / Types

<!-- The 3–7 most important things in this module:
     name, signature/type, one-line description -->

---

## Data Flow

<!-- How data enters and exits this module.
     ASCII diagram if there are multiple paths. -->

---

## Rules

<!-- Module-specific rules. Be concrete.
     e.g.:
     - "All DB queries go through the DAL functions — never call the session directly"
     - "This module never imports from <other-module>"
     - "Error responses must always include the resource ID" -->
```

---

## Phase 11 — Wire into GitHub Copilot

Update `.github/copilot-instructions.md` so Copilot auto-loads context every session:

```markdown
# GitHub Copilot Instructions — <ProjectName>

This file is automatically read by Copilot as repository-level context every session.

---

## Extended Documentation

| File | Contents |
|------|----------|
| [`CLAUDE.md`](../CLAUDE.md) | Root project memory — start here |
| [`docs/architecture.md`](../docs/architecture.md) | Full architecture & data flows |
| [`docs/decisions/`](../docs/decisions/) | Architectural Decision Records |
| [`docs/runbooks/`](../docs/runbooks/) | Ops runbooks |

---

## Skill Prompts (attach with `#` in Copilot Chat)

| Prompt | When to use |
|--------|-------------|
| [`setup-claude-code-project-general.prompt.md`](prompts/setup-claude-code-project-general.prompt.md) | Bootstrap this structure in a new repo |
| [`.claude/skills/code-review/SKILL.md`](../.claude/skills/code-review/SKILL.md) | Review a PR or diff |
| [`.claude/skills/refactor/SKILL.md`](../.claude/skills/refactor/SKILL.md) | Safe refactor |
| [`.claude/skills/release/SKILL.md`](../.claude/skills/release/SKILL.md) | Prepare a release |
| [`tools/prompts/add-domain-model.md`](../tools/prompts/add-domain-model.md) | Add a new model |
| [`tools/prompts/write-tests.md`](../tools/prompts/write-tests.md) | Write tests |

---

## Non-Negotiable Rules

<!-- Copy the Non-Negotiable Rules from CLAUDE.md here verbatim
     so Copilot sees them without needing to open CLAUDE.md -->
```

---

## Quality Checklist

Before finishing, verify every item:

- [ ] `CLAUDE.md` root links to **every** sub-CLAUDE.md, every docs section, and `.claude/skills/`
- [ ] Every sub-CLAUDE.md has a `> Parent: [CLAUDE.md](...)` back-link to the root
- [ ] `.claude/settings.json` → `alwaysRead` lists all sub-CLAUDE.md files
- [ ] Non-negotiable rules appear in **both** `CLAUDE.md` AND `.claude/settings.json → rules`
- [ ] `code-review/SKILL.md` → Common Issues table has ≥ 5 real patterns from this codebase
- [ ] `refactor/SKILL.md` → "What NOT to Refactor" table has ≥ 3 real entries with specific reasons
- [ ] Every runbook is self-contained — commands work copy-paste without extra context
- [ ] ADRs exist for every major technical decision visible in the codebase
- [ ] `tools/prompts/` has one file per common task type in this repo
- [ ] `.github/copilot-instructions.md` lists all skill prompt files and links to `CLAUDE.md`

---

## How to Use This Structure Day-to-Day

| Goal | Action |
|------|--------|
| General coding help | Just ask — Copilot auto-reads `.github/copilot-instructions.md` |
| Deep context on a module | Attach `src/<module>/CLAUDE.md` with `#` |
| Review a PR | Attach `.claude/skills/code-review/SKILL.md` + paste the diff |
| Safe refactor | Attach `.claude/skills/refactor/SKILL.md` + the file |
| Prepare a release | Attach `.claude/skills/release/SKILL.md` |
| Add a new feature | Attach the relevant `tools/prompts/<task>.md` |
| Architecture question | Attach `docs/architecture.md` |
| "Why was X decided?" | Attach `docs/decisions/00N-*.md` |
| Bootstrap a new repo | Attach this file and say "Set up the Claude Code project structure" |

---

## Reference Implementation

| Repository | Stack | Notes |
|-----------|-------|-------|
| `SS.RMS.BetMapper` | Rust · Kafka · ScyllaDB | First project this was applied to — fully worked example of every file above |