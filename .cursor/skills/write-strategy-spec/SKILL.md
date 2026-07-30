---
name: write-strategy-spec
description: >-
  Writes a competition bot strategy specification under
  docs/research/strategies/ with named thresholds, pseudocode, a diversity
  claim, and one falsifiable hypothesis. Use when designing a new heuristic bot,
  planning a strategy before any code, or revising a spec after a measurement
  round.
---

# Write strategy spec

## Model split

- Think model: writes the specification, the thresholds, and the hypothesis
- Composer: nothing — do not implement bot code in this step

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

## Inputs

- Bot name and target niche
- [`RULES.md`](../../../RULES.md), [`docs/competition/`](../../../docs/competition/)
- Diversity register: [`docs/research/strategies/diversity-matrix.md`](../../../docs/research/strategies/diversity-matrix.md)
- Latest round report: [`docs/research/measurements/`](../../../docs/research/measurements/)

## Output

- `docs/research/strategies/<bot>.md`
- One link line in [`docs/index.md`](../../../docs/index.md)

Reference example: [`docs/research/strategies/garrison.md`](../../../docs/research/strategies/garrison.md).

## Section template

```text
# <bot> strategy specification
## Goal
## Diversity claim          (how this bot differs from every existing bot)
## Strategy features
## State
## Phases and thresholds     (named constants, one value per line)
## Threat or scoring model
## Candidate move rules
## Pseudocode for act()
## Edge cases
## Expected behavior against existing bots
## Experiment hypothesis     (one falsifiable claim + seed grid + metrics)
```

## Rules

- Give every threshold a **name and a value**. Composer must not invent a number.
- Write one falsifiable hypothesis.
- Name the seed grid and the opponent set.
- Do not write Python. Pseudocode only.
- Run **check-bot-diversity** before implementation.

Authority: [`docs/research/strategies/skills-workflow.md`](../../../docs/research/strategies/skills-workflow.md).

## Changelog

- 2026-07-31 — Initial skill from skills-workflow taxonomy (cause: skills-workflow build)
