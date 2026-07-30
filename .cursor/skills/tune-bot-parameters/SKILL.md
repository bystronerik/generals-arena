---
name: tune-bot-parameters
description: >-
  Applies a parameter revision to bots/<name>/agent.py by editing named
  constants only, then re-runs the baseline seed grid and writes an experiment
  note. Use when retuning a bot after a measurement round, changing thresholds,
  or applying a parameter revision from a strategy spec.
---

# Tune bot parameters

## Model split

- Think model: chooses the new constant values from the round report
- Composer: edits named constants in `agent.py`, re-runs the baseline grid, writes the experiment note

**Composer must not invent a threshold.** When a value is absent from the specification or parameter revision note, Composer stops and asks the think model.

## Inputs

- Parameter revision note (from think model after a measurement round)
- `bots/<bot>/agent.py`
- Round report: [`docs/research/measurements/`](../../../docs/research/measurements/)

## Outputs

- Edited named constants in `agent.py` only
- One experiment note under [`docs/research/experiments/`](../../../docs/research/experiments/)
- One verification match under `--mode competition`

## Rules

- Edit **named constants only**. Do not restructure the decision code.
- Change **one parameter group** per revision.
- Record old value and new value in the experiment note.
- Re-run the same seeds and the same opponents as the baseline arm.
- Revert when the change does not improve the reported metric.
- Use **evaluate-bot-change** or **run-measurement-round** for the re-run grid.

Protocol: [`docs/research/experiment-protocol.md`](../../../docs/research/experiment-protocol.md).

## Changelog

- 2026-07-31 — Initial skill from skills-workflow taxonomy (cause: skills-workflow build)
