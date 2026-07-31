---
name: run-classic-grid
description: >-
  Runs the classic-approximate measurement grid through scripts/measure_classic.py,
  stores games under data/classic_games/, and writes
  docs/research/measurements/<round>.{json,md}. Use when tuning remote bots on
  the local classic harness before live queue time.
---

# Run classic grid

## Model split

- Think model: chooses bots, seeds, and proxy metrics to read from the report
- Composer: runs the script and summarizes winrate and decisive games

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

## Environment

```bash
source .venv/bin/activate   # if present; prefer python3.12
pip install -e competition-module
pip install -r requirements.txt
```

## Command

```bash
python scripts/measure_classic.py \
  --bots classic_duel army_convey \
  --seeds 0-2 \
  --swap-sides \
  --round classic-grid-<N>
```

Lower-level tournament (no report):

```bash
python -m arena.classic_tournament \
  bots/classic_duel/run.sh bots/army_convey/run.sh \
  --seeds 0-2 --swap-sides --jobs 2
```

## Outputs

| Path | Contents |
| --- | --- |
| `data/classic_games/*.json` | Classic harness records (gitignored when `*.json` rule applies) |
| `docs/research/measurements/<round>.{json,md}` | Round report (commit this) |

Classic results **never** enter `data/games/` or `data/ratings/`.

## Rules

- Prefer both seat orders (`--swap-sides`) when judging strength.
- Tune thresholds here before opening a remote human block.
- Authority: [`docs/research/strategies/human-95-plan.md`](../../../docs/research/strategies/human-95-plan.md) §4.2.

## Changelog

- 2026-07-31 — Initial skill (repo velocity review S1)
