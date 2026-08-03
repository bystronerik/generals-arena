# Part -1: Process substrate

## Deliverable

Define repository placement, retention, test-gate, and fixture rules before any
Morpheus code lands.

**Touches**

- `AGENTS.md`: add `training/morpheus/` and derived training-data placement.
- `pytest.ini`: register and exclude the heavy `morpheus` marker by default.
- `.gitignore`: ignore local Morpheus shards, checkpoints, and materializations.
- `tests/fixtures/`: reserve `tests/fixtures/submission_bots/`.
- `bots/`, `arena/`, `scripts/`, and `data/bot_versions/`: no playing code.

## Prerequisites

None.

## Source specifications

- [Repository workflow](../../AGENTS.md)
- [Morpheus design](../bots/morpheus/index.md)

## Defaults and replacement measurement

This part has no strategy default.

Use `training/morpheus/` for training-only modules and `data/morpheus/` for
local derived training data. Keep the latter gitignored. Modal Volumes remain
the primary storage for remote runs.

Use `tests/fixtures/submission_bots/` for controlled process failures.

## Implementation boundary

Register this marker:

```ini
markers =
    morpheus: heavyweight Morpheus, subprocess, JAX, or model gate
```

Change the default selection to:

```ini
addopts = -m "not replay and not morpheus"
```

Morpheus gates run explicitly with `-m morpheus`. Fast pure tests can remain
unmarked only after a warm-suite measurement proves the 7-second limit.

Add the training package and derived-data rows to the `AGENTS.md` placement
table before Part 00 creates `training/morpheus/tests/`.

## Isolated test

```bash
python -m pytest --collect-only -q
python -m pytest -q
python -m pytest --markers
python -m pytest -m morpheus --collect-only -q
```

Measure the warm default suite and record its elapsed time.

## Specification gaps

The repository currently has no training-only package or training-data
retention rule. This part resolves placement only. It does not select a model,
trainer, or data schema.

## Exit criterion

Answer `yes` if `AGENTS.md`, `.gitignore`, and `pytest.ini` agree; fixture bots
live under `tests/fixtures/`; the default suite excludes heavy Morpheus tests;
the explicit marker is registered and selectable; and the warm default suite
stays below 7 seconds.

Answer `no` for an unregistered marker, unowned path, committed derived shard,
or suite-budget regression.
