# Morpheus Part 14 trainer / checkpoints

> Verdict: **no** — Trainer resume and immutable checkpoints pass the isolated
> tests under research scope. Part 13 remains `no`
> (`narrower_non_promotable_research_scope`), so no promotable main-run candidate
> completed Modal deployment-matched calibration for Part 15.

Generated: 2026-08-05T09:33:53+00:00

## Isolated tests

```bash
python -m pytest \
  training/morpheus/tests/test_trainer_step.py \
  training/morpheus/tests/test_checkpoint_resume.py \
  -m morpheus -q
```

Result: **6 passed**.

Proven:

- tiny deterministic shard overfits (finite loss decreases);
- interrupt → resume produces the same next update as an uninterrupted run;
- checkpoint directory names are content-addressed and never overwritten;
- run manifest event names keep `curriculum_class_advance`,
  `snapshot_saved`, and `arena_checkpoint_accept` distinct;
- resume rejects tensor-schema drift.

## Config freeze

`training/morpheus/configs/promotable-run.json` is the Part 14 entry config.
It sets:

- `promotable_main_run: false`
- `scope: narrower_non_promotable_research_scope`
- `part13_verdict: no`
- Part 12 provisional objective via `pilot-objective.json`
- Part 13 layout `cpu1-seq`, 16 workers/A100, cadence numbers from the
  qualification report
- explicit AdamW, constant LR schedule, batch size, replay window, class
  balance, snapshot cadence, and stopping rule

## Modal entry points

```bash
modal run scripts/morpheus_modal.py::train \
  --config training/morpheus/configs/promotable-run.json
modal run scripts/morpheus_modal.py::resume \
  --config training/morpheus/configs/promotable-run.json --run-id <run_id>
modal run scripts/morpheus_modal.py::inspect_run --run-id <run_id>
modal run scripts/morpheus_modal.py::download \
  --run-id <run_id> --checkpoint <checkpoint_id> --dest /tmp/ckpt
```

Volume root: `morpheus-training` → `/vol/morpheus/runs`, `/vol/morpheus/buffer`.

## Exit criterion mapping

| Criterion | Status |
| --- | --- |
| Interruption/resume deterministic | yes (unit) |
| Checkpoints immutable and self-describing | yes (unit) |
| A100 accounting within Part 13 schedule | planned budget fits; no main Modal train spent here |
| One candidate completes deployment-matched self-play + calibration | no |
| Part 13 prerequisite `yes` | no |

## Next action

Re-open Part 13 with a learning-curve cadence measure. Only after verdict
`yes` set `promotable_main_run: true` and run Modal `train` through
deployment calibration.
