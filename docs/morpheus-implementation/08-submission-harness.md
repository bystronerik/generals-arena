# Part 08: Submission-shaped harness

## Deliverable

Add an arena-owned harness that runs an extracted submission bundle under the
judge's process contract.

The harness enforces:

- first reply within 10 seconds;
- each later reply within 150 ms;
- peak process-tree memory below 2 GB;
- pass plus a fault for late, missing, or malformed replies;
- forfeit at 50 faults;
- immediate forfeit on crash or early exit;
- voluntary status-0 exit after stdin closes.

Promotion is blocked until this part passes its own negative controls.

**Touches**

- `bots/`: none.
- `arena/`: add the process harness under `arena/matches/` and bundle support.
- `scripts/`: none. The CLI is `python -m arena.matches.submission`.
- `data/bot_versions/`: none.
- `tests/fixture_bots/`: add controlled failure programs.

## Prerequisites

- [Part 01: Protocol shell](01-protocol-shell.md)

## Source specifications

- [Match constraints](../../RULES.md#08--match-constraints-competition-infrastructure)
- [Evaluation gate](../bots/morpheus/evaluation.md#competition-verification-gate)
- [Competition protocol](../competition/protocol.md)

The harness ownership and scope are decided requirements. They are not open
defaults.

## Defaults and replacement measurement

Use the judge limits above as fixed acceptance values.

The judge does not publish an EOF grace. Use the existing local runner's
3-second wait as the visible local default. Record actual EOF latency and
replace this value when judge documentation provides one.

## Implementation boundary

Keep enforcement in `arena/`, not in a script or bot. Keep judged runs separate
from `arena.matches.run_match`; controlled failure games must never enter
`data/games/` or ratings.

Start each reply clock after the complete observation frame is flushed. Read
one complete line without a blocking `readline`. Validate exactly five
integers. A correctly formatted but game-invalid action is a silent pass and
not a protocol fault.

Monitor the whole process group. Report peak RSS and terminate a group that
crosses the cap. Use an injected smaller cap for the memory fixture.

After game end, close stdin and wait for voluntary exit. Record whether the
harness had to signal or kill the process.

## Isolated test

```bash
python -m pytest tests/test_submission_harness.py \
  -m morpheus -q
```

The table-driven test must accept a good fixture and reject each controlled
case:

- first-reply timeout;
- normal-reply timeout;
- missing reply;
- non-integer and wrong-arity output;
- memory over the injected cap;
- crash;
- EOF hang or nonzero EOF exit.

Only after that command passes, test the shell bundle:

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

```bash
python -m arena.bundle morpheus --force
python -m arena.matches.submission \
  data/bundles/morpheus-<content_hash>.zip \
  --opponent bots/smoke/run.sh \
  --mode competition --seed 0
```

## Specification gaps

The judge CPU model, scheduler, timing start boundary, memory accounting
method, and EOF grace are not published. The harness can enforce the documented
limits, but it cannot prove hardware equivalence.

Peak RSS and hard address-space limits differ between macOS and Linux.
Promotion must use the Linux enforcement path. The macOS path remains useful
for functional negative controls.

## Exit criterion

Answer `yes` only if every controlled failure is rejected for the correct
reason, the good fixture is accepted, no judged fixture writes an arena game,
and the extracted Morpheus shell exits voluntarily.

Answer `no` for any false accept, false reject, leaked process, stored fixture
game, or forced EOF shutdown.
