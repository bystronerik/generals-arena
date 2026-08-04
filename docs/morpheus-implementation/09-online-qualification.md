# Part 09: Online qualification

## Deliverable

Select one deployment configuration by measuring the complete Morpheus turn on
one dedicated CPU core and the final submission process shape.

The result fixes the inference runtime, network width, quantization, particle
count, search target, forward limit, batch shapes, reserve, tree bounds, and
first-move setup for one checkpoint family. It also fixes the p99 estimator,
warm-up rule, sample window, admission guard, and resident-memory target.

**Touches**

- `bots/`: add named deployment configuration and passive timing counters.
- `arena/`: use the submission harness and recorded probe path.
- `scripts/`: add the coupled runtime benchmark.
- `data/bot_versions/`: none.

## Prerequisites

- [Part 04: Network and export](04-network-export.md)
- [Part 05: Belief filter](05-belief-filter.md)
- [Part 06: Simultaneous search](06-simultaneous-search.md)
- [Part 07: Runtime controller](07-runtime-controller.md)
- [Part 08: Submission-shaped harness](08-submission-harness.md)

## Source specifications

- [Coupled online compute](../bots/morpheus/open-questions.md#coupled-online-compute)
- [Runtime](../bots/morpheus/runtime.md)
- [Network inference budget](../bots/morpheus/network.md#inference-budget)
- [Evaluation checks](../bots/morpheus/evaluation.md#promotion-decision)

## Defaults and replacement measurement

Start the sweep from the current joint default:

- static 8-bit, 64-channel network;
- 64 particles;
- 32 target simulations;
- at most 113 forward-equivalents;
- 125 ms internal normal deadline.

Measure particle counts 32, 64, and 128 with multiple simulation targets and
candidate sandbox-available runtimes. Width candidates must come from the
capacity and export results in Part 04, not from an invented list.

Replace the default only from complete-turn p50 and p99 results. Zero protocol
faults are required.

Select the admission guard from the zero-fault boundary under measured
scheduler variation. Select the resident target from complete-process peak RSS
while preserving margin below the fixed 2 GB limit.

Sweep bounded windows for Part 07's nearest-rank empirical p99. Reject a window
that forgets a measured slow path or adapts too slowly after a cost change.
Write the selected window and offline component p99 values into the runtime
configuration.

## Implementation boundary

Measure tensor construction, belief proposal batch, particle transitions,
hashing, root inference, leaf and enemy-prior batches, backup, serialization,
and reply delivery together.

Use an idle, pinned core. Record CPU identity and scheduler state. Run both cold
first moves and warm normal moves across all board sizes and turn bands.

Write every selected estimator and safety field into
`scripts/configs/morpheus/online-runtime.json` before the final acceptance run.
Do not leave a runtime field to an implementation default.

Use the instrumented unbundled bot for component metrics. Use the extracted
bundle and Part 08 for external timing, memory, crash, and EOF acceptance.

## Isolated test

```bash
python scripts/morpheus_measure.py online-runtime \
  --config scripts/configs/morpheus/online-sweep.json \
  --single-core \
  --output docs/research/measurements/morpheus-online-runtime.json
```

```bash
python -m arena.matches.submission \
  data/bundles/morpheus-<content_hash>.zip \
  --opponent bots/smoke/run.sh \
  --mode competition --seed 0
```

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

## Specification gaps

No judge CPU model is published, so local latency is conditional on the
recorded machine.

The specs do not define the configuration sweep, acceptable quantization
error, moving p99 estimator, or a belief-quality threshold. This part must
resolve the estimator fields and record the remaining belief limitation. It
must not hide a belief collapse behind a fast reply.

## Exit criterion

Answer `yes` only if one joint configuration has zero faults, first replies
inside both the 8.5-second internal and 10-second judge limits, normal p99
replies inside the selected internal deadline, peak RSS below 2 GB, belief plus
root always complete, and every measured normal nonterminal move after warm-up
completes at least the current minimum search target of 8 simulations.

Answer `no` if belief plus root cannot fit, any measured normal move misses the
minimum search target, an estimator field remains undefined, or the submission
harness rejects the bundle.

After a recorded `no`, follow
[Part 09a: Complete-turn cost reduction](09a-complete-turn-cost.md) before a
re-qualification attempt.
