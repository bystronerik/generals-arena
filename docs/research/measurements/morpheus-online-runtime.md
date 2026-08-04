# Morpheus online runtime qualification

> Verdict: **no** on the recorded Apple M3 Pro host (Part 09a Phase 6
> re-qualification).

No joint configuration with particle count in {32, 64, 128} finished belief
plus root on every warm move, kept zero internal-deadline faults, and completed
at least 8 simulations on every warm normal move.

## Host

See `morpheus-online-runtime.json` for CPU brand, platform, and scheduler pin
state. Local latency is conditional on this machine. No judge CPU model is
published.

## Phase 6 gate order

| Gate | Result |
| --- | --- |
| 1. 32 particles, 8 simulations, targeted | Fail: belief+root ok rate ≈ 0.73, mean sims ≈ 5.1, min sims = 0, normal p99 ≈ 145 ms |
| 2. Board sizes 18 and 21, turn bands | Fail: same rejection pattern on both sides |
| 3. Scheduler variation, positive admission guard | Fail: no survivor; guard not selected |
| 4. Full 32 / 64 / 128 sweep | Fail: 36 trials, 0 survivors |
| 5. Competition match vs `bots/smoke/run.sh` | Pass: truncated draw at turn 1200 |
| 6. Extracted-bundle submission harness | Fail: `normal_reply_timeout`, 8 faults |

## Sweep

- Config: `scripts/configs/morpheus/online-sweep-phase6.json`
- Widths from Part 04: `64` only
- Sandbox runtimes: qnnpack on this host (fbgemm / x86 not loaded)
- Particles: 32, 64, 128 (8-particle configs excluded)
- Simulation targets: 8, 32
- Proposal batches: 4, 16, 64
- Search depth: 2
- Internal deadlines: 125 ms, 140 ms
- Admission guard candidates: 5 ms, 10 ms (zero guard excluded)

Command:

```bash
python scripts/morpheus_measure.py online-runtime \
  --config scripts/configs/morpheus/online-sweep-phase6.json \
  --single-core \
  --dry-run \
  --output docs/research/measurements/morpheus-online-runtime.json
```

## Result

- 36 trials, 0 survivors
- Every trial missed belief-plus-root completeness and the 8-simulation minimum
- Best measured trial: 32 particles, target 32 sims, proposal batch 4, 140 ms
  deadline — belief+root ok rate ≈ 0.73, mean sims ≈ 6.05, min sims = 0,
  normal p99 ≈ 153 ms
- Calibrated belief+root forecast for 32 particles ≈ 48 ms, but timed warm
  moves still miss belief+root and the simulation floor
- Competition gate vs smoke finished (truncated draw at turn 1200)
- Submission harness rejected `data/bundles/morpheus-4ea1859eb4d1.zip` for
  `normal_reply_timeout` (judge 150 ms)

## Estimator fields

No accepted deployment was written. Phase 6 used `--dry-run` so the prior
best-effort `online-runtime.json` / `deployment.json` play files were not
replaced. Those files remain a local play configuration, not an accepted
deployment.

## Belief limitation

No belief-quality threshold is defined. The measurement keeps recovery and ESS
visible and rejects configs that hide collapse behind a fast reply.

## Next action

Continue cost reduction under
[Part 09a: Complete-turn cost reduction](../../morpheus-implementation/09a-complete-turn-cost.md).
Belief plus root must complete on every warm move before a new Part 09
re-qualification.
