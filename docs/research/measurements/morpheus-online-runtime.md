# Morpheus online runtime qualification

> Verdict: **no** on the recorded Apple M3 Pro host.

The coupled one-core sweep did not find a joint configuration that finishes
belief plus root, keeps zero internal-deadline faults, and completes at least
8 simulations on every warm normal move.

## Host

See `morpheus-online-runtime.json` for CPU brand, platform, and scheduler pin
state. Local latency is conditional on this machine. No judge CPU model is
published.

## Sweep

- Widths from Part 04: `64` only
- Sandbox runtimes from Part 00b: qnnpack / fbgemm / x86 (only the loaded
  artifact runtime executes on this host)
- Particles: 8, 32, 64, 128
- Simulation targets: 8, 32
- Proposal batches: 4, 16, 64
- Search depth: 2
- Internal deadlines: 125 ms, 140 ms

Command:

```bash
python scripts/morpheus_measure.py online-runtime \
  --config scripts/configs/morpheus/online-sweep.json \
  --single-core \
  --output docs/research/measurements/morpheus-online-runtime.json
```

## Result

- 48 trials, 0 survivors
- 36 trials rejected because calibrated belief plus root could not fit the
  internal deadline (typical for particle counts 32+)
- Best belief-plus-root-complete trial: 8 particles, proposal batch 4, depth 2,
  140 ms internal deadline — mean completed simulations ≈ 6.7, so it misses the
  fixed minimum of 8 on some warm turns
- Competition gate vs smoke finished (truncated draw at turn 1200)
- Submission harness rejected the extracted bundle for `normal_reply_timeout`
  (judge 150 ms), which matches the Part 09 `no` verdict

## Estimator fields

The written `scripts/configs/morpheus/online-runtime.json` and
`bots/morpheus/deployment.json` carry explicit:

- `p99_estimator_type`: nearest-rank empirical
- `p99_warmup_rule`: max of offline p99 and local samples until the window is full
- `p99_window`, `admission_guard_ms`, and per-component `offline_p99_ms`
- `resident_memory_target_mb` from measured peak RSS with margin below 2 GB

These files are a **best-effort play configuration**, not an accepted
deployment. The belief-limitation note in the JSON states that explicitly.

## Belief limitation

No belief-quality threshold is defined. The measurement keeps recovery and ESS
visible and rejects configs that hide collapse behind a fast reply.

## Next action

Re-run on a Linux judge-like CPU, or reduce selection cost, before treating any
configuration as Part 09 accepted.
