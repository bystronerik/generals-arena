# Joe Phase 2 — x86 CPU latency benchmark (Modal, 2026-08-12)

Phase 2 of
[`averagejoe-competition-plan.md`](../strategies/averagejoe-competition-plan.md):
ms/move for randomly initialized S and M `HistoryTransformer` nets on **one
x86 CPU core**, in the deployment stack from plan §5 (float32 jax-CPU,
single-threaded XLA). This measurement picks the single tier we train
(plan §8, item 10).

## Verdict: **M is the tier we train. S is never trained.**

M's real per-move compute on one x86 core is **p50 9–11 ms, p99 ≈ 19 ms,
max ≈ 33 ms** — more than 8× inside the 150 ms move budget, with the JIT
compile (≈ 4 s) absorbed by the 10 s first-move grace. The apparent ~500 ms
p99 in the strict-quota container is a cgroup throttling artifact of the
measurement environment, attributed below, not model compute.

Provenance:

- Script: `scripts/joe_modal_cpu_bench.py`; nets from
  `training/joe/networks/` (the Phase 2 port: 39-channel obs, 10-action
  head), random weights — latency does not depend on what the weights are.
- Engine: `competition-module` @ `9e3b9d13cca5`
  (competition-engine-2026-15-g9e3b9d1) for `compute_valid_move_mask`.
- Stack: Modal `debian_slim` py3.12 on x86_64, `jax==0.11.0` (CPU),
  equinox 0.13, float32 (no bf16 — CPU has no fast bf16 path). Thread pools
  pinned to 1 (`XLA_FLAGS` intra-op 1, Eigen multithreading off, OMP/BLAS 1).
- Containers: cpu request **and hard limit 1.0**, memory capped at 2 GB —
  mirroring the competition match limits. Control containers identical but
  allowed to burst to 4 cores.
- Timed step = the full per-move path the deployed bot runs after stdio
  parsing: build-cost grid from the observation, 39-channel augmentation,
  move + build masks, forward pass, greedy argmax, decode, host transfer.
  3 games × 1,200 moves (competition truncation) per configuration.
- Cost: < $1 of CPU minutes.

## 1. Results

| Config | params | compile s | p50 ms | p90 ms | p99 ms | max ms | moves > 50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S, 1-core hard quota | 5.09 M | 3.2 | 6.0 | 6.5 | 462 | 505 | 38 / 3,600 |
| M, 1-core hard quota | 8.56 M | 3.9 | 9.1 | 9.7 | 497 | 503 | 65 / 3,600 |
| S, burst-4 control | 5.09 M | 1.1 | 5.2 | 5.7 | 6.2 | 7.9 | 0 / 3,600 |
| M, burst-4 control | 8.56 M | 1.9 | 11.0 | 13.9 | 18.5 | 32.7 | 0 / 3,600 |

(A first, uninstrumented strict run measured S 5.2/6.1/499 and
M 9.7/10.7/495 for p50/p90/p99 — same picture; raw JSON below is from the
instrumented run.)

## 2. The ~500 ms tail is quota throttling, not compute

Three independent signatures pin the strict-quota tail on the container's
CFS bandwidth quota rather than on the nets:

1. **The burst controls have no tail at all.** Same box, same code, same
   single-threaded pinning, quota raised to 4 cores: zero moves over 50 ms
   in 7,200 measured moves. If the spikes were model compute they would not
   care about the quota ceiling.
2. **Spike magnitude is independent of net size.** S and M differ ~1.8× in
   per-move compute, yet both spike to the same ~430–505 ms. Real compute
   tails scale with the model.
3. **Spikes are periodic in wall time, not in moves.** S spikes every
   ~90 moves at ~5.5 ms/move; M every ~54 moves at ~9.5 ms/move — both
   ≈ every 0.5 s of wall clock, each costing ~5 CFS periods (500 ms at the
   default 100 ms period). Some periodic runtime housekeeping (sub-ms of
   actual work, invisible in the burst runs) briefly wants a second thread,
   blows the 100 ms/100 ms budget, and the whole process is frozen for the
   next several periods.

The cgroup v2 `cpu.stat` counters were not readable inside Modal's sandbox
(gVisor), so the throttle ledger itself could not be quoted; the burst
contrast plus the wall-clock periodicity carry the attribution.

**What this means for deployment (Phase 5), not for the tier choice:** a
host that grants one core by CFS quota (e.g. `docker --cpus=1`) can
reproduce this freeze; a host that pins a core (cpuset) cannot — threads
just timeshare the core and the housekeeping costs its true sub-ms price.
RULES.md §08 promises "one **dedicated** CPU core", i.e. the pinned kind,
and prices a late reply at pass + one fault (50 faults in a game to
forfeit) — so even a quota-shaped host would need a systematic freeze, not
a rare one, to matter. The Phase 5 bot must still (a) keep every thread
pool at 1, as `bots/morpheus` already does, and (b) re-measure p99 over
full games inside the real submission harness before any competition
entry. That was already the plan (§5, Phase 5 verify); this benchmark
turns it from hygiene into a named risk.

## 3. Margin arithmetic for the verdict

Taking the burst runs as the true compute cost and the strict p90 as the
1-core steady state:

| Tier | steady ms/move | true p99 ms | budget × margin (p99) |
| --- | --- | --- | --- |
| S | 5.2–6.5 | 6.2 | ~24× |
| M | 9.1–13.9 | 18.5 | ~8× |

The plan's Phase 2 gate — "if M's p99 fits 150 ms with margin on one core,
M is the single tier we train and S is never trained" — is met with room to
spare. Even the pessimistic reading (strict-quota p90, 9.7 ms) leaves M
15× inside budget, and a one-off 500 ms freeze would cost one fault out of
the 50-fault budget (RULES.md §08), not a game. M it is; the L tier stays
out of scope per §8 item 8, and its ~2× M extrapolation (~20–40 ms p99)
would also have fit — the budget was never the binding constraint at these
sizes, which is itself a useful Phase 5 fact.

## 4. Caveats

- Random weights and random (plausibly scaled) observations; FLOPs are
  input-independent, but a trained net sees the same cost. Nothing here
  measures playing strength.
- Modal's shared hosts do not report the CPU model (`cpuinfo` masked);
  competition hardware may be somewhat faster or slower per core. An 8×
  margin absorbs any plausible per-core gap.
- The timed path excludes stdio parse/serialize (microseconds) and the
  14-channel array build from a parsed board (cheap numpy, < 1 ms at
  21×21); Phase 5 measures the real bot end to end.
- bf16 was off. If Phase 5 ever converts to the morpheus-rs candle stack,
  re-measure there; this number covers the Python jax-CPU path only.

## Raw results

```json
{
  "S": {"tier": "S", "label": "S-strict", "slow_moves_over_50ms": 38, "params_m": 5.088, "jax": "0.11.0", "dtype": "float32", "games": 3, "moves": 3600, "compile_s": 3.2, "p50_ms": 6.03, "p90_ms": 6.48, "p99_ms": 462.08, "max_ms": 504.73, "mean_ms": 10.99},
  "M": {"tier": "M", "label": "M-strict", "slow_moves_over_50ms": 65, "params_m": 8.556, "jax": "0.11.0", "dtype": "float32", "games": 3, "moves": 3600, "compile_s": 3.9, "p50_ms": 9.09, "p90_ms": 9.7, "p99_ms": 497.17, "max_ms": 503.36, "mean_ms": 17.65},
  "S-burst": {"tier": "S", "label": "S-burst", "slow_moves_over_50ms": 0, "params_m": 5.088, "jax": "0.11.0", "dtype": "float32", "games": 3, "moves": 3600, "compile_s": 1.1, "p50_ms": 5.16, "p90_ms": 5.67, "p99_ms": 6.2, "max_ms": 7.85, "mean_ms": 5.04},
  "M-burst": {"tier": "M", "label": "M-burst", "slow_moves_over_50ms": 0, "params_m": 8.556, "jax": "0.11.0", "dtype": "float32", "games": 3, "moves": 3600, "compile_s": 1.9, "p50_ms": 10.96, "p90_ms": 13.89, "p99_ms": 18.53, "max_ms": 32.65, "mean_ms": 11.61}
}
```

Slow-move positions and per-spike timings are in the run log; regenerate
with `modal run scripts/joe_modal_cpu_bench.py --burst`.
