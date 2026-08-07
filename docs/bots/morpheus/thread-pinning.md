# Thread pinning

Morpheus pins every math backend to **one thread**, always. This is a
correctness constraint, not a tuning parameter — there is deliberately no knob.

## Where

- [`run.sh`](../../../bots/morpheus/run.sh) exports `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
  `NUMEXPR_NUM_THREADS` **before** `exec`. OpenMP and BLAS size their pools at
  library init, so Python cannot undo an oversubscribed pool later.
- [`agent.py`](../../../bots/morpheus/agent.py) calls `_pin_torch_threads()` at
  the top of `Agent.__init__`, before the session loads. This covers callers
  that construct `Agent` in-process, where the `run.sh` exports never ran.

## Why: search was completely dead without it

The numbers in this section and the tables below come from the one-off
isolation experiment recorded in commit `5b110d0`; they are not republished
under `docs/research/measurements/`. The component forecasts they rest on are
published in
[`morpheus-float32-p99.md`](../../research/measurements/morpheus-float32-p99.md).

Two bot subprocesses on one host each spawn one thread per core by default.
Measured on an 11-core M3 Pro, that oversubscription pushed a 4-leaf batch of
network forwards to **154 ms — more than the entire 140 ms turn budget**.

`can_admit("leaf_batch")` therefore denied **921 of 929** checks, and search
completed **zero simulations on 100% of normal turns**. Every move came from the
shaped root prior plus hard rules; the tree search was inert while still burning
~36 ms/turn on selection that could never finish a simulation.

The lockout was permanent, not intermittent. The warm-up rule is
`forecast = max(offline_p99, max(samples))` until the window fills. The first
move runs under an 8500 ms budget with cold caches, so it contributed a 153.91 ms
sample — above the normal deadline. `leaf_batch` is the only component that could
produce a *new* sample, so the window never filled and the forecast never moved:

```
need=  35.35  rem= 8464.91  ok=True    <- first move, 8500 ms budget
need= 143.13  rem= 8207.96  ok=True    <- cold sample poisons the estimator
need= 153.91  rem=  109.97  ok=False   <- every normal turn thereafter
```

Same seed and opponent, isolating the two factors:

| | threads = default | threads = 1 |
| --- | ---: | ---: |
| shipped runtime | 0.00 sims, **0.0%** search-decided | 5.44 sims, **74.8%** |
| + warm-up escape hatch | 0.00 sims, 0.2% | 7.84 sims, 73.7% |

Pinning alone revives search with no code change. The warm-up escape hatch alone
does essentially nothing — it is a real second-order bug, but not the cause.

## Why: play must match calibration

`training/morpheus/measure_online.py` pins one thread (`pin_single_core`), and so
does every `offline_p99_ms` measurement. An unpinned bot ran on latency numbers
measured for a machine it was never on. Pinning makes the calibrated budget
describe the environment the bot actually plays in.

`test_play_and_calibration_pin_the_same_thread_count` guards the invariant: if
either side stops pinning, the test fails rather than the bot silently
regressing.

## Known consequence: the budget no longer fits

Pinning revives search **and** pushes latency past the judge's limit.
`JUDGE_NORMAL_REPLY_S = 0.150`. Same seed vs `cm_expander`:

| | search-decided | `move_ms` p50 / p99 / max | over 150 ms |
| --- | ---: | ---: | ---: |
| unpinned | 0.0% | 61 / 116 / 170 | 0.11% |
| pinned | **77.1%** | 116 / 168 / 181 | **11.59%** |

So this change is necessary but **not sufficient**. The coupled budget
(`target_simulations = 16`, `pending_leaf_batch = 4`, `normal_deadline_ms = 140`,
`reserve_ms = 10`) was selected for a configuration where search never ran; with
search live it does not fit under 150 ms. Re-deriving it is a Part 09-class
re-qualification, not a constant edit. `deployment.json` already records a
`verdict is no on this host` note.

**Do not read a strength result from the pinned bot until the budget is
re-qualified.** The current state trades a dead search for ~12% of moves at risk
of a judge timeout.

## Related

- [Runtime](runtime.md) — deadline control and admission
- [Prior shaping](prior-shaping.md) — what decides the move when search is inert
- [`morpheus-float32-p99.md`](../../research/measurements/morpheus-float32-p99.md)
  — the component forecasts, measured single-threaded
