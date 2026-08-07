# Morpheus float32 component p99 re-measurement (Part 17 C0)

Prerequisite for the Part 17 shaping ablation: the `offline_p99_ms` block in
`deployment.json` dated from the retired int8 qualification, so every admission
forecast was stale and the C1/C2 arms would have been compared under wrong
budgets.

Raw data: [`morpheus-float32-p99.json`](morpheus-float32-p99.json).

## Method

Canonical routine, same as the `e4f944c` `belief_proposal` recalibration:
`training.morpheus.measure_online._calibrate_offline_p99`, board side 18, seeds
0 / 100 / 200, `torch.set_num_threads(1)`. Shipped value per component is the
max over the three seeds. Host: Apple M3 Pro, macOS 26.5.2, Python 3.12.11.
`sched_setaffinity` is unavailable on this host, so the pin is thread-count only
— latency here is conditional on this machine, as always.

## Result

| Component | Shipped (int8-era) | Re-measured | Change |
| --- | ---: | ---: | ---: |
| `belief_tensor` | 2.104 | 2.799 | +33% |
| `belief_proposal` | 2.497 | 3.821 | +53% |
| `particle_transitions` | 0.000 | 99.484 | see below |
| `hashing` | 0.000 | 0.001 | — |
| `root_inference` | 5.882 | 11.861 | +102% |
| `leaf_batch` | 21.188 | 35.350 | +67% |
| `enemy_prior_batch` | 19.779 | 30.086 | +52% |
| `backup` | 2.257 | 1.785 | −21% |
| `reply` | 0.000 | 0.001 | — |
| `selection` | 7.308 | 1.840 | −75% |

All values in milliseconds.

The two large under-forecasts are `root_inference` (2x) and `leaf_batch` (1.7x)
— exactly the components that dominate a normal turn, so admission had been
approving work the runtime could not finish.

## `particle_transitions` — measured, not a regression

The seed was `0.0` and the stage measures 74.5 / 99.5 / 87.4 ms across the three
seeds. This is the instrumentation defect already recorded in `e4f944c`:
`runtime.decide` calls `recover_belief` **inside** that stage's timing block, and
uniform enemy proposals collapse the 8-particle set often enough that recovery
dominates the measurement. The number is therefore a real wall cost of that block
but not a steady-state transition cost.

Shipping it was checked rather than assumed. Three arms, side 18, 24 turns,
seeds 0/100/200:

| `offline_p99_ms` block | mean sims | deadline faults | belief+root ok |
| --- | ---: | ---: | ---: |
| shipped (int8-era) | 5.86 | 0 | 0.986 |
| re-measured (all, incl. 99.484) | **6.84** | 0 | **1.000** |
| re-measured, transitions forced to 0 | 6.43 | 0 | 1.000 |

Shipping the honest block is strictly better on both secondaries: the higher
forecasts make the controller decline work it cannot finish instead of starting
it and overrunning. With `admission_guard_ms = 0` and a 140 ms normal deadline,
99.484 ms still admits at the top of the turn, so there is no repeat of the
`ce6643e` lockout. The full block ships.

The underlying `recover_belief`-inside-the-timing-block defect is **not** fixed
here; it stays recorded for a separate change.

## Scope

This re-measures forecasts only. It does not re-run qualification and does not
re-select `p99_window`, `admission_guard_ms`, particle count, or simulation
targets. The `belief_limitation_note` verdict in `deployment.json` is unchanged.
