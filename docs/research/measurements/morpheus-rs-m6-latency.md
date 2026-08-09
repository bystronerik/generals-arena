# Morpheus-rs M6 — the Rust bot's latency, on M0's schedule

Milestone M6 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md), success criteria 2 and 4. The Python numbers are M0's, re-quoted from [the baseline](morpheus-rs-baseline.md) — same host, same panel, same seeds, same serial schedule, both bots timing themselves with their own clock.

- Bot: `morpheus-rs@5456f5532cc2`
- Host: Apple M3 Pro
- Measured: 2026-08-09T12:49:34+00:00
- Games: 20 · frames: 10994 (10974 normal moves)

## Per-move wall time

| bot | n | p50 | p99 | p99.9 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| morpheus (M0) | 10082 | 133.0 | 164.0 | 177.0 | 272.0 |
| morpheus-rs | 10974 | 105.0 | 135.0 | 140.0 | 158.0 |

Normal moves over the judge's 150 ms limit: **2** (0.02%).
The Python bot's figure on the same schedule was 883 (8.76%).

## Per-component p99

Both columns are **per turn, over the turns the component ran** — which is what `deployment.json`'s `offline_p99_ms` means and what the admission controller forecasts from. Two things it is not. It is not a per-call speedup: the search components run once per simulation and this bot completes more simulations per turn, so a total-versus-total ratio charges it for the extra work it managed to fit; the three whose call count the trace carries are normalized below. And `belief_tensor` is a first-move cost in both bots — both time only `first_move_setup` with it — so its row is twenty values per arm, not ten thousand.

| component | morpheus-rs p99 ms | morpheus (M0) p99 ms | ratio |
| --- | ---: | ---: | ---: |
| particle_transitions | 26.8708 | 142.5778 | 5.3x |
| leaf_batch | 73.3523 | 106.6267 | 1.5x |
| enemy_prior_batch | 52.4654 | 48.8603 | 0.9x |
| root_inference | 4.8327 | 17.5513 | 3.6x |
| belief_proposal | 0.6015 | 4.1316 | 6.9x |
| belief_tensor | 0.1270 | 2.4395 | 19.2x |
| selection | 3.3270 | 17.722 | 5.3x |
| backup | 1.7766 | 5.159 | 2.9x |
| hashing | 0.0001 | 0.0009 | 9.0x |
| reply | 0.0001 | 0.0007 | 7.0x |

### Per call, where the trace counts calls

The Python side divides its per-turn p99 by its *mean* calls per turn, which is the only figure M0 recorded — so its column is an estimate and the Rust column is not.

| component | morpheus-rs p99 ms/call | morpheus (M0) est. ms/call | ratio |
| --- | ---: | ---: | ---: |
| enemy_prior_batch | 12.7707 | 20.5123 | 1.6x |
| leaf_batch | 18.8831 | 32.5876 | 1.7x |
| selection | 0.1282 | 1.2034 | 9.4x |

## Search throughput

Completed simulations per normal move: p50 **16**, p99 16, max 16, mean 15.83.
The Python bot: p50 12, p99 16, mean 12.86.

Belief update *and* root inference both completed on 100.0% of turns, against 18.4% for the Python.

| degradation level | turns |
| --- | ---: |
| average | 6967 |
| policy | 4027 |

## First-move budget

Success criterion 4: load + warmup + init has to fit `first_move_limit_ms` with the same reserve discipline as today.

| stage | p50 ms | p99 ms | max ms |
| --- | ---: | ---: | ---: |
| load | 3.712 | 8.201 | 8.201 |
| warmup | 109.064 | 119.832 | 119.832 |
| init | 0.073 | 0.389 | 0.389 |
| total | 114.81 | 124.659 | 124.659 |

