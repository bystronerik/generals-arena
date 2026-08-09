# Morpheus-rs M7 — qualification on one x86 core

Milestone M7 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md). The exit gate is measured here rather than on the laptop: the competition host is one x86 core running Linux, and an arm64 developer machine has no deployment authority over it.

- Container: `@app.function(cpu=1)`, 17 CPUs visible
- CPU: unknown
- Toolchain: rustc 1.97.1 (8bab26f4f 2026-07-14), built in 18.2 s
- Games per config: 20 · `admission_guard_ms` = 0.0

| config | turns | p50 | p99 | p99.9 | max | >150 ms | sims p50 | belief ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `n8-s16-b4-d2` | 10382 | 94 | 131 | 138 | 140 | 0 | 16 | 100.0% |
| `n8-s32-b4-d8` | 10595 | 130 | 140 | 140 | 141 | 0 | 20 | 100.0% |

## Per-component p99, per call

The unit that survives a knob change: the search components run once per simulation, so a per-turn total moves when the simulation count does and says nothing about the kernel.

### `n8-s16-b4-d2`

| component | ms/call p99 | calls/turn | ms/turn p99 |
| --- | ---: | ---: | ---: |
| backup | 0.1259 | 15.99 | 2.0140 |
| belief_proposal | 0.7354 | 1.00 | 0.7354 |
| enemy_prior_batch | 11.2377 | 3.79 | 44.6427 |
| hashing | 0.0002 | 1.00 | 0.0002 |
| leaf_batch | 16.2313 | 4.00 | 64.8862 |
| particle_transitions | 29.4327 | 1.00 | 29.4327 |
| reply | 0.0001 | 1.00 | 0.0001 |
| root_inference | 5.1170 | 1.00 | 5.1170 |
| selection | 0.1524 | 24.84 | 3.8731 |

### `n8-s32-b4-d8`

| component | ms/call p99 | calls/turn | ms/turn p99 |
| --- | ---: | ---: | ---: |
| backup | 0.2267 | 21.61 | 3.6209 |
| belief_proposal | 0.7162 | 1.00 | 0.7162 |
| enemy_prior_batch | 10.3671 | 5.78 | 59.3669 |
| hashing | 0.0002 | 1.00 | 0.0002 |
| leaf_batch | 16.2920 | 5.42 | 106.6250 |
| particle_transitions | 26.9104 | 1.00 | 26.9104 |
| reply | 0.0002 | 1.00 | 0.0002 |
| root_inference | 5.2170 | 1.00 | 5.2170 |
| selection | 0.1879 | 34.91 | 6.2709 |

