# Morpheus-rs M0 baseline — one x86 core (Modal)

Milestone M0 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md).
The deployment-authority half: the competition host is one x86 core running Linux, so the arm64 laptop numbers in [the local baseline](morpheus-rs-baseline.md) do not settle latency.

- Oracle: `morpheus@73967d2125cc`
- Container: `@app.function(cpu=1)`, 17 cpus visible, `unknown`
- Games: 6 · frames: 2972 (2966 normal moves) · suite wall 581.8 s

Modal is a proxy for "a one-core x86 Linux container", not the generals.bot sandbox, which cannot be measured directly.

## Per-move wall time

| series | n | p50 | p99 | p99.9 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| all moves | 2972 | 119.0 | 193.0 | 407.0 | 412.0 |
| normal moves | 2966 | 119.0 | 182.0 | 407.0 | 412.0 |
| control, uncaptured | 1141 | 101.0 | 197.0 | 423.0 | 479.0 |

Normal moves over the judge's 150 ms limit: **483** (16.29%); uncaptured control 13.76%.

## Per-component p99 — measured against shipped

`shipped` is the `offline_p99_ms` table in `bots/morpheus/deployment.json`, calibrated on the M3 Pro for a configuration whose qualification verdict is *no*. These are the numbers M7 has to re-derive against.

| component | measured p99 ms | shipped p99 ms | ratio |
| --- | ---: | ---: | ---: |
| particle_transitions | 357.529 | 99.484 | 3.59x |
| leaf_batch | 92.003 | 35.350 | 2.60x |
| enemy_prior_batch | 36.037 | 30.086 | 1.20x |
| root_inference | 26.939 | 11.861 | 2.27x |
| belief_proposal | 7.832 | 3.821 | 2.05x |
| belief_tensor | 5.261 | 2.799 | 1.88x |
| selection | 24.791 | 1.840 | 13.47x |
| backup | 8.349 | 1.785 | 4.68x |
| hashing | 0.002 | 0.001 | 1.60x |
| reply | 0.001 | 0.001 | 1.27x |

## Search throughput

Completed simulations per normal move: p50 **8**, p99 12, max 12, mean 7.72.

Belief update *and* root inference both completed on 11.7% of turns; the belief update was deferred on 86.9%.

