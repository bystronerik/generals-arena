# Morpheus-rs M0 baseline — the Python oracle, re-measured

Milestone M0 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md). Every latency claim the rewrite is judged against starts from this table.

- Oracle: `morpheus@73967d2125cc`
- Host: Apple M3 Pro
- Captured: 2026-08-08T11:45:32+00:00
- Games: 20 · frames: 10102 (10082 normal moves)

## Per-move wall time

| series | n | p50 | p99 | p99.9 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| all moves | 10102 | 133.0 | 165.0 | 209.0 | 272.0 |
| normal moves | 10082 | 133.0 | 164.0 | 177.0 | 272.0 |

Normal moves over the judge's 150 ms limit: **883** (8.76%).

### Control: the same games, uncaptured

4 game(s) over the head of the same schedule — same opponents, same seeds, same seats — played with capture disarmed. `move_ms` is the agent's own clock in both arms, so the gap is what measuring costs, and it bounds how much of the table above to discount.

| series | n | p50 | p99 | p99.9 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| control (uncaptured) | 1860 | 133.0 | 162.0 | 171.0 | 172.0 |
| captured | 10082 | 133.0 | 164.0 | 177.0 | 272.0 |

Capture overhead at p99: **1.01x** (+2.0 ms). Over-limit moves: 6.72% control vs 8.76% captured.

## Per-component p99 — measured against shipped

`shipped` is the `offline_p99_ms` table in `bots/morpheus/deployment.json`, which the plan quotes only for shape: it was fitted on a host and a configuration whose qualification verdict is *no*.

| component | measured p99 ms | shipped p99 ms | ratio |
| --- | ---: | ---: | ---: |
| particle_transitions | 142.578 | 99.484 | 1.43x |
| leaf_batch | 106.627 | 35.350 | 3.02x |
| enemy_prior_batch | 48.860 | 30.086 | 1.62x |
| root_inference | 17.551 | 11.861 | 1.48x |
| belief_proposal | 4.132 | 3.821 | 1.08x |
| belief_tensor | 2.439 | 2.799 | 0.87x |
| selection | 17.722 | 1.840 | 9.63x |
| backup | 5.159 | 1.785 | 2.89x |
| hashing | 0.001 | 0.001 | 0.90x |
| reply | 0.001 | 0.001 | 0.99x |

### Where the turn goes

Belief update (149.1 ms at p99) plus root inference (17.6 ms) is **166.7 ms** against an internal deadline of 140 ms. Search gets whatever is left, which is nothing at p99 — the deadline is already spent before a single simulation is admitted.

That is the rewrite's premise stated in measured numbers, and it is why `particle_transitions` is the first kernel ported (plan §7), not inference.

## Search throughput

Completed simulations per normal move: p50 **12**, p99 16, max 16, mean 12.86.
Forward equivalents per normal move: p50 17, p99 19, max 20.

Belief update *and* root inference both completed on 18.4% of turns; the belief update was deferred on 81.2%.

| degradation level | turns |
| --- | ---: |
| average | 6295 |
| policy | 3807 |

## Parity corpus

Frames are captured for every turn; the heavy payload (belief particles, root tensor, shaping scores) rides a per-stratum stride. Format: [parity-corpus.md](../../bots/morpheus-rs/parity-corpus.md).

| stratum | frames | heavy |
| --- | ---: | ---: |
| post_contact+recovery | 6738 | 851 |
| pre_contact | 1602 | 208 |
| pre_contact+recovery | 1310 | 172 |
| post_contact | 272 | 37 |
| late_game+recovery | 140 | 20 |
| first_move | 20 | 20 |
| first_contact | 20 | 20 |

RNG draws recorded: 562558 (55.69 per turn) — `choice` 547303, `integers` 8748, `random` 6507.

Unrecorded generator calls: **none** — the whole draw stream is replayable.

## Read alongside

- [one x86 core](morpheus-rs-baseline-modal.md) — the same suite on Linux, which is the shape the competition host has and this one does not
- [CPU-feature probe](morpheus-rs-cpu-probe.md) — the compile target these measurements do not settle
- [parity corpus format](../../bots/morpheus-rs/parity-corpus.md) — what the captured frames contain and what they cannot prove

Every number here is this host's. Morpheus is deadline-driven, so its play is a function of how fast the machine under it happens to be — these are not seeds anyone can replay into the same games.

