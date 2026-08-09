# M6's strength result does not replicate

> **Retraction.** [The M6 strength report](morpheus-rs-m6-strength.md) published
> `morpheus@73967d2125cc → morpheus-rs@5456f5532cc2` at **+425.06 ± 22.77 Elo,
> P(B > A) = 1.0000**, from 4,032 games. That number is not reproducible. Four
> later measurements of the same two programs — including one that replays M6's
> own map seeds under M6's own job configuration — agree with each other and
> disagree with M6 by **7.4 sigma**. The current best estimate of the same
> contrast is **+138 Elo**, from the M7 round. **The cause of M6's result has
> not been identified.**

Found on 2026-08-10 while reading the M7 knob A/B, which put the two knob
settings 18 Elo apart and, incidentally, put the two *bots* far closer together
than M6 had.

## What was measured

All five rows are the same head-to-head pair, `morpheus` against
`morpheus-rs`, with seats alternated. `morpheus@73967d2125cc` is byte-identical
throughout. `morpheus-rs` differs between M6 and the rest only by the M7
telemetry commit, which adds fields inside `Trace` — inactive unless
`MORPHEUS_RS_TRACE` is set, and it was set in none of these runs — plus one
string built at startup. Verified by diffing the two commits.

| condition | W–L–D | morpheus-rs score | n |
| --- | ---: | ---: | ---: |
| **M6 round**, 11 jobs, other applications running | 163–22–7 | **0.867 ± 0.024** | 192 |
| M7 round, 11 jobs, quiet machine | 51–44–1 | 0.536 ± 0.051 | 96 |
| serial, idle, fresh seeds | 5–4–1 | 0.550 ± 0.157 | 10 |
| probe, 11 jobs, quiet, **M6's own seeds** | 10–12–0 | 0.455 ± 0.106 | 22 |
| probe, 11 jobs, **6 CPU spinners**, M6's own seeds | 9–13–0 | 0.409 ± 0.105 | 22 |
| **everything except M6, pooled** | 70–69–1 | **0.504 ± 0.042** | 140 |

M6 against the pooled remainder: **+0.364 in score, 7.4 sigma**.

## What has been ruled out, by experiment rather than by argument

- **Map seeds.** The probe replays eleven seeds `morpheus-rs` *won* in M6, both
  orientations. M6 scored 0.773 on that subset; the replay scored 0.455. Same
  maps, same seats.
- **Tournament parallelism.** The probe uses the same `--jobs 11` on the same
  11-core host, with both processes in every game being the heavy bots — the
  most contended arrangement the runner can produce. 0.455.
- **External CPU load.** Six spin loops alongside the same probe moved the
  result to 0.409 — *away* from M6, not toward it. Whatever M6 was, it was not
  a busy machine in this sense.
- **The programs.** Identical content hashes; the Rust diff is telemetry that
  never executed. No trace file exists in either round's time window.
- **The panel.** Ten panel-only pairings — games neither morpheus played —
  are statistically identical across the two rounds (mean z = −0.38, nothing
  above 1.7σ). Whatever moved did not move the heuristics.

## The one asymmetry that is real

Across the five panel opponents, between M6 and M7:

| bot | mean score shift | combined z |
| --- | ---: | ---: |
| `morpheus` | **+0.215** | **+10.77** |
| `morpheus-rs` | −0.053 | −1.27 |

**Only the Python bot moved.** The Rust bot is statistically unchanged. So the
question is not "why did the Rust bot look strong in M6" but "why did the
Python bot play so badly in M6, against every opponent at once".

That shape has an obvious *candidate* explanation — the Python bot has ~10 ms
of CPU headroom inside its 140 ms deadline (p50 133 ms) while the Rust bot has
~38 ms (p50 102), and the panel heuristics answer in microseconds, so anything
that slows the host hurts exactly one of the three. It is the explanation this
document cannot confirm: the 6-spinner probe is the direct test of it and it
failed.

## What this means beyond one number

The failure mode is not that a measurement was noisy. It is that **a
4,032-game round produced a tight confidence interval around an
unreproducible number, and nothing in the pipeline noticed.**
[`decision-rule.md`](../../arena/decision-rule.md) already warns that
round-to-round drift between byte-identical programs reached +46 Elo once; this
is the same phenomenon at nine times the size, and it passed every gate that
document specifies — registered hashes, one engine version, shared panel,
alternated seats, 1,152 games per arm, 973 decisive.

Two things follow, and they are process, not arithmetic:

1. **A published contrast for a deadline-driven bot needs a replication**, in a
   separate round, before it is written down. The gates cannot substitute:
   every one of them passed here.
2. **Host state is an unrecorded experimental variable.** The round manifest
   records the roster, the seeds, the seat policy, the engine and the job
   count. It records nothing about what else the machine was doing, and for a
   bot whose strength is a function of how much thinking fits in 140 ms, that
   is a first-class input. It should be captured — load average at minimum —
   and a round whose host state is unknown should be treated as unpublishable.

## What survives

The deployment case for the rewrite does not rest on the retracted number, and
is unaffected by any of this, because it was measured serially on the
deployment host rather than in a parallel round
([M0](morpheus-rs-baseline-modal.md), [M7](morpheus-rs-m7-knobs.md)):

| on one x86 core | morpheus | morpheus-rs |
| --- | ---: | ---: |
| moves over the 150 ms limit | **13.8%** (uncaptured control) | **0 of 21,000** |
| p99.9 move time | 423 ms | 138–140 ms |
| simulations per move | 4–8 | 16–20 |
| belief update *and* root inference complete | 11.7% of turns | 100% |

`RULES.md` charges one fault per late reply and forfeits at 50, so at 13.8% the
Python bot reaches the fault limit around move 360 of a 450–575 turn game. That
claim needs no tournament at all.
