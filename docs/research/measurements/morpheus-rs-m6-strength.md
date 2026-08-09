# Morpheus-rs M6 — the arena gate, at parity knobs

> # ⚠ RETRACTED, 2026-08-10
>
> **The headline number below does not replicate and must not be quoted.**
> Four later measurements of the same two programs — one of which replays this
> round's own map seeds under this round's own job configuration — agree with
> each other at a score of **0.504 ± 0.042** and disagree with this round's
> 0.867 by **7.4 sigma**. The current best estimate of the same contrast is
> **+138 Elo**, from the M7 round. The cause has not been identified; map
> seeds, tournament parallelism, external CPU load and the programs themselves
> are each ruled out by direct experiment.
>
> Full account: [M6's strength result does not replicate](morpheus-rs-m6-replication.md).
>
> What survives is everything measured **serially on the x86 deployment host**,
> which is untouched by this: the Python bot is late on 13.8% of moves there
> and would forfeit on faults, while this binary is late on 0 of 21,000. That
> case never depended on the round below.
>
> The rest of this page is kept as written, as the record of what was
> published and how it was reasoned about.

> Verdict: **improvement.** `morpheus@73967d2125cc` → `morpheus-rs@5456f5532cc2`
> is **Δ = +425.06 ± 22.77 Elo, CI₉₅ [+380.44, +469.68], P(B > A) = 1.0000**,
> 1,152 games per arm. The M6 exit gate asks for `no change` or better, so it
> passes — but not for the reason §4 expected, and the gap between the
> expectation and the result is the finding. See
> [what the number is measuring](#what-the-number-is-measuring).

Milestone M6 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md),
success criterion 3. Thresholds and gates come from
[`decision-rule.md`](../../arena/decision-rule.md); nothing here invents one.

Date: 2026-08-09. Host: Apple M3 Pro, 11 parallel match workers.

## Design

- **Arms.** `morpheus@73967d2125cc` — the frozen oracle, registry step 18 —
  against `morpheus-rs@5456f5532cc2`, registry step 1. Both are the current
  lineage head of their bot id; the baseline is far from provisional.
- **Parity knobs, verified rather than asserted.** The two `deployment.json`
  files differ in exactly three fields, none of them behavioural: the
  limitation note, `inference_runtime` (`torch.jit.script+float32` against
  `morpheus-rs-bespoke+float32`), and `qualification_host`. Every deadline,
  particle count, batch size and shaping constant is identical.
- **Panel.** `cm_expander` (the rating anchor), `macaria`, `sosipolis`,
  `castle_builder`, `metro` — five bots spanning the range, the same panel M0
  captured its corpus against.
- **Round.** One round, `morpheus-rs-m6-strength`: seven bots, 21 unordered
  pairs, `--games-per-pair 192 --round-seed 7 --seat-policy alternate
  --strict-versions`. 4,032 games, all `mode == "competition"`, one
  `engine_version`. Both arms are in the same round against the same panel on
  the same seed set, which is what makes the contrast a contrast
  ([why](../../arena/decision-rule.md#cross-round-baselines-are-not-comparators)).

## Gate

| gate | requirement | morpheus | morpheus-rs |
| --- | --- | ---: | ---: |
| registered, competition, one engine | — | yes | yes |
| same connectivity group | `fit.comparable` | True | True |
| games per arm | ≥ 200 | 1,152 | 1,152 |
| games per (arm, opponent) | ≥ 30 | 192 | 192 |
| decisive games per arm | ≥ 60 | 973 | 1,121 |

## Results

| opponent | morpheus W-L-D | score | morpheus-rs W-L-D | score |
| --- | ---: | ---: | ---: | ---: |
| `cm_expander` | 130-9-53 | 0.815 | 175-0-17 | 0.956 |
| `macaria` | 44-141-7 | 0.247 | 107-85-0 | 0.557 |
| `sosipolis` | 79-108-5 | 0.424 | 162-29-1 | 0.846 |
| `castle_builder` | 131-3-58 | 0.833 | 190-0-2 | 0.995 |
| `metro` | 99-44-49 | 0.643 | 182-6-4 | 0.958 |

Head to head, 192 games with seats alternated on matched seeds:
**morpheus-rs 163 – 22 – 7**, a score of 0.867.

Fitted, on the whole pool (4,426 rated games, 30 entities):

```
morpheus-rs@5456f5532cc2   2361.0
morpheus@73967d2125cc      1935.9
delta = +425.06 +/- 22.77   CI95 [+380.44, +469.68]   P(B>A) = 1.0000
```

## What the number is measuring

§4 predicted `no change`: "at identical configuration the Rust bot should play
the same bot, so anything worse means a real divergence." The first half of
that sentence is wrong, and it is wrong in a way worth writing down, because
the same reasoning will show up again at M7.

**Identical configuration is not identical behaviour when the configuration is
a deadline.** Every knob in `deployment.json` is a time budget or a bound on
work attempted inside one, and the two implementations do different amounts of
work inside the same budget. That is not a divergence between the bots; it is
the entire premise of the rewrite (§7), and the amount is measured rather than
inferred — from
[the latency report](morpheus-rs-m6-latency.md), on the same host and the same
schedule:

| | morpheus | morpheus-rs |
| --- | ---: | ---: |
| belief update **and** root inference both completed | 18.4% of turns | 100% |
| completed simulations per normal move (p50) | 12 | 16 |
| normal moves over the judge's 150 ms limit | 8.76% | 0.02% |

A bot that finishes its belief update on 18% of turns is playing most of the
game from a stale posterior and a degraded fallback. The contrast measures that
gap closing. It is emphatically **not** evidence that the port is faithful —
faithfulness is what the parity harness proves, at 533,726 cases and a
tier-3 decision surface reading 1,322 / 1,322. The two results answer different
questions, and this one would look much the same if the port were faithful *or*
subtly wrong in a way that happened to help.

## What this does not settle

- **The host is not the judge.** Eleven workers on an eleven-core laptop, two
  deadline-driven processes per game. Both arms met the same contention, and
  M0 measured the same degradation *serially*, so the effect is not an artifact
  of the parallelism — but the competition host gives one dedicated core, and
  the qualification numbers M7 needs come from there, not here.
- **The pool is global.** The fit reads every game under `data/games/`, which
  includes the 20 serial latency games of round `morpheus-rs-m6` (morpheus-rs
  against the same five panel bots) and 374 older games. That is 1.7% extra
  games on the candidate arm, all against panel opponents, and it moves an
  estimate this wide by nothing that changes a verdict.
- **Nothing about knobs.** Both arms ran a configuration whose own file records
  *"Qualification verdict is no on this host"*. M7 re-derives every field from
  the costs this binary actually has, and the sensible expectation is that the
  gap grows: `morpheus-rs` spent this round hitting `target_simulations = 16`
  on the median move — a ceiling fitted to a bot that could not reach it.

## Read alongside

- [M6 latency](morpheus-rs-m6-latency.md) — where the difference comes from
- [M0 baseline](morpheus-rs-baseline.md) — the Python's numbers, same schedule
- [decision rule](../../arena/decision-rule.md) — the thresholds this quotes
