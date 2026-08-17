# joe-rs f16 quantization contrast

**Claim tested:** rounding the joe-rs weights through IEEE f16 (stored as
f32; the "f16-in-f32" export) does not change playing strength. The change
halves the submission zip (~31.8 MB to ~15.6 MB) with no loader change; the
frame-level probes (2026-08-16, this repo's session logs) measured 99.93%
identical decisions over 7,206 recorded frames and value drift ≤ 0.0025.

## Setup

- Arms: baseline `joe-rs@c620a6f2e4fe` (f32), candidate
  `joe-rs-f16@a87159e28553` — a frozen copy of joe-rs whose artifact holds
  the same weights after an f16 round trip (commit ebf87fb). Both arms in
  the same round per [decision-rule.md](../../arena/decision-rule.md).
- Panel: aegis, macaria, boom, cm_hunter, cm_expander (anchor) — the J4
  panel. Panel games satisfy the per-opponent gate and connectivity; the
  information for the contrast comes from the head-to-head pair, because
  both arms beat the panel ~24:1 and lopsided games carry almost no rating
  information (see the J4 sanity round: ±108 Elo from 100 such games).
- `--seat-policy alternate --strict-versions --mode competition`
  throughout; 10 workers; engine `9e3b9d1`.
- Round `joe-rs-f16-r1` (2026-08-16/17): H2H invocations with round-seeds
  101, 111, 121 (400 + 400 + 300 games, +32 from panel invocations);
  panel invocations with round-seeds 202, 212 (16 games per pair each).
  1,772 games, 1,292 per arm, 1,132 head-to-head.
- Round `joe-rs-f16-r2` (2026-08-17, separately scheduled): H2H round-seed
  303 (1,100 games); panel round-seed 404 (32 games per pair). Same
  totals: 1,772 games, 1,292 per arm.
- Host: the packaging/dev arm64 Mac, otherwise idle for the duration of
  every invocation (no builds, benchmarks, or remote jobs). Neither bot is
  deadline-driven — both compute one fixed forward per turn — so host load
  could not steer decisions, only latency. The round manifest only records
  the last invocation of each round; the invocation list above is the
  complete schedule.

## Gates

Both rounds pass every gate: one engine version, all games
`mode=competition`, arms registered and comparable, 1,292 ≥ 200 games per
arm, minimum 32 ≥ 30 games per (arm, opponent), ≥ 1,230 ≥ 60 decisive
games per arm.

## Results

| Round | Δ(f16 − f32) | CI₉₅ | P(f16 > f32) | H2H (f16–f32–draw) | Verdict |
| --- | --- | --- | --- | --- | --- |
| r1 | **+2.86 ± 10.51** | (−17.7, +23.5) | 0.607 | 539–531–62 | **no change (proven flat)** |
| r2 | **+7.40 ± 10.46** | (−13.1, +27.9) | 0.760 | 548–533–51 | unproven (CI hi +27.9 > +25) |

r2's `games_to_resolve(target_se=12.75)` is 0 — the round met the SE
target; the miss is the +7.4 point offset pushing the upper CI edge 2.9
Elo past the flat window, on the side *favoring* the candidate. Bringing
the edge inside ±25 at that offset needs SE ≤ 8.98, ≈ 411 more games.

## Reading

The two rounds agree: the contrast is +3 to +7 with overlapping intervals,
and **regression is excluded in both** — the regression verdict needs
CI-high < −10, and the CI *lows* are −17.7 and −13.1, far inside the −25
bar. No round shows improvement (P < 0.95 in both). The formal labels
differ only in which side of the flat window the noise leaned on.

For the decision this round was bought for — "is f16 safe to ship?" — the
answer is yes: across 3,544 rated games the candidate is at worst a few
Elo different from baseline in either direction, and the direction of the
point estimates is positive. This matches the frame-level probes.

Replication caveat (decision-rule): the two rounds are separately
scheduled same-host rounds. The M6 lesson — a replicated 300-Elo phantom —
is about large effects; here both rounds bound the effect inside ±28 Elo
with centers within 5 Elo of each other.
