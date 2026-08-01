# 018 — yankee: a scoped search and a §07 endgame core

Bot: [`bots/yankee/`](../../../bots/yankee/), a fork of
[`bots/proteus/`](../../../bots/proteus/). Follows
[017](017-proteus-detection-rework.md). Spec:
[`strategies/yankee.md`](../strategies/yankee.md). Results:
[`measurements/yankee.md`](../measurements/yankee.md).

Baseline `proteus@7ae237e99d34` · candidates `yankee@67a2a9e6e444` (superseded)
and `yankee@5f9aff334f46`.

## Target

≥95% against `cm_hunter` over ≥200 games, with no regression on a panel of
≥5 including the anchor `cm_expander`.

## Verdict

**Target not met.** Best measured result against cm_hunter is **93.7%**
(253-17-0 over 270 seat-alternated games, CI₉₅ [90.1, 96.0]) against a matched
proteus arm of 90.7% on the same rounds. The target sits inside the interval
and above the point estimate; more games narrow the interval and do not move
the estimate.

The panel contrast is `unproven` rather than `improvement` or `no change` —
see [`measurements/yankee.md`](../measurements/yankee.md) §2 for the numbers
and §3 for why the later rounds under-measure the candidate.

## Four things measured, three of them negative

| increment | result | shipped |
| --- | --- | --- |
| parameter tuning | best cell +2 games in 200, sign test p ≈ 0.8 | no — kept at proteus's values |
| standing home guard | 0.922 → 0.850 → 0.770, monotone in the dose | no — kept at 0 |
| MCTS, unscoped | 0.917 → **0.450**, and 0.183 at a wider window | no |
| MCTS, scoped to a live general threat and to the blitz core | 0.890 → **0.945** | **yes** |
| §07 endgame core | draws 37 → 17 over 400 games on four draw-heavy opponents | **yes** |

## What the negatives are worth

**The premise did not survive re-measurement.** The brief's baseline — 46-4
over 50 games, all four losses at turns 202-314, all four with cm_hunter in
seat A — describes `proteus@91d2a88f8316`, two lineage steps behind the head.
Re-measured on the head at 300 games the pooled winrate reproduces (92.7%) and
the description does not: 13 of 22 losses land before turn 160, and 15 of 22
fall in the *other* seat. n=4 was the whole earlier sample. **Re-measure the
baseline before optimising against its description.**

**A pooled contrast can be flat and still hide a large regression.** The first
candidate read Δ = −1.67 ± 12.66 over 2310 games — flat — while carrying
fog_scout 0.894 → 0.824 and metro 0.912 → 0.859, roughly −95 Elo each, paid for
by gains on cm_hunter, boom and aegis. The per-opponent table is not a
courtesy; it is the check.

**A search may only override a core that does not depend on having chosen.**
Both regressions were the MCTS overriding **boom**, whose `build_target`,
`build_stack` and `striking` are latches set while choosing and trusted
afterwards — and whose castle build is the last move of a multi-turn plan the
search cannot even represent, so vetoing it cost every castle rather than one.
blitz survives an override because `ensure_stack` and the chain-head check
re-derive from the board. Restricting the search to blitz and forbidding it
from vetoing builds recovered both cells and kept the whole cm_hunter gain.

**A self-enforced wall-clock cap is conditional on being scheduled.** Idle, the
40 ms budget holds: p95 41.8 ms, max 50.6 ms, zero moves over 100 ms. Under a
13× oversubscribed machine the same code produced a 177 ms search and four
moves over §08's 150 ms limit — the deadline check cannot fire while the
process is not running. The guarantee is real *on the competition's dedicated
core* and should be stated that way.

## What was not tried

The replays say the army is in the wrong place *before* the threat is
observable. The guard attacked that by banking army at home and lost more than
it saved. The untried version biases *expansion direction* toward home during
the fist window instead — same hypothesis, no static reserve. That is the next
experiment.

## Reproduction

```bash
python -m arena.tournaments.competition \
  bots/yankee/run.sh bots/proteus/run.sh bots/cm_expander/run.sh bots/cm_hunter/run.sh \
  --round yankee-r4 --games-per-pair 220 --round-seed 41 \
  --seat-policy alternate --strict-versions
```

Control arms are environment overrides, not separate bots:
`YANKEE_TUNE='{"mcts_enabled": false}'` and
`YANKEE_TUNE='{"deathtouch_enabled": false}'` (see `bots/yankee/params.py`).
Every stored game above ran with the variable unset.
