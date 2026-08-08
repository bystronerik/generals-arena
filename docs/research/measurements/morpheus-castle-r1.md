# Morpheus measurement round `morpheus-castle-r1`

Strength of `morpheus@17c8ac2684ec` (the head after the castle-savings commit
`d57b545`) against macaria, blitz, and cm_hunter.

Raw games: `data/games/morpheus-castle-r1/` (60 games, gitignored).

## Method

```bash
python -m arena.tournaments.competition \
  bots/morpheus/run.sh bots/macaria/run.sh bots/blitz/run.sh bots/cm_hunter/run.sh \
  --round morpheus-castle-r1 --games-per-pair 10 --round-seed 7 \
  --seat-policy alternate --strict-versions --jobs 4
```

All six unordered pairs ran, so morpheus played 10 games per opponent (30
total) and the opponent-vs-opponent games gave the fit local connectivity.
Every game is `mode == "competition"`, one `engine_version`, seats exactly
5/5 per pair by alternation, no truncations. `--jobs 4` on 11 physical cores
was deliberate: morpheus is deadline-driven, so heavy CPU contention would
degrade its search and bias the round.

## This is not a decision

The [decision rule](../../arena/decision-rule.md) gate needs **≥30 games per
(arm, opponent)** and **≥200 per arm**; this round has 10 and 30. It is a
probe: strong enough to rank morpheus against these three, not to license a
verdict on any code change. No arm here is a frozen same-round baseline, so
nothing below is a before/after claim.

## Record

| Opponent | W | L | D | n | Winrate | Score | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cm_hunter | 9 | 1 | 0 | 10 | 90% | 0.90 | 496 |
| blitz | 3 | 7 | 0 | 10 | 30% | 0.30 | 536 |
| macaria | 2 | 8 | 0 | 10 | 20% | 0.20 | 384 |
| **total** | **14** | **16** | **0** | **30** | **47%** | **0.47** | — |

**Zero draws in 30 games**, against a pool draw rate near 35%. Games now
resolve; the truncation-draw failure mode is gone.

## Pairwise contrast

Positive = morpheus stronger. From a full-pool refit (339 rated games, 27
entities); morpheus is non-provisional here at exactly 30 games.

| Opponent | Δ | SE | CI₉₅ | P(morpheus stronger) | Games for SE=25 |
| --- | ---: | ---: | --- | ---: | ---: |
| cm_hunter@500fbc94bb55 | **+264.5** | 101.2 | [66, 463] | 0.996 | +317 |
| blitz@b4a69aad6389 | −65.0 | 87.3 | [−236, 106] | 0.228 | +202 |
| macaria@5760b30723e4 | **−278.6** | 92.5 | [−460, −97] | 0.001 | +328 |

All three are `comparable` (one connectivity group). Two intervals exclude
zero, so those directions hold even at n=10: **morpheus clearly beats
cm_hunter and clearly loses to macaria.** Blitz is a coin-flip on this
evidence — the interval spans ±170 Elo and needs ~200 more head-to-head
games to resolve to ±25.

## Castles

| Opponent | morpheus castles/game | opponent castles/game |
| --- | ---: | ---: |
| cm_hunter | 1.8 | 0.0 |
| blitz | 1.7 | 0.0 |
| macaria | 1.3 | 0.0 |

The savings pipeline fires in real rounds: morpheus builds 1.3–1.8 castles
per game and no opponent builds any. The macaria figure is lowest because
those games end soonest (mean 384 turns), which is also where the economy
has least time to pay back.

## Loss shape

Mean turns in morpheus wins vs losses: cm_hunter 543/76, blitz 610/503,
macaria 546/343. Losses to macaria and blitz are *mid-game* (343–503), not
early rushes and not attrition timeouts — consistent with losing a
mid-game exchange rather than being out-farmed to exhaustion.

## Next

- A verdict on today's stack needs a decision round: a frozen pre-session
  morpheus under `bots/`, same rounds, `--seat-policy alternate`, ≥200 games
  per arm. Note the pool also holds `morpheus@d3399c6dc5f5` (101 games,
  1639) from earlier rounds — it is **not** a valid comparator (measured
  round-to-round drift reaches +46 Elo between byte-identical programs).
- The macaria gap is the largest and best-measured deficit; it is the
  natural target for the next change.
