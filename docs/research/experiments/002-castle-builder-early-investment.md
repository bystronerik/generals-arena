# 002 — castle_builder: rested-general castle investment

Bot: [`bots/castle_builder/`](../../../bots/castle_builder/). Baseline: `expander_python`, `smoke`.

## Hypothesis

A built castle produces one army every other turn, like a second general
(RULES.md section 03/04). Investing early into one or two cheap castles
(built at Manhattan distance ≥ 7 from other own structures, base cost 35)
should pay for itself over a 1200-turn game and raise late-game army/land
versus pure greedy expansion.

**First attempt (rejected before the tournament):** build only when some
already-owned cell opportunistically has spare army above cost + margin.
Instrumented with `obs.turn`/`obs.my_land`/surplus logging, the largest
single-cell army stayed roughly 20–30 below the 35 base cost for the whole
game — greedy capture always spends a large stack on the next capture
before it can idle long enough to afford a castle. Zero castles were built
in 3/3 probe games. This is itself a measured result: **opportunistic-only
building does not work under greedy expansion** and needed a deliberate
mechanism instead of a passive check.

**Revised mechanism (what is measured below):** once land is established
(`my_land >= 8`), stop spending the *general's own* army on captures (every
other owned cell keeps expanding normally) so the general's stack idles and
grows. Once it can afford the cheapest reachable neighbor cell, relocate
the whole stack one hop and build next turn. Cap at 2 castles, turn window
20–900.

## Seed grid

- Opponents: `smoke`, `expander_python`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (from `data/games/`, see tournament summary)

| Matchup | Games | Castles built (castle_builder) | castle_builder W-L-D | Mean turns |
| --- | --- | --- | --- | --- |
| castle_builder vs smoke | 3 | 2 per game (6/6) | 0-0-3 | 1200.0 |
| castle_builder vs expander_python | 3 | 2 per game (6/6) | 0-0-3 | 1200.0 |

The revised mechanism reliably builds both castles every game (verified via
`[matchup] castles built:` log lines), unlike the rejected opportunistic
version. All games still ran the full 1200-turn cap and drew — the current
game-record schema stores only winner/turns/terminated/truncated, not final
land/army, so this grid cannot yet measure the intended economic payoff
(only that resting the general does not cost the game outright).

## Decision

**Keep** the rested-general mechanism over the opportunistic one (it is
the only version that ever fires). Treat the economic payoff claim as
**unverified, not confirmed** — no regression observed (still finishes
clean, still draws), but proving a growth/winrate edge needs richer
per-game telemetry (final land/army at truncation, or army-over-time)
which is a schema follow-up, not a strategy change.
