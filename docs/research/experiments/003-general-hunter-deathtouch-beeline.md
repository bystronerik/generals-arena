# 003 — general_hunter: deathtouch beeline

Bot: [`bots/general_hunter/`](../../../bots/general_hunter/). Baseline: `expander_python`, `smoke`.

## Hypothesis

From turn 800, any move that *executes* onto the enemy general's tile wins
instantly regardless of defending army (RULES.md section 07 /
`docs/competition/deathtouch.md`). Generals never move, so a single
sighting of the enemy general (even one that later fades back into fog)
gives an exact, permanent target. A bot that remembers that sighting and
beelines a runner stack toward it once turn ≥ 800 — attacking immediately
whenever adjacent with army ≥ 2 — should convert sightings into wins that
a pure expander would not, since a pure expander only attacks a visible
enemy general opportunistically and stops tracking it once it fogs over.

## Verification of the mechanism (unit-level, not a full match)

Before running full games, the beeline/execute logic was checked directly
against a synthetic observation (own stack 2 tiles from an enemy general
with a 100-army garrison, `turn=850`): the agent advances one step on the
first call, then correctly issues the execute move onto the general's tile
on the next call, ignoring the (irrelevant, per deathtouch) garrison size.
This confirms the mechanism is implemented correctly, independent of
whether it fires in a full game.

## Seed grid

- Opponents: `smoke`, `expander_python`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (from `data/games/`, see tournament summary)

| Matchup | Games | general_hunter W-L-D | Mean turns |
| --- | --- | --- | --- |
| general_hunter vs smoke | 3 | 0-0-3 | 1200.0 |
| general_hunter vs expander_python | 3 | 0-0-3 | 1200.0 |

All 6 games ran the full 1200-turn cap and drew. None of the greedy
opponents ever pushed deep enough into `general_hunter`'s territory (nor
vice versa) for either general to be sighted before truncation — the two
sides expand outward from their own corners and meet at a shared frontier
without either side's core, let alone its general, coming into view. The
beeline mechanism therefore never activated in this grid; the unit check
above is the only positive evidence it works.

## Decision

**Keep**, but flag as **unproven in full games** with the current
opponent pool. All three bots here (`smoke`, `expander_python`,
`general_hunter`) only expand outward toward visible neutrals — none of
them probe into enemy territory, so the enemy general is rarely if ever
sighted before the 1200-turn cap. A meaningful test needs either (a) an
opponent/bot that deliberately scouts or raids past the frontier, or (b) a
longer/larger seed grid sampling for the rare games where a sighting does
happen. Revisit once such an opponent exists rather than reverting a
mechanism that has not had a fair chance to fire.
