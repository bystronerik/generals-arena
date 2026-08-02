# 026 — sosipolis: Kubic align (castle + feed + tip)

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Prior: [`025-sosipolis-tip-mass.md`](025-sosipolis-tip-mass.md).
Kubic profile: observational batch over `competition-replays/Kubic/` (390 played).

## Hypothesis

Re-enabling one early castle and lowering contact/strike tip gates moves
Sosipolis toward Kubic timing: first castle ~tick 10, first sight toward ~182,
and shorter sight→kill conversion, while raising win rate on the Macaria panel.

## Parameter revision

| Param | Old | New | Rationale |
| --- | ---: | ---: | --- |
| `CASTLE_MAX` | 0 | 1 | Kubic median 1 castle at tick 10 in ~76% of wins |
| `CONTACT_ASSAULT_STACK` | 55 | 30 | Cut exclusive contact feed that delayed sight |
| `STRIKE_MIN_TIP` | 50 | 25 | Kubic kills from ~23 army at sight in median 24 ticks |

Content hash: baseline `6709f0bacff9` → treat `77b19da7ac39`.

## Gate

Seed 0 vs `smoke` (`--mode competition`): sosipolis captured at turn 335, built
1 castle (turn 100), no faults.

## Macaria grid

Round: [`sosipolis-tip4`](../measurements/sosipolis-tip4.md) — seeds 0–19,
`--seat-policy alternate` (40 games), `--bots macaria sosipolis`.

| Round | Games | sosipolis W-L-D | Mean turns | Notes |
| --- | ---: | --- | ---: | --- |
| sosipolis-tip1 | 10 | 1-9-0 | 323 | tip delivery |
| sosipolis-tip2 | 10 | 1-9-0 | 333 | mass + frac gate |
| sosipolis-tip3 | 10 | 0-10-0 | 357 | path+min march |
| sosipolis-tip4 | 40 | **5-35-0** | **245** | castle+lower tip gates |

Observed on tip4 (from game `metrics`):

| Metric | Value | Kubic target |
| --- | --- | --- |
| sosipolis games with ≥1 castle | 31/40 (77.5%) | ~76% |
| mean sosipolis castles | 0.78 | ~1.14 |
| sosipolis win turns (median) | 217 | kill ~204 |
| losses under 150 turns | 7/35 | — |

Formal pairwise Elo contrast vs tip3 hash is **unproven** (one panel,
≪ 200 games/arm; tip3 was a different round — not a matched dual-arm).

## Decision

**Keep** `CASTLE_MAX=1`, `CONTACT_ASSAULT_STACK=30`, `STRIKE_MIN_TIP=25`.

Castle programme now matches Kubic’s ≥1-castle rate. Raw win rate on the
Macaria panel rose vs tip3 (0% → 12.5%) and mean turns fell (357 → 245), with
wins landing near Kubic’s kill band. Macaria still wins most games; early
losses under 150 turns remain. Next: raise sight rate / hunt tempo without
re-raising the tip gates that this arm just cut.
