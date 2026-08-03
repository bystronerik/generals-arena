# 055 — sosipolis: engagement as a rule, not a search

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 054.

Two things drove this: single Kubic details keep measuring as nothing on their
own, and the bot's play depends on how much search fits in a tick, which makes
both the bot and the measurements unstable.

## Measurement first: the grid got bigger

054 established the noise floor the hard way — a behaviourally identical build
with one extra 5x5 scan per macro scored 7/20 where the original scored 8. So:

- **Serial only.** Parallel grids are 2.2x faster and *not reproducible*: the
  same build over seeds 0-19 at 4 jobs scored 6, 6, 8. Contention changes what
  fits in the deadline. Serial reruns are exact.
- **Never alongside other work.** A background 100-seed grid run while JAX
  replays were analysing was scoring 3/16; it was measuring my own CPU load.
- **60 seeds, not 20.** Baseline on 0-59 is **20/60 = 33%**, against 17/40 on
  0-39 — seeds 40-59 contribute only 3/20, so the smaller grid was flattering.

## Where the instability lives

Move time by phase, over 20 recorded games:

| phase | median | p90 | at budget (>=50 ms) |
| --- | --- | --- | --- |
| search | 1 ms | 40 | 0% |
| contact | 10 ms | 18 | 0% |
| **strike** | **55 ms** | 56 | **58%** |

Search and contact finish early. Strike spends its entire
`STRIKE_MARCH_BUDGET_MS` on nearly every tick, so its iteration count — and
therefore its move — depends on machine load. That is the phase where kills
happen.

Kubic does not search here. Post-sight it marches the remembered cell at a
median path overhead of 11-20%, and its first wave kills ~85%.

## The bundle

Three measured Kubic engagement details, applied together because separately
they are worth nothing (see the table below):

1. **Attack source** = largest stack already adjacent to enemy land — Kubic
   94.5-95.6%, ours 43%. `select_mass_tip(prefer_front=True)` in contact.
2. **Attack destination** = enemy neighbour minimising BFS distance to the
   believed general (Kubic ~87% pre-sight), never at a losing margin
   (`P(attack | negative margin)` — Kubic ~1%, **ours measured 3.1%**, and on
   seed 1 alone that threw away 520 army). `components/tip.py::attack_move`.
3. **Post-sight march** = deterministic cheapest-route step to the remembered
   general, replacing StrikeMCTS on the hot path.
   `components/tip.py::strike_march`.

## Result

| build | wins /60 |
| --- | --- |
| baseline | **20** |
| front-tip only (one call site) | 21 |
| front-tip only (both call sites) | 20 |
| + attack rule, forward-only | 18 |
| + attack rule, minimise-BFS | 19 |
| **+ deterministic strike march (all three)** | **20** |

**No winrate change.** What it buys is predictability:

| | before | after |
| --- | --- | --- |
| strike decision, median | 55 ms | **1 ms** |
| strike ticks at budget | 58% | **15%** |
| decisions made by rule | 0 | 509 (`attack` 401, `march` 108) |

That is the reason it ships despite a flat win rate: the same position now
plays the same way regardless of what else the machine is doing, and 509
engagement decisions a game no longer depend on an iteration count. It also
removes StrikeMCTS from the common path entirely.

## Caveat

A flat 60-seed result is not proof of *no* effect — 20 vs 20 is consistent with
anything up to about +/-4. The claim here is narrow: this bundle does not cost
winrate, and it removes a measured source of run-to-run variation. It is not
evidence that Kubic's engagement model is worth nothing.

## Reproduction

```bash
for s in $(seq 0 59); do
  python competition-module/competition/matchup.py \
    bots/sosipolis/run.sh bots/macaria/run.sh --mode competition --seed $s
done
```

Run serially, on an otherwise idle machine.
