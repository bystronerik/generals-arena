# Strategy spec — boom

Bot: `bots/boom/`. Migrated from the generals-bot repo as a grid-native
rewrite. Grounded idea: **fast-expand economy — refuse to fight until the
economy has won, then cash it in with one fist.** Baselines for comparison:
`expand_plus`, `castle_builder`, `smoke`.

Rule references: [`RULES.md`](../../../RULES.md) sections 03 (build castles),
04 (army growth), 08 (draw at 1200). Turn cadence matches the source
ruleset, so source-tuned values transfer as **hypotheses**.

## 1. Priority ladder (every turn)

| Priority | Trigger | Behavior |
| --- | --- | --- |
| Finish | enemy general sighted and beatable | Adjacent capture (deathtouch-aware); `chase_defence` from 780. |
| Defend | `threatened` (enemy at the door, home power short) | Kill the intruder or walk the biggest reserve home *whole*. |
| Evict | enemy cell within `INTERCEPT_DIST = 8` of the general | Retake it — a two-land swing that stops raids early. |
| Free captures | frontier stack next to neutral land | Never skipped, even mid-attack: land bonuses compound. |
| Attack | endgame latch flipped | Collect one fist, walk it at the (estimated) general. |
| Build | `cost × 2 ≤ spare army`, turn ≥ 60 | Castle build — the arena's replacement for city purchases. |
| Run | `stack army ≥ distance to neutral` | Expansion run down the gradient, collecting en route. |
| Consolidate | nothing else | Walk stranded army to where it will be spent next. |

## 2. The economy engine

One capture costs two turns of production but only one move, leaving one
spare move per capture for travel — hence the run rule
`army >= distance` (`RUN_SLACK = 0`, break-even). Direct captures rank
sources to protect the accumulator: frontier stacks are spent before the
general's bank. Until `AVOID_ENEMY_ADJACENT_UNTIL = 160`, neutral cells
adjacent to the enemy are deprioritized (no early border fights).

## 3. Home guard — the measured defence

The guard is *home power*: army on the general plus the
`DEFENDERS_COUNTED = 2` biggest stacks close enough to walk back before the
nearest enemy arrives — deliberately not the sum of every two-army cell.
The threat is discounted `1/cell` of our land it must cross (a deep raid
bleeds out on its own), remembered threats count at half weight, and the
guard never exceeds `MAX_GUARD_FRACTION = 0.7` of total army. When the
guard is short, the bank is locked out of every other ladder step.

## 4. City purchases → castle builds (the arena adaptation)

The source bought neutral cities gated by `cost × 2 ≤ spare`. The arena has
no neutral cities; the same gate now funds **castle builds**:

- Site: cheapest owned plain cell (`build_cost` incl. crowding surcharge),
  preferring cells already holding army.
- Funding: the biggest field stack collect-walks onto the site; the build
  (`2 r c 0 0`) fires once the cell holds `cost + BUILD_KEEP = 3`.
- Hysteresis: a started plan survives while `cost × 1.5 ≤ spare` and the
  cell is still ours.

Differentiator vs `castle_builder`/`castle_rush`: boom funds builds from
**field surplus routed by collect-walks**, never by resting the general,
and only when captures/attack leave the army idle. Boom's identity stays
"tiles first"; builds are a sink for surplus, not the plan.

## 5. Endgame latch

Flips when: army ratio ≥ 1.5; or their army collapsed (`drop ≥ 20`) while
we lead ≥ 1.15; or no land is left to take (turn ≥ 150); or expansion
stalled while we lead. Releasable below ratio 0.95 — attacking from behind
throws away a won game. **Arena addition:** `FORCE_COMMIT_TURN = 950`
flips the latch unconditionally — with a hard draw at 1200, hoarding wins
nothing. The strike is sized `max(25, 0.8 × opp mobile army)`, tracks its
identity (a spent fist re-gathers rather than trickling), and never
launches from a locked bank.

## 6. Arena-rule adaptations (vs the generals-bot source)

| Rule | Adaptation |
| --- | --- |
| No neutral cities | City purchase module → castle build module (§4). |
| Deathtouch (800) | Finish accepts any 2-army neighbour; `chase_defence` from 780. |
| Draw at 1200 | `FORCE_COMMIT_TURN = 950` latch override. |
| No AFK kick | Fallback shuffle kept for tempo, not survival. |

## 7. Diversity check

vs `expand_plus` (greedy capture + march): boom schedules runs against the
`army >= distance` break-even, locks its bank behind a measured guard, and
has an explicit economy→attack latch. vs `castle_builder` (rested-general
home builds): boom never rests the general to fund a build. vs `garrison`
(defensive reserves): boom's guard is threat-scaled and capped, not a
standing reserve.

## 8. Experiment hypothesis

**Hypothesis:** out-expanding the pool early converts to wins via the
endgame latch before the draw cap; the guard model prevents the classic
economy-bot loss (general sniped mid-boom). Verification games: boom beat
`smoke` (capture, turn 379) and `expand_plus` (capture, turn 337) with
zero faults. Castle builds fired 0 times in both — captures and the attack
latch dominated; whether the build gate should loosen
(`BUILD_SPARE_MULTIPLE`) is a measurement-round question.

Seed grid: opponents `smoke`, `expand_plus`, `castle_builder`,
`late_rush`; seeds 0–2; `--mode competition`. Metrics: W-L-D, mean turns,
castles built, guard at end, faults.

See [`013-boom-economy-latch.md`](../experiments/013-boom-economy-latch.md).
