# Diversity constraints

Hard rules that keep the bot roster different from each other. This file is a
contract for every bot author and every optimization pass. It contains no bot
code.

Read this file before you change a bot. If a change breaks a rule in
[§2](#2-non-negotiable-differentiators) or [§3](#3-forbidden-cross-copying),
do not make the change.

Related files:

- [`experiment-protocol.md`](../experiment-protocol.md) — how to measure a change.
- [`optimize-existing.md`](optimize-existing.md) — the approved parameter revisions.
- [`tournament-plan.md`](tournament-plan.md) — the round schedule.

## 1. Why this file exists

The roster shares one ancestor. `expander_python` supplies the greedy capture
loop, and `bots/expand_plus/agent.py` supplies the BFS march fallback. Seven of
the eight bots contain that same loop with the same weights
(`score = army * 10 * (2 if opponent else 1)`).

Two forces push the roster toward one policy:

1. **Shared ancestry.** Each new bot forks from `smoke` plus `expand_plus`, so
   every bot starts from the same base policy.
2. **Copying the leader.** After each tournament round, the natural move is to
   copy the top bot's tactic into every other bot. That raises the mean rating
   and destroys the roster.

A single-policy roster is worthless for research. Mirror matches produce draws,
Elo stays flat at the initial value, and no experiment can separate cause from
noise. The current leaderboard already shows this failure: 31 rated games, 0
wins, 0 losses, 31 draws, every bot at 1500.0 Elo.

The roster is a **measurement instrument**, not a ladder. Its value is the
spread of behavior it covers, not the peak strength of one member.

## 2. Non-negotiable differentiators

Each bot owns exactly one differentiator. The differentiator is the reason the
bot exists. Remove it and the bot must be deleted, not repaired.

| Bot | Non-negotiable differentiator | Must always be true | Must never be true |
| --- | --- | --- | --- |
| `smoke` | Protocol reference. Minimum viable policy, no scoring, no memory, no map analysis. | Picks the first legal expansion move it finds. Holds no state between turns. | Any scoring weight, any BFS, any turn-based gate, any castle build. |
| `expand_plus` | Land maximization with zero investment and zero targeting. | Every action serves land growth: greedy capture, else BFS march to the nearest capturable tile. | Any castle build. Any general targeting. Any resting cell. Any defensive reserve. |
| `castle_builder` | Conservative economy. Rests the general to fund a small number of late, well-funded castles. | Builds only with a thick surplus margin and a long cooldown. Keeps a low castle cap. | A build cap or timing that reaches `castle_rush` values. Any hunt behavior. |
| `castle_rush` | Aggressive economy. Same build mechanism as `castle_builder`, every knob shifted toward early and thin. | Build start earlier, cap higher, cooldown shorter, and surplus margin thinner than `castle_builder` on every parameter. | Any parameter that equals or crosses a `castle_builder` value. Any hunt behavior. |
| `general_hunter` | Deathtouch execution. Expansion until turn 800, then a beeline onto a remembered enemy general cell. | Stores the enemy general cell forever after one sighting. Switches to the beeline at turn 800. | Any castle build. Any attack committed before turn 800. |
| `phase_switch` | Explicit hard phase clock. One `if/elif` on `obs.turn`, no blending. | Three disjoint turn windows, each running one policy only. Builds are confined to the mid window. | Soft weights across phases. Hysteresis. Any behavior that runs in more than one phase. |
| `fog_scout` | Information gain. Scores unknown map area above known land. | Keeps an `ever_seen` grid. Applies a fog bonus that exceeds the opponent bonus. | A fog bonus reduced to or below the opponent bonus. Any castle build. |
| `army_convey` | Interior-to-frontier logistics. Moves idle interior army forward, not new captures. | Runs a convey step whenever no frontier capture exists. Scores convey moves by stack size over distance to frontier. | Replacing the convey step with the `expand_plus` BFS march. Any castle build. |
| `garrison` (spec) | Defense first. Holds a reserve on the general before it expands. | Sizes a reserve from a threat model and refuses expansion moves that break the reserve. | Spending the reserve for a capture. Any castle build. Any hunt before turn 800. |
| `late_rush` (spec) | Single-stack commitment. Merges army into one main stack and commits it in a fixed turn window. | Chooses one rally cell and one main stack, then commits between turn 700 and turn 800. | Splitting the main stack for land. Committing before the accumulation phase ends. |

`expander_python` is the external baseline from `competition-module`. Never
change it and never treat it as a roster member.

### 2.1 One axis per bot

Each differentiator sits on a named axis. Two bots may not occupy the same axis
in the same direction.

| Axis | Owner (direction) |
| --- | --- |
| Baseline / protocol floor | `smoke` |
| Land rate | `expand_plus` |
| Economy, conservative | `castle_builder` |
| Economy, aggressive | `castle_rush` |
| Kill condition, deathtouch | `general_hunter` |
| Control structure, hard clock | `phase_switch` |
| Information | `fog_scout` |
| Logistics | `army_convey` |
| Defense | `garrison` |
| Force concentration | `late_rush` |

A new bot needs a free axis. If the axis is taken, the proposal is rejected or
it replaces the current owner. It never runs beside the current owner.

## 3. Forbidden cross-copying

The following transfers are forbidden, whatever the measured gain.

| Tactic | Owner | May not be copied into |
| --- | --- | --- |
| Castle build and general rest | `castle_builder`, `castle_rush`, `phase_switch` (mid window only) | `smoke`, `expand_plus`, `general_hunter`, `fog_scout`, `army_convey`, `garrison` |
| Deathtouch beeline from turn 800 | `general_hunter`, `phase_switch` (late window only), `late_rush` | `smoke`, `expand_plus`, `castle_builder`, `castle_rush`, `fog_scout`, `army_convey` |
| Permanent enemy-general memory | `general_hunter`, `phase_switch`, `late_rush`, `garrison` (threat model only) | `smoke`, `expand_plus`, `castle_builder`, `castle_rush`, `army_convey` |
| Fog bonus and `ever_seen` grid | `fog_scout` | every other bot |
| Interior convey scoring | `army_convey` | every other bot |
| General reserve sizing | `garrison` | every other bot |
| Single main-stack rally | `late_rush` | every other bot |
| Hard turn-window `if/elif` control | `phase_switch` | every other bot |

### 3.1 The three tests

Apply all three tests to every proposed change. A change that fails any test is
rejected.

1. **Identity test.** Read the row for this bot in
   [§2](#2-non-negotiable-differentiators). Does the change keep the "must
   always be true" column true and the "must never be true" column false? If
   not, reject.
2. **Distance test.** After the change, does at least one behavior separate this
   bot from every other roster member on a shared seed? If two bots produce the
   same action sequence on the same seed, they are one bot. Reject the change
   and delete one bot.
3. **Source test.** Where did the idea come from? An idea taken from a stronger
   roster member's differentiator is forbidden. An idea taken from `RULES.md`,
   from the bot's own measurements, or from an axis nobody owns is allowed.

### 3.2 What is always allowed

These are shared infrastructure, not tactics. Copy them freely.

- The stdio wire protocol in `main.py` and `run.sh`.
- Passability checks: mountain (`2`) and fogged structure (`5`) are impassable.
- Bounds checks, the direction table, and the `PASS` fallback.
- The build-cost formula from `RULES.md` section 03, for bots that own the
  build tactic.
- Bug fixes: an out-of-range index, a fault, or a timeout. A bug fix is not a
  tactic transfer.
- Parameter values inside a bot's own axis, when
  [`optimize-existing.md`](optimize-existing.md) approves them.

## 4. Convergence alarms

Check these after every tournament round. Any alarm blocks the next round until
it is cleared.

| Alarm | Threshold | Required response |
| --- | --- | --- |
| Draw rate | above 80% of rated games | Do not raise aggression across the roster. Change the differentiator of one bot only, and record which one. |
| Identical action traces | two bots match on more than 90% of turns on the same seed | Fail the distance test. Delete or re-specify one bot. |
| Elo spread | all bots inside a 25-point band after 30+ rated games | The instrument cannot separate the bots. Add a bot on a free axis. Do not tune the current bots closer together. |
| Parameter drift | a `castle_rush` build parameter reaches a `castle_builder` value | Revert the parameter. The gap between these two bots is the experiment. |
| Shared-loop share | more than 8 bots run the unmodified greedy capture loop | Freeze new forks from `expand_plus`. The next bot must bring its own action-selection core. |

## 5. Change record

Every change to a bot must state, in its experiment note under
[`../experiments/`](../experiments/):

1. The bot's axis from [§2.1](#21-one-axis-per-bot).
2. The result of each of the three tests in [§3.1](#31-the-three-tests).
3. The source of the idea, per the source test.

An experiment note without these three items is incomplete, and the change does
not merge.
