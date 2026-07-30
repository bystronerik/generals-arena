# Diversity constraints

Hard rules that keep the twelve bots different from each other. This file is a
contract for every bot author and every optimization pass. It holds no bot
code and no tuning advice.

Read this file before you change a bot. If a change breaks a rule in
[§3](#3-non-negotiable-differentiators) or [§4](#4-forbidden-cross-copying),
do not make the change.

Related:

- [`../measurements/round1.md`](../measurements/round1.md) — the evidence in §2.
- [`optimize-existing.md`](optimize-existing.md) — the approved revisions.
- [`tournament-plan.md`](tournament-plan.md) — the round schedule.
- [`../experiment-protocol.md`](../experiment-protocol.md) — how to measure.

## 1. Why this file exists

The roster shares one ancestor. `expander_python` supplies the greedy capture
loop and `bots/expand_plus/agent.py` supplies the BFS march fallback. Ten of
the twelve bots were forked from that pair.

Two forces push the roster toward one policy:

1. **Shared ancestry.** Each new bot forks from `smoke` plus `expand_plus`, so
   every bot starts from the same action-selection core.
2. **Copying the leader.** After each round, the tempting move is to copy the
   top bot's tactic into every other bot. That raises the mean rating and
   destroys the roster.

A single-policy roster is worthless. Mirror matches draw, Elo stays flat, and
no experiment can separate cause from noise. The season before round 1 showed
the end state: 31 rated games, 0 wins, 0 losses, 31 draws, every bot at exactly
1500.0 Elo.

The roster is a **measurement instrument**, not a ladder. Its value is the
spread of behavior it covers, not the peak strength of one member.

## 2. What round 1 says about convergence

58 games, 24 decisive, draw rate 58.6%
([`../measurements/round1.md`](../measurements/round1.md)). Three readings bear
on diversity.

**The roster is separating, and diversity is why.** Round Elo runs from 1616.4
to 1453.5, a 163-point spread, after a season that produced a 0-point spread.
The three bots at the top own three different axes: logistics
(`army_convey`), force concentration (`late_rush`) and information
(`fog_scout`). No two of them share a differentiator.

**The economy cluster is the failure case this file exists to prevent.**
`castle_builder`, `castle_rush` and `phase_switch` all build castles from the
same cost model and the same rested-general funding step. Every game among them
drew at 1200 turns, and their round Elo sits inside 22 points: 1495.2, 1473.8,
1473.7. Three bots produced one measurement. Castle count and result also run
opposite — 4 castles gave 0 wins, 3 gave 0 wins, 2 gave 1 win, and all three
winning bots built none — so the cluster is not merely redundant, it is
redundant around a tactic that does not convert.

**The shared helper module is an active convergence risk.**
`bots/*/strategy_common.py` is duplicated in `expand_plus`, `castle_builder`
and `general_hunter`. It contains `chase_defence` (the `garrison` axis),
`BeliefState` with `probe_move` (the `fog_scout` axis), and `sentry_convey`
with `march_toward_frontier` (the `army_convey` axis). Three baseline bots now
carry three other bots' differentiators.
[`optimize-existing.md`](optimize-existing.md) §3.7 permits this only as defect
repair. [§4.4](#44-bounds-on-the-shared-helper-module) turns that permission
into numbers you can check.

## 3. Non-negotiable differentiators

Each bot owns exactly one differentiator. The differentiator is the reason the
bot exists. Remove it and the bot is deleted, not repaired.

| Bot | Non-negotiable differentiator | Must always be true | Must never be true |
| --- | --- | --- | --- |
| `smoke` | Frozen protocol floor. Minimum viable policy, no scoring, no memory, no map analysis. | Takes the first legal expansion move it finds. Holds no state between turns. `agent.py` never changes. | Any scoring weight, any BFS, any turn gate, any castle build, any belief state. |
| `expand_plus` | Land maximization with no investment and no targeting. | Every action serves land rate: greedy capture, else a march toward the nearest capturable tile. | Any castle build. Any policy that spends land tempo to chase a general. |
| `castle_builder` | Conservative economy. Few castles, floor price, full garrison. | Thicker surplus margin, longer cooldown, lower cap and an earlier build deadline than `castle_rush`, on every shared parameter. | Any parameter that reaches a `castle_rush` value. Any hunt behavior. |
| `castle_rush` | Aggressive economy. Same build mechanism, every knob shifted early and thin. | Earlier start, higher cap, shorter cooldown and thinner margin than `castle_builder`, on every shared parameter. | Any parameter that reaches a `castle_builder` value. Any hunt behavior. |
| `general_hunter` | Deathtouch execution reached by cheap scouting. Wins by touching a remembered general cell, not by army superiority. | Remembers the enemy general cell forever after one sighting. Sends split detachments at a candidate prior. The probe path executes at least once per game. | Massing one main stack (that is `late_rush`). Whole-map revelation (that is `fog_scout`). Any castle build. |
| `phase_switch` | Explicit hard phase clock. One `if/elif` on `obs.turn`, no blending. | Three disjoint turn windows, one policy each. Builds confined to the mid window. | Soft weights across phases. Hysteresis. Any behavior that runs in more than one window. |
| `fog_scout` | Information gain. Scores unknown map area above known land. | Keeps an `ever_seen` grid. Its fog bonus exceeds its opponent bonus. | A fog bonus at or below the opponent bonus. Any castle build. |
| `army_convey` | Interior-to-frontier logistics. Moves idle interior army forward instead of taking new land. | Runs a convey step whenever no frontier capture exists. Scores convey moves by stack size over distance to the frontier. | Replacing the convey step with the `expand_plus` march. Any castle build. |
| `garrison` | Defense first. Holds a sized reserve on the general before it expands. | Sizes the reserve from a threat model and refuses expansion that breaks it. | Spending the reserve for a capture. Any castle build. |
| `late_rush` | Single-stack commitment. Merges army into one main stack and commits it in a fixed window. | Chooses one rally cell and one main stack, then commits between turn 700 and turn 800. | Splitting the main stack for land. Committing before accumulation ends. |
| `splitter` | `split=1` as a policy. Half-army moves where they beat all-but-one. | Decides the split flag on every move and leaves a usable garrison on the source. | Falling back to `split=0` everywhere, which makes it `expand_plus`. |
| `choke_control` | Corridor control. Claims and holds narrow passages. | Detects chokes from local passability and holds an owned choke instead of draining it. | Draining a held choke cell for a sideways capture. Any castle build. |

`expander_python` is the external baseline in `competition-module`. Never
change it and never treat it as a roster member.

### 3.1 One axis per bot

| Axis | Owner |
| --- | --- |
| Protocol floor | `smoke` |
| Land rate | `expand_plus` |
| Economy, conservative | `castle_builder` |
| Economy, aggressive | `castle_rush` |
| Kill condition by scouting | `general_hunter` |
| Control structure, hard clock | `phase_switch` |
| Information | `fog_scout` |
| Logistics | `army_convey` |
| Defense | `garrison` |
| Force concentration | `late_rush` |
| Move granularity | `splitter` |
| Terrain | `choke_control` |

Every axis is taken. A new bot therefore either brings a new axis or replaces
the current owner of an existing one. It never runs beside the current owner.

### 3.2 The two pairs that need active separation

Two pairs sit close enough that they need a stated gap, not a judgement call.

**`castle_builder` and `castle_rush`.** The gap *is* the experiment: same
mechanism, opposite parameters. Round 1 shrank it — `castle_builder` moved to
cap 3 against `castle_rush`'s cap 4 — and the two drew every game 22 Elo
points apart. Required: a strict inequality on **every** shared build
parameter, in the conservative direction for `castle_builder` and the
aggressive direction for `castle_rush`. Equality on any one of them is a
violation.

**`general_hunter` and `late_rush`.** Both end in a deathtouch execution. The
difference is targeting, and only targeting. `general_hunter` locates the
general with cheap split probes against the spawn-distance prior and touches
with 2 army. `late_rush` masses one stack and walks it at a remembered or
guessed target. If a round 2 trace shows the two producing the same action
sequence, delete one.

## 4. Forbidden cross-copying

These transfers are forbidden, whatever the measured gain.

| Tactic | Owner | May not be copied into |
| --- | --- | --- |
| Castle build and build-funding | `castle_builder`, `castle_rush`, `phase_switch` (mid window only) | every other bot |
| Massing one main stack | `late_rush` | every other bot |
| Whole-map revelation, fog bonus, `ever_seen` scoring | `fog_scout` | every other bot |
| Interior convey scoring as a policy | `army_convey` | every other bot |
| Threat model and sized general reserve | `garrison` | every other bot |
| Choke detection and hold rules | `choke_control` | every other bot |
| `split=1` as a general move policy | `splitter` | every other bot |
| Hard turn-window `if/elif` control | `phase_switch` | every other bot |
| Candidate prior plus split probes | `general_hunter` | every other bot |

### 4.1 The three tests

Apply all three to every proposed change. A change that fails any test is
rejected.

1. **Identity test.** Read this bot's row in
   [§3](#3-non-negotiable-differentiators). Does the change keep the "must
   always be true" column true and the "must never be true" column false?
2. **Distance test.** After the change, does at least one behavior separate
   this bot from every other roster member on a shared seed? Two bots that
   produce the same action sequence on the same seed are one bot.
3. **Source test.** Where did the idea come from? An idea taken from a stronger
   roster member's differentiator is forbidden. An idea taken from `RULES.md`,
   from this bot's own measurements, or from a free axis is allowed.

### 4.2 What is always allowed

These are infrastructure, not tactics. Copy them freely.

- The stdio wire protocol in `main.py` and `run.sh`, and the §5.2 telemetry
  line.
- Passability checks: mountain (`2`) and fogged structure (`5`) are impassable.
- Bounds checks, the direction table, and the `PASS` fallback.
- The build-cost formula from `RULES.md` section 03, for bots that own the
  build tactic.
- Bug fixes: an out-of-range index, a fault, a timeout, or an inverted
  comparison. A bug fix is not a tactic transfer.
- Parameter values inside a bot's own axis, when
  [`optimize-existing.md`](optimize-existing.md) approves them.

### 4.3 Copying a *result* is allowed; copying a *tactic* is not

Round 1 shows that castle income does not convert into wins. Acting on that —
by cutting build deadlines, or by declining to add building to a bot that does
not have it — is reading a measurement. That is the point of the roster.

Adding `army_convey`'s convey scoring to `castle_builder` because
`army_convey` finished first is copying a tactic. That is forbidden, and it is
forbidden precisely when it looks most attractive.

### 4.4 Bounds on the shared helper module

`strategy_common.py` may hold defect repairs shared by `expand_plus`,
`castle_builder` and `general_hunter`. It may not hold another bot's identity.
Checkable limits:

| Helper | Bound | Violated when |
| --- | --- | --- |
| `general_reserve` | a flat floor from turn alone | it reads enemy army, distance or any threat signal — that is the `garrison` threat model |
| `chase_defence` | one adjacent-cell chase, at most one tick of lookahead | it plans interception paths or picks intercept cells in advance |
| `march_toward_frontier` | fallback only, used when no capture scores | it ranks moves against captures, or it scores by stack size over distance — that is `army_convey` |
| `sentry_convey` | one named cell next to the own general | it conveys toward a rally point or a main stack — that is `late_rush` |
| `BeliefState.candidates` | the spawn-distance prior, narrowed by `ever_seen` | it scores cells by revealed area, or drives capture choice — that is `fog_scout` |
| `probe_move` | at most 2 live detachments, 2–4 army each | the probe budget grows, or probes are used to take land |
| `split=1` | probes and the general reserve only | any other move emits `split=1` — that is `splitter` |

Any helper that exceeds its bound moves out of `strategy_common.py` and into
the bot that owns the axis.

## 5. Convergence alarms

Check after every round. Any alarm blocks the next round until it is cleared.

| Alarm | Threshold | Round 1 | Required response |
| --- | --- | --- | --- |
| Draw rate | above 80% of rated games | 58.6%, clear | Do not raise aggression across the roster. Change one bot's differentiator, and record which. |
| Cluster Elo | three or more bots inside 25 points over ≥ 10 games each | **firing** — economy cluster inside 22 points | The cluster is one bot in several copies. Widen §3.2 gaps, or delete a member. |
| Identical action traces | two bots match on more than 90% of turns on the same seed | not measured | Fail the distance test. Delete or re-specify one bot. |
| Elo spread, whole roster | all bots inside 25 points after 30+ rated games | 163 points, clear | Add a bot on a free axis. Do not tune the roster closer together. |
| Parameter drift | any shared build parameter equal between `castle_builder` and `castle_rush` | cap 3 against cap 4, thin | Restore a strict inequality on every shared parameter (§3.2). |
| Dead differentiator | a bot's differentiator never executes in a game | **firing** — `general_hunter`'s probe path is disabled by its active-probe test | Repair or delete. A bot whose differentiator never runs supplies a rating and no information. |
| Zero decisive rate | a bot neither wins nor loses across a full round | **firing** — `castle_builder`, 4 games, 4 draws | Give it a grid that includes the winning bots before drawing any conclusion about it. |
| Shared-loop share | more than 8 bots run the unmodified greedy capture loop | 10 of 12 | Freeze new forks from `expand_plus`. The next bot brings its own action-selection core. |

Four alarms are firing after round 1. Clear the two repair items — the dead
differentiator and the parameter drift — before round 2 opens. The cluster and
shared-loop alarms need a roster decision, not a parameter change.

## 6. Change record

Every bot change states, in its experiment note under
[`../experiments/`](../experiments/):

1. The bot's axis from [§3.1](#31-one-axis-per-bot).
2. The result of each of the three tests in [§4.1](#41-the-three-tests).
3. The source of the idea, per the source test.

An experiment note without these three items is incomplete, and the change does
not merge.
