# Plan: 95 wins in 100 games against human players

The remote end goal is one heuristic bot that wins **at least 95 of 100 logged
games against human opponents** on live generals.io through
`client/generals_client` (`arena/remote_bridge.py`, `scripts/remote_play.py`).

This file is the program plan for that goal. It holds no bot code. It is a
**classic generals.io** plan, not a competition plan. A remote result never
enters `data/ratings/`.

Read first:

- [`../../engine/remote-eval-heuristics.md`](../../engine/remote-eval-heuristics.md)
  — what remote play measures, and the log schema.
- [`../../engine/remote-play-setup.md`](../../engine/remote-play-setup.md)
  — env vars, CLI, and the current logger.
- [`../../../RULES.md`](../../../RULES.md) — the competition ruleset the roster
  was built for.
- [`diversity-constraints.md`](diversity-constraints.md) — the roster contract.

---

## 0. Status and the honest reading of the target

Two facts set the whole plan.

**Fact 1 — the target is a block criterion, not a rate.** "95 of 100" is a
count. A bot whose true win rate is exactly 95% passes a single 100-game block
only about 6 times in 10. The Wilson 95% lower bound on 95/100 is about
**88%**, so a passing block does not prove a true rate of 95%. To certify a
true rate of 95% at 95% confidence takes about **73 wins with zero losses**.
Report the block count as the mandate requires, and report the pooled estimate
next to it. Never present one passing block as a proven 95% win rate.

**Fact 2 — the roster was built for a different game.** Every bot in `bots/`
was specified against `RULES.md`. Classic generals.io removes the build action,
adds pre-placed neutral cities, removes deathtouch, and removes the 1200-turn
draw. Three of the twelve bots lose their whole plan, and one of the three top
bots does something actively harmful (§1.3). The champion is therefore a **new
remote-only bot**, not a promoted arena bot.

The transport layer is already built and offline-verified:
`arena/remote_adapter.py` and `scripts/remote_play.py` exist, and
`data/remote_games/` is the (gitignored) store. What is missing is result
fidelity (§5.4), opponent telemetry (§3.3), a classic-rules practice harness
(§4.2), and the champion itself (§2).

---

## 1. Classic against competition: the rule mismatch

### 1.1 Verified differences

| Rule | Competition (`mode="competition"`) | Classic generals.io | Source |
| --- | --- | --- | --- |
| Structures | none at start; players **build** (`pass=2`) | **cities pre-placed**, neutral, garrisoned | `env.py` `_MODE_PRESETS`, `strip_neutral_castles` |
| Build action | legal | **does not exist**; `_generate_action` returns `None` for a truthy `pass` | `generalsio_client.py:151` |
| Deathtouch | any execution onto the enemy general wins from turn 800 | **never**; a general capture always needs strictly more army | `RULES.md` §07 |
| Game end | draw at 1200 turns | elimination or surrender; **no draw** | `generalsio_client.py:178` handles only `game_won` / `game_lost` |
| Map | rectangle, sides 18–21, 24–26% mountains | varies by room; sizes and mountain density are not ours to choose | `env.py` preset |
| Turn unit | one tick; structures grow on even ticks, all land on `% 50` | **same cadence**; server ticks are the same half-turn unit | `game.py:262-279` |
| Turn offset | `turn` from the engine | `timestep = turn - 1` | `generalsio_state.py:67` |
| Terrain codes decoded | plain, mountain, castle, general | only `>= 0`, `-1`, `-2`, `-3`, `-4` are decoded | `generalsio_state.py:55-60` |
| Fog | on, vision radius 1 | on | both |

**One correction to the older note.** The turn *units* do match. Both clocks
count half-turns, and both grow structures every 2 ticks and all land every 50
ticks. What does not transfer is the **landmarks** (800, 1200) and the map
scale. A turn-keyed threshold is therefore not meaningless on classic; it is
simply calibrated to a board size and a game length that classic does not
share. Prefer land-keyed and army-keyed gates over turn-keyed gates in the
champion.

### 1.2 Cities are the largest single gap

Classic cities arrive as `type_grid == 3` with `owner_grid == 0` and a
garrison in `army_grid`. No bot in the roster has a city policy, because the
competition strips neutral castles. `late_rush` scores `dest_type == 3` at 20
only when `owner == 2`; a neutral city scores as plain land, and the universal
`src_army > dest_army + 1` guard then blocks the capture until a stack happens
to exceed the garrison by accident.

The result is predictable: the bot expands, never buys a city, and loses the
mid-game income race to any human who takes one or two. A city policy is not
an optimization for the champion. It is a precondition.

Required decisions the champion must make explicitly:

1. Whether to take a city at all, from the current land count and the garrison.
2. Which city, from distance, garrison size, and exposure to the opponent.
3. How to fund the attack, since a city needs one stack larger than the
   garrison, not many small ones.
4. When to stop buying and convert income into an attack.

The exact garrison range must be **measured from live observations**, not
copied from lore, and recorded in the first block report.

### 1.3 Deathtouch code is an active loss, not dead code

`bots/late_rush/agent.py` sets `DEATHTOUCH_TURN = 800`. From server turn 800
its `_contact_moves` branch scores any 2-army move onto the remembered enemy
general at 1,000,000 and returns it. On classic that move meets a defended
general and is destroyed. The branch does not degrade gracefully; it feeds the
stack to the opponent, every turn, for the rest of the game.

`general_hunter` and `phase_switch` carry the same class of assumption. The
adapter's build rewrite already neutralizes `castle_builder`, `castle_rush`
and `phase_switch` builds, but nothing neutralizes a deathtouch branch. Any
remote bot must contain **zero** references to turn 800, turn 1200, or a
one-unit lethal touch.

### 1.4 Adapter behavior worth knowing before tuning

| Item | Current behavior | Effect on the champion |
| --- | --- | --- |
| Build actions | rewritten to `pass`, counted in `builds_dropped` | correct; champion should never emit `pass=2`, so the counter must stay 0 |
| `structures_in_fog` (`-4`) | mapped to type `5`, which every bot treats as impassable | conservative and safe for legality, but BFS routes are pessimistic on classic, where `-4` is common; a routing-only relaxation is worth testing |
| Unknown terrain codes | not decoded; the cell falls through to plain-neutral | only classic 1v1 rooms are safe; swamps, deserts and lookouts would be misread as free land |
| `player_id` | fixed at `0`; perspective is pre-applied | correct, no change needed |
| Board size | read once from the first observation | classic boards do not resize mid-game, so this is safe |
| Per-move budget | not enforced remotely | the server does not wait; a slow turn is a **missed move**, which is worse than a competition fault |

---

## 2. Champion selection

### 2.1 Screening the current roster

Round 1 arena Elo ([`../measurements/round1.md`](../measurements/round1.md)):
`army_convey` 1616, `fog_scout` 1580, `late_rush` 1576. Applying the §1 filter:

| Bot | Build logic | Deathtouch or 1200 clock | City policy | Kill conversion | Remote verdict |
| --- | --- | --- | --- | --- | --- |
| `army_convey` | none | none | none | none | **cleanest base**; expands and conveys, never converts |
| `fog_scout` | none | none | none | direct general attack when sighted | good information behavior; fog bonus overspends on a larger board |
| `late_rush` | none | **both** | none | strongest in the roster | unusable as-is (§1.3) |
| `expand_plus` | none | none | none | none | baseline only |
| `castle_builder`, `castle_rush`, `phase_switch` | yes | yes | — | — | excluded |
| `general_hunter` | none | deathtouch-dependent | none | deathtouch only | excluded |
| others | none | none | none | none | not competitive |

No single roster bot is a credible champion. `army_convey` has the best
measured expansion engine and zero rule coupling, but it has no city policy,
no defense of its own general, and no way to finish a game.

### 2.2 Decision

Build a **new remote-only bot**, `bots/classic_duel/`, on three pillars:

1. **Expansion engine** — the interior-to-frontier logistics idea that
   `army_convey` measures best in the arena.
2. **City policy** — new, classic-only, per §1.2.
3. **Kill conversion and general defense** — new; a classic win needs a stack
   strictly larger than the defending general, plus a reserve at home so the
   opponent's stack does not win the same race first.

### 2.3 Roster carve-out (required before any code)

[`diversity-constraints.md`](diversity-constraints.md) §3.1 states every axis is
taken and a new bot must bring a new axis or replace an owner. The champion
does neither, because it plays a different ruleset. Resolve this explicitly
rather than by exception:

- `classic_duel` is **remote-only**. It never enters `arena/tournament.py`
  fields, never appears in `data/games/`, and never receives an Elo rating.
- It is exempt from the cross-copying table because it is not a roster
  measurement instrument.
- Nothing may flow **back** from `classic_duel` into a roster bot without the
  three tests in `diversity-constraints.md` §4.1.

Record this carve-out in `diversity-constraints.md` before the bot is written.

---

## 3. Measurement protocol

### 3.1 Endpoint and opponent mix

Play on the **bot endpoint** `https://botws.generals.io/`, the client default,
with the `[Bot]` username prefix. The arena has no flag that switches endpoints:
`resolve_server_url()` returns the bot endpoint unless you pass an explicit
`--server-url`. Do not point it at the human endpoint; that is for humans, and
running a bot there breaks the site convention and is the fastest way to lose
access.

The consequence must be stated up front: the bot endpoint's 1v1 queue holds
both bots and humans. **Only games against human opponents count toward the
100.** Bot-versus-bot games are logged and reported separately as a control.

### 3.2 The three room modes and what each proves

| Mode | Command | Proves | Counts toward 100 |
| --- | --- | --- | --- |
| Private lobby against a known human | `--mode lobby` | the adapter works end to end; a controlled first human game | yes, flagged `lobby` |
| Public 1v1 queue | `--mode 1v1` | the headline claim | yes, when the opponent is human |
| Public 1v1 against another bot | same | fault rate, latency, endurance | no |

Report the lobby share of the 100. A block that is mostly lobby games against
one cooperative opponent is not the same evidence as a queue block.

### 3.3 Opponent classification and telemetry

`opponent_is_bot` currently comes only from the `[Bot]` username prefix, which
is a convention and not enforced. Strengthen it with two more signals:

1. **`stars`** — `GeneralsIOstate.update` already parses `data["stars"]` and
   stores it, and the logger does not record it. Log
   `opponent_stars` at the first `game_update`. This is the only opponent
   strength signal available, and it makes the win rate readable by rating
   band.
2. **Replay review** — every game that decides the 95 threshold gets its
   `replay_id` opened and the opponent confirmed by hand.

Fields to add to the per-game JSON, on top of the existing schema:

| Field | Why |
| --- | --- |
| `opponent_stars` | the only opponent-strength control that exists |
| `endpoint` | proves the bot endpoint was used |
| `result_reason` | `game_won`, `game_lost`, `disconnect`, `receive_error` — see §5.4 |
| `map_height`, `map_width` | classic board size is not ours to choose; it is a confound |
| `neutral_cities_seen`, `first_city_capture_turn` | scores the §1.2 city policy |
| `land_at_turn_100`, `land_at_turn_200` | the transferable expansion metric |
| `max_move_latency_ms` | a slow turn is a missed move (§1.4) |

### 3.4 Blocks, and the stop / go ladder

A **block** is 100 human games played by **one immutable bot commit**. Changing
the bot ends the block. Never pool two commits into one block; `bot_commit` is
already logged, so this is checkable after the fact.

Inside a block, run these gates in order. Each gate is a stop-or-continue
decision, not a target.

| Gate | After | Continue if | Otherwise |
| --- | --- | --- | --- |
| A — legality | 10 games | 0 adapter exceptions, `builds_dropped == 0`, `faults == 0`, no missed-move stall | stop; the bot is not fit to measure |
| B — viability | 30 games | at least 21 wins | stop; a 70% bot cannot reach 95 without a design change |
| C — approach | 60 games | at least 51 wins | stop; tune, then open a new block |
| D — block result | 100 games | at least 95 wins | record the failed block; do not retry the same commit and cherry-pick |

**Hard early-stop rule.** 95 of 100 allows exactly five losses. At the **sixth
loss** the block is mathematically dead. Stop immediately, diagnose from the
replays, and open a new block with a new commit. Continuing a dead block only
burns queue time.

### 3.5 Reporting

Per block, publish `docs/research/measurements/remote-block<N>.md` with:

1. The block count (`wins / 100`) and the pooled win rate across all blocks.
2. The Wilson 95% lower bound next to both. Never quote the count alone.
3. Win rate split by `opponent_stars` band, and the count of unrated opponents.
4. Lobby share, bot-opponent control games, faults, missed moves, and latency.
5. The `replay_id` of every loss, with a one-line cause.

Raw game JSON stays in `data/remote_games/` and stays out of git. The block
report is committed.

---

## 4. Iteration loop

### 4.1 Why live games cannot be the iteration loop

A classic 1v1 game runs minutes of wall clock, cannot be seeded, cannot be
repeated, and gives one bit of outcome against an uncontrolled opponent. One
100-game block is hours of attended play. Tuning a threshold directly against
live games is the slowest and least reliable loop available.

### 4.2 The classic-approximate local harness (highest-value item)

`GeneralsEnv` already supports the classic ruleset through its constructor. No
submodule edit is needed:

| Setting | Value | Effect |
| --- | --- | --- |
| `build_castles` | `False` | no build action, and neutral castles are **not** stripped |
| `deathtouch_turn` | `None` | a general capture needs strictly more army |
| `truncation` | large | classic has no draw cap |
| `num_castles_range` | classic-like | pre-placed neutral cities to practice §1.2 against |
| `castle_val_range` | set from measured live garrisons | the city price the champion must plan for |
| grid size | wider than 18–21 | classic board sizes vary |

`matchup.py` cannot express this from the CLI — it only accepts `--mode`,
`--grid-size`, `--truncation` and `--perfect-info`. The harness is therefore a
repo-side wrapper under `arena/`, which is the "wrap and document" rule, not a
submodule edit. It reuses the existing stdio loop, so bots stay unchanged.

This converts an unrepeatable hours-long live block into a seeded grid that
runs in minutes, and it is the only place where a threshold should ever be
tuned.

### 4.3 Proxy metrics that plausibly predict human strength

Rank a change on the local classic harness first, then confirm on live games.

| Proxy | Read on | Predicts | Falsified if |
| --- | --- | --- | --- |
| Land at turn 100 and 200 | harness and live | expansion efficiency, the skill humans exploit first | a bot leads on land and still loses the block |
| First city capture turn, cities held | harness and live | mid-game income, the §1.2 gap | city count rises and win rate does not |
| Army per land at turn 300 | harness | whether income is being converted or hoarded | — |
| Turn of first enemy-general sighting | harness and live | the kill condition; you cannot win what you cannot find | — |
| Own-general survival against a rush | harness | the defensive hole `garrison` failed to fill in round 1 | — |
| Max move latency | live | missed moves (§1.4) | — |

Every proxy is confirmed or rejected against block outcomes. A proxy that does
not move with the block result is dropped from the list, not defended.

### 4.4 When to spend an expensive Think pass, and when to use Composer

| Work | Model | Why |
| --- | --- | --- |
| Reading a block's losing replays and forming one hypothesis | **Think** | the decision is judgement over messy evidence, and it sets the next block |
| Champion design and revision of the city or kill policy | **Think** | the choice is structural, and it is expensive to unwind |
| Stop / go at gates B, C and D | **Think** | mis-stopping wastes a block; mis-continuing wastes hours |
| Rule-mismatch analysis when a new room type appears | **Think** | a wrong reading silently corrupts every later measurement |
| Adapter, logger and harness code | Composer | specified work with a verification step |
| Parameter sweeps on the local harness | Composer | mechanical, and bounded by the spec |
| Running blocks and generating block reports | Composer | mechanical |
| Doc synchronization | Composer | mechanical |

Rule of thumb: **one Think pass per block boundary; Composer inside a block.**
Existing skills carry this split already — `write-strategy-spec` and
`check-bot-diversity` are Think, `build-bot-from-spec`,
`run-measurement-round` and `tune-bot-parameters` are Composer.

---

## 5. Risks

### 5.1 Credentials

- Live play uses `generals_client` (EIO v4) with `GENERALS_USER_ID`. No
  `bot_key` is required on the bot websocket endpoint.
- `GENERALS_USER_ID` is a secret the operator invents and is bound to a
  username once. It is a password. Environment only, never a CLI flag, never
  committed. `.env` and `.env.agent` are already gitignored.

### 5.2 Queue time and attendance

Each game is minutes long, and queue waits are not controlled by us. A 100-game
block is a multi-hour attended session, plausibly several. Sessions must be
bounded and supervised; unattended ladder farming is out of scope for this
plan and out of line with site convention.

### 5.3 Ethics and site convention

- Bot endpoint only. `[Bot]` username prefix, always.
- Bounded sessions, not indefinite `autopilot`.
- No play on the human endpoint.
- Human opponents are people spending their time. Do not requeue against the
  same opponent to farm a favourable record, and do not count such games as
  independent samples.

### 5.4 A result-fidelity defect that would corrupt the headline number

The repo-side client in `arena/remote_client.py` (`FidelityRemoteSession`) must
never count disconnects or malformed frames as wins. `scripts/remote_play.py`
routes live play through `arena/remote_bridge.UnifiedBot`, not the legacy
competition-module remote client.

Verify with `--mode dry-run` and the offline checks in
`arena/remote_adapter.verify_adapter_offline` before the first counted game.
Any `result_reason` other than `game_won` / `game_lost` is excluded from the
block count.

### 5.5 Sample size and opponent mix

- 95/100 has a Wilson 95% lower bound near 88%. State it every time (§0).
- Opponent strength is uncontrolled and mostly unrated. A 95/100 against
  unrated new accounts is a weaker claim than the same count against a rated
  band, and the report must let a reader tell the two apart.
- Repeated opponents are correlated samples. Log opponent usernames and report
  the number of distinct opponents in the block.
- Room variants that add swamps, deserts or lookouts are misdecoded by the
  state translator (§1.4). Restrict to classic 1v1 rooms and log the room type.

### 5.6 Version drift

A block is one commit. Any edit — including a "harmless" logging change that
touches the decision path — ends the block. `bot_commit` is logged; audit it
when the block closes.

---

## 6. Immediate engineering tasks, in order

| # | Task | Blocks what | Done when |
| --- | --- | --- | --- |
| 1 | Fix result fidelity in the repo-side client subclass: `disconnect` and `receive_error` are not wins, add a receive timeout, add `result_reason` | every counted game | a simulated malformed event produces a non-win record |
| 2 | Add credential and telemetry plumbing: `GENERALS_BOT_KEY` override, `endpoint`, `opponent_stars`, map size, latency, city fields | opponent classification and §3.5 | a dry run shows every §3.3 field present |
| 3 | Build the classic-approximate local harness under `arena/` (`build_castles=False`, `deathtouch_turn=None`, no truncation cap, neutral cities on) | all cheap iteration | a seeded classic game finishes between two existing bots |
| 4 | Write the `classic_duel` strategy spec (city policy, kill conversion, general defense, no turn-800 or turn-1200 logic) and add the §2.3 carve-out to `diversity-constraints.md` | the champion | spec has named thresholds and one falsifiable hypothesis |
| 5 | Obtain the generals.io bot key and run gate A: 10 lobby games, then 10 bot-opponent queue games, fault-free | block 1 | gate A passes with 0 faults, 0 builds dropped, 0 missed moves |
| 6 | Implement `classic_duel` from the spec; tune only on the task-3 harness | block 1 | it beats `army_convey` on the classic harness over a seeded grid, both seat orders |
| 7 | Open block 1 and run the §3.4 gate ladder | the mandate | `remote-block1.md` published with the count, pooled rate and Wilson bound |

---

## 7. Related

- [`../../engine/remote-eval-heuristics.md`](../../engine/remote-eval-heuristics.md)
  — log schema and why remote results stay out of Elo
- [`../../engine/remote-play-setup.md`](../../engine/remote-play-setup.md) —
  environment variables and the CLI
- [`tournament-plan.md`](tournament-plan.md) — the competition-side rounds
- [`diversity-constraints.md`](diversity-constraints.md) — the roster contract
  and the §2.3 carve-out
- [`../experiment-protocol.md`](../experiment-protocol.md) — per-change protocol
