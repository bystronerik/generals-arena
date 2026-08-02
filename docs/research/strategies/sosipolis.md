# sosipolis strategy specification

## Goal

`sosipolis` finds the enemy general early, then captures it. The bot keeps
persistent fog memory and runs purpose MCTS (Search / Contact / Strike) under a
**hard Kubic priority shell** and a **mod-50 gather/wave clock**. MCTS only
chooses UNKNOWN destinations and path steps inside chain and clock masks.

The bot must use the competition observation and the five-integer stdio action.
The bot must not use hidden engine state. The bot must finish each decision
inside the 150 ms move limit and must keep a 50 ms reserve (100 ms hard cap).

## Diversity claim

Sosipolis is a **research bot**, not a competition-roster axis owner. It does
not replace `fog_scout`, `general_hunter`, or `castle_builder`.

| Axis | Value |
| --- | --- |
| Primary objective | general kill (find-and-strike on a Kubic conveyor) |
| Risk posture | aggressive expand, rare recall only |
| Time profile | opening ≤50, contact ~82 target, castles ≥116, strike after sight |
| Information use | persistent fog memory + section priors + contact hunt |
| Army handling | one chain; gather drain leave-1; wave toward objective |

**Differentiator:** clocked purpose MCTS (Search / Contact / Strike) with a
Kubic §2 hard shell (opening, rare recall, castle earliest 116, tip floor).

**Must always be true:** SearchMCTS before enemy land; ContactMCTS after enemy
land and before general sight; StrikeMCTS after first sighting. Chain head is
the preferred source when army ≥ 2. Gather residues stay on own land; wave
closes on the persistent objective.

**Must never be true:** Pure rule conveyor as the live policy. Standing home
garrison that fights scheduled drain. Castles before turn 116. Averaged
UNKNOWN scorers (center vs nearest-unseen). Imports from `bots/_common`.

Verdict vs existing bots: **distinct** (research axis).

## Strategy features

1. Latch own general from `owner == 1` and `type == 4`.
2. Merge every visible cell into persistent memory (terrain, ownership).
3. Latch enemy general forever after first sight.
4. Maintain candidate cells at BFS distance ≥ `MIN_GENERAL_DISTANCE`.
5. Mark mountain-enclosed regions of size ≤ `POCKET_MAX_CELLS` as dead pockets.
6. Partition the board into sections and keep prior mass over candidates.
7. Hard Kubic priority each tick (see Candidate move rules).
8. Mod-50 clock masks every Search/Contact/Strike root set.
9. One active chain; continue when prev destination still has army ≥ 2.
10. Castles from `CASTLE_START_TURN` (116) on frontier sites, up to `CASTLE_MAX`.
11. First-move grace precomputes pockets and section priors.
12. Defense is rare tip recall (`RECALL_PROX_D`), not a fourth MCTS.

## State

- `own_general` / `enemy_general` / fog memory / candidates / dead pockets /
  section priors (unchanged MapMemory + sections).
- `phase`: `search` | `contact` | `strike` (info axis).
- `clock_phase`: `gather` | `wave` (mod-50).
- `chain_head`, `last_out_dir`, `objective`, `muster`.
- `strike_tip` / `strike_tip_turn`, `build_site`, `castles_owned`.
- `home_threat_dist`, `land_at_50`, `recall_fired` (probes).

## Named thresholds

- `LATENCY_CAP_MS` = 100
- `GATHER_BUDGET_MS` = 40 / `WAVE_BUDGET_MS` = 80 / `STRIKE_MARCH_BUDGET_MS` = 55
- `GATHER_PHASE_LO/HI` = 10 / 27
- `OPEN_END` = 50 / `OPEN_FLOOD_START` = 27 / `OPEN_PULSE_TICKS` = {3,6,9}
- `RECALL_PROX_D` = 3 (UNKNOWN candidate)
- `TIP_AT_SIGHT_FLOOR` = 10 / `STRIKE_MIN_TIP` = 23 / `STRIKE_TOWARD_BIAS` = 0.93
- `CASTLE_START_TURN` = 116 / `CASTLE_MAX` = 4 / `CASTLE_ABORT_TURN` = 900
- `CASTLE_SPACING` = 7 / `HOME_BANK_*` = 1
- `MIN_GENERAL_DISTANCE` = 17 / `TARGET_CONTACT_TURN` = 82 (probe only)

## Candidate move rules

1. If adjacent to a known enemy general and capture is legal, take it.
2. If no owned cell has army ≥ 2 with a passable neighbour, PASS.
3. Rare recall when `d_home ≤ RECALL_PROX_D` (imminent adjacent loss first).
4. If `t ≤ OPEN_END`: opening tempo + SearchMCTS under opening mask.
5. Else if castle conditions hold: build or gather to site (≥116, not strike).
6. Else if strike and tip army < `TIP_AT_SIGHT_FLOOR`: exclusive tip feed.
7. Else MOVE by `t%50` mask + info-phase MCTS (Search / Contact / Strike).
8. Fallback: PASS.

## Pseudocode for act()

```text
FUNCTION act(obs):
    state.update(obs)   # phase, clock, objective, muster
    IF kill_shot: RETURN kill
    IF NOT has_leave1_move: RETURN PASS
    IF imminent_loss OR recall_move: RETURN defense
    IF t <= OPEN_END: RETURN decide_opening (SearchMCTS + opening mask)
    castle = economy.decide(...)
    IF castle: RETURN castle
    IF strike AND tip < TIP_AT_SIGHT_FLOOR: RETURN feed_tip
    clock = gather|wave from t%50
    RETURN MCTS[phase].search(..., clock masks, chain-first)
    # after commit: update_chain(state, action)
```

## Experiment hypothesis

Clocked shell + purpose MCTS raises land@50 into [20,25], keeps castles ≥116,
raises chain-continue rate, and improves win rate vs the unclocked tip5 baseline
against `macaria` on a fixed seed grid — without collapsing pass rate.

Kubic calibration (observational, spend-detector): contact 82, sight→kill ~20–24,
first real castle ≥116 (operating 116–150), tip@sight floor 10 / operate ~23.
Scraped games never enter `data/games/` or ratings.

## Layout

```text
bots/sosipolis/
  run.sh, main.py, stdio.py, state.py, brain.py, params.py, probe.py
  components/{map_memory,pockets,sections,search_mcts,contact_mcts,
              strike_mcts,economy,threat,tip,army,clock,conveyor,opening}.py
```

No `bots/_common` imports.
