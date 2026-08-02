# sosipolis strategy specification

## Goal

`sosipolis` finds the enemy general as early as possible, builds a small early
castle economy, then captures that general. The bot keeps persistent fog
memory, skips mountain-enclosed dead pockets, and biases exploration with
board-section priors. After first enemy land contact it runs a directed
ContactMCTS hunt. When the general is known, StrikeMCTS finds the cheapest
path to kill.

The bot must use the competition observation and the five-integer stdio action.
The bot must not use hidden engine state. The bot must finish each decision
inside the 150 ms move limit and must keep a 50 ms reserve (100 ms hard cap).

## Diversity claim

Sosipolis is a **research bot**, not a competition-roster axis owner. It does
not replace `fog_scout`, `general_hunter`, or `castle_builder`.

| Axis | Value |
| --- | --- |
| Primary objective | general kill (early find-and-strike) |
| Risk posture | aggressive |
| Time profile | early castle, early contact, mid sight, immediate strike |
| Information use | persistent fog memory + section priors + contact footprint |
| Army handling | early castle funding, then gather for strike path |

**Differentiator:** triple-mode MCTS (SearchMCTS / ContactMCTS / StrikeMCTS)
with mountain-pocket skip, geometric section priors, and a Kubic-seeded early
castle programme.

**Must always be true:** SearchMCTS runs before any enemy land is known.
ContactMCTS runs after enemy land is remembered and before the general is
sighted. StrikeMCTS runs after the first sighting. Dead pockets are skipped
unless they hold enemy land or contact-sector candidates. Persistent memory
retains ever-seen terrain and ownership under fog.

**Must never be true:** `fog_scout`-style fog bonus greater than opponent bonus
as the expansion score. `general_hunter`-style split probes with
deathtouch-only kill. Imports from `bots/_common` or other bots. Stealing the
roster `castle_builder` differentiator (thick conservative multi-castle
economy) — Sosipolis keeps a thin early programme only.

Verdict vs existing bots: **distinct** (research axis).

## Strategy features

1. Latch own general from `owner == 1` and `type == 4`.
2. Merge every visible cell into persistent memory (terrain, ownership).
3. Latch enemy general forever after first sight (`type == 4`, `owner == 2`).
4. Maintain candidate cells at BFS distance ≥ `MIN_GENERAL_DISTANCE` from own
   general; prune cells that were seen and are empty of a general.
5. Mark mountain-enclosed regions of size ≤ `POCKET_MAX_CELLS` as dead pockets.
6. Partition the board into `SECTION_ROWS` × `SECTION_COLS` sections and keep a
   prior mass over sections that still hold candidates.
7. Before enemy land is known: SearchMCTS expands on a wide front with mild
   section prior and pocket skip.
8. After enemy land is known and the general is unknown: ContactMCTS sharpens
   priors with `CONTACT_SECTOR_FOCUS` toward the contact footprint and hunts
   that sector.
9. After the enemy general is known: StrikeMCTS gathers and advances; no new
   fog exploration and no new castle projects.
10. From `CASTLE_START_TURN`, fund and build up to `CASTLE_MAX` castles on the
    cheapest safe owned plain near the general.
11. Use the first-move 10 s grace (up to `FIRST_MOVE_GRACE_MS`) to precompute
    pockets, sections, and root priors.

## State

- `own_general`: fixed own general cell.
- `enemy_general`: latched enemy general cell or null.
- `ever_seen`: boolean grid of cells ever in vision.
- `known_type`: last known type for each ever-seen cell.
- `known_owner`: last known owner for each ever-seen cell (retained in fog).
- `candidates`: set of remaining possible enemy-general cells.
- `dead_pockets`: set of cells in skip-expand regions.
- `section_prior`: mass per section index.
- `first_contact_cell`: first enemy-owned cell ever seen.
- `enemy_land_known`: true once any enemy ownership is remembered.
- `build_site`: current castle funding cell or null.
- `castles_owned`: count of owned castles.
- `phase`: `search` | `contact` | `strike`.

## Phases and thresholds

Named constants (seeded from Kubic aggregates and RULES.md):

- `LATENCY_CAP_MS` = 100
- `SEARCH_BUDGET_MS` = 70
- `CONTACT_BUDGET_MS` = 75
- `STRIKE_BUDGET_MS` = 80
- `FIRST_MOVE_GRACE_MS` = 9000
- `MIN_GENERAL_DISTANCE` = 17
- `SECTION_ROWS` = 3
- `SECTION_COLS` = 3
- `POCKET_MAX_CELLS` = 12
- `CONTACT_REWEIGHT` = 0.80
- `CONTACT_SECTOR_FOCUS` = 0.95
- `TARGET_CONTACT_TURN` = 82
- `TARGET_SIGHT_TURN` = 182
- `STRIKE_TOWARD_BIAS` = 0.84
- `GATHER_WAVE_HINT` = 4
- `DEATHTOUCH_TURN` = 800
- `FINISH_MARGIN` = 2
- `MCTS_C` = 1.2
- `MCTS_MAX_ROOT` = 12
- `MCTS_ROLLOUT_DEPTH` = 8
- `CASTLE_MAX` = 1
- `CASTLE_START_TURN` = 10
- `CASTLE_KEEP` = 3
- `CASTLE_MIN_LAND` = 5
- `CASTLE_ENEMY_CLEAR` = 5
- `CASTLE_ABORT_TURN` = 100
- `BUILD_BASE_COST` = 35

## Threat or scoring model

**Search reward:** land gained + mild section prior − dead-pocket / thin-corridor
penalty.

**Contact reward:** candidate prune expected in the contact sector + closing on
the sector centroid + pressure captures − corridor penalty outside the sector.

**Strike cost:** BFS distance to the latched general weighted by required army
(`defender + FINISH_MARGIN`, or 2 after `DEATHTOUCH_TURN`) and gather turns.

**Castle cost:** `35 + sum(max(0, 14 - 2 * Manhattan))` over own structures.

## Candidate move rules

1. If adjacent to a known enemy general and capture is legal, take it.
2. If own general is under immediate visible threat, defend.
3. Else if a funded castle site exists and castles `< CASTLE_MAX`, build or
   gather to the site (not in `strike` for new projects).
4. Else if `enemy_general` is known: StrikeMCTS under `STRIKE_BUDGET_MS`.
5. Else if enemy land is known: ContactMCTS under `CONTACT_BUDGET_MS`.
6. Else: SearchMCTS under `SEARCH_BUDGET_MS`.
7. Fallback: PASS.

## Pseudocode for act()

```text
FUNCTION act(obs):
    start = monotonic()
    deadline = start + LATENCY_CAP_MS / 1000
    IF first move:
        deadline = start + FIRST_MOVE_GRACE_MS / 1000
    state.update(obs)
    IF first move:
        state.precompute_pockets_and_sections(deadline)
    IF kill_shot_available(obs, state):
        RETURN kill_shot
    IF home_under_immediate_threat(obs, state):
        RETURN defense_move
    castle_move = economy.decide(obs, state, deadline)
    IF castle_move is not null:
        RETURN castle_move
    IF state.phase == strike:
        move = StrikeMCTS.search(...)
    ELSE IF state.phase == contact:
        move = ContactMCTS.search(...)
    ELSE:
        move = SearchMCTS.search(...)
    IF move is null:
        move = PASS
    RETURN move
```

## Edge cases

- Mountain pad cells beyond the true map stay impassable; sections use `H` and
  `W` from the handshake.
- Fog army is unknown; searches do not invent fog armies.
- Mutual capture and chase-cancel draws follow RULES.md.
- If all candidates are pruned without a sighting, rebuild candidates from
  remaining never-seen passable cells at distance ≥ 17.
- Invalid builds are a silent pass; the bot must only emit funded legal sites.

## Expected behavior against existing bots

- Versus `smoke`: win or early capture after the castle footing and search.
- Versus `fog_scout`: ContactMCTS should close sight timing.
- Versus `macaria`: castles plus directed contact hunt should raise survival
  and conversion versus the dual-MCTS baseline.

## Experiment hypothesis

On seeds 0–19 vs `macaria` (both seats), Sosipolis with ContactMCTS + early
castles raises win rate versus the pre-change content hash, and median
first-sight turn moves toward Kubic’s ~182 (or below the prior loss band),
while per-move wall time stays ≤ 100 ms.

Kubic calibration (observational): contact 82, sight 182, toward 84%, first
castle tick median 10, median castles 1 (76% ≥1). Scraped games never enter
`data/games/` or ratings.

## Layout

```text
bots/sosipolis/
  run.sh, main.py, stdio.py, state.py, brain.py, params.py, probe.py
  components/{map_memory,pockets,sections,search_mcts,contact_mcts,
              strike_mcts,economy,army,clock}.py
```

No `bots/_common` imports.
