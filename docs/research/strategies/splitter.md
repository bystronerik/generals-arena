# splitter — split-move strategy spec

Status: **implemented** at `bots/splitter/` (fork of `bots/expand_plus/`;
`main.py` / `run.sh` protocol intact). Measured in round 1 — see
`## Parameter revision 1` below.
Baseline: `expand_plus` (greedy capture + BFS frontier march, `split=0`
everywhere). One change only: a split-flag decision layer plus the scoring
adjustments it needs.

## Goal

Use split moves (`split=1`, half army) where they beat the default
all-but-one move (`split=0`):

1. **Multi-front pressure** — one large stack feeds two expansion
   directions over consecutive turns instead of committing its whole army
   to a single direction.
2. **Efficient capture** — stop draining a 100-army stack onto a cheap
   neutral tile; send what is needed, keep a useful remainder.
3. **Garrison the source** — when a frontier stack advances, leave half
   behind so the source cell is not recaptured by any adjacent enemy stack
   of 2+.

## Engine semantics (exact, from `generals/core/game.py`)

- `split=0`: moves `army - 1`, leaves `1`.
- `split=1`: moves `army // 2` (floor), leaves `army - army // 2` (ceil).
- Capture requires moved army **strictly greater** than defender army; an
  exact tie keeps the defender (RULES.md section 05). Neutral plains are
  captured by any moved army larger than the neutral garrison shown in
  `army_grid`.
- Moving onto an own cell merges; no combat.
- A move is still one source to one destination. "Multi-front" pressure
  emerges over turns: this turn the stack pushes east with half, next turn
  the remainder pushes north.
- Move-order priority (RULES.md section 02): chasing > reinforcing >
  **smaller army first**. A split move halves the moving army, so it tends
  to resolve earlier among same-priority moves. Do not exploit this in v1;
  record it as a known interaction.
- Invalid moves are silent passes. A split that arrives with `<=` defender
  army is not invalid — it is a losing trade and the moved army dies. The
  scorer must never propose that as a "capture".

## Features (per candidate move)

All read from the observation (`owner_grid`, `type_grid`, `army_grid`,
`turn`, scalars) — no hidden state beyond the expand_plus BFS field.

| Feature | Meaning |
| --- | --- |
| `A` | army on the source cell |
| `D` | visible army on the destination |
| `need` | `D + 1`, minimum moved army that captures |
| `extra_fronts(s)` | count of *other* capturable tiles adjacent to source `s` (0–3) |
| `contested(s)` | 1 if `s` borders an opponent cell or a fog edge, else 0 (garrison value) |
| `overkill(A, D)` | ratio `A / need` — how oversized the stack is for this capture |
| `is_general(s)` | source is our general tile (type 4) |
| `frontier_dist` | expand_plus multi-source BFS distance field to capturable tiles |

## Scoring

Score every candidate `(move, split)` pair; keep the best.

```
base(moved, dest) = moved * 10 * (2 if dest is opponent-owned else 1)

score(split=0) = base(A - 1, dest)                      # expand_plus default
score(split=1) = base(A // 2, dest) * split_bonus(s, dest)

split_bonus(s, dest) = 1
    + W_SECOND_FRONT * extra_fronts(s)
    + W_GARRISON     * contested(s)
    + W_OVERKILL     * (1 if overkill >= OVERKILL_MIN else 0)
```

Initial constants (tune by experiment, do not treat as validated):

| Constant | Value | Guard |
| --- | --- | --- |
| `SPLIT_MIN_ARMY` | 8 | never split a source below this; crumbs result |
| `SPLIT_MARGIN` | 2 | split capture requires `A // 2 >= need + SPLIT_MARGIN` |
| `OVERKILL_MIN` | 4 | overkill ratio that triggers `W_OVERKILL` |
| `W_SECOND_FRONT` | 0.5 | per extra capturable direction from the source |
| `W_GARRISON` | 0.5 | source borders opponent or fog frontier |
| `W_OVERKILL` | 0.25 | stack heavily oversized for the target |

Tie-break: exact score ties resolve to `split=0` (commit; safer against
visible opponent stacks).

## When to use `split=1` vs `split=0`

**Use `split=1` when all hold:**

- `A >= SPLIT_MIN_ARMY` and `A // 2 >= need + SPLIT_MARGIN` (half captures
  with margin), and
- at least one of:
  - `extra_fronts(s) >= 1` — the remainder has an immediate second target
    next turn (true multi-front),
  - `contested(s)` — the source sits on the frontier and the leftover 1
    from a `split=0` move would be recaptured,
  - `overkill >= OVERKILL_MIN` and destination is neutral — avoid parking
    the whole main stack on a trivial tile,
- or deathtouch-era runner split: turn >= 800, enemy general known, and
  the stack can divide into two runners on parallel approach paths (two
  independent one-unit execution threats; see
  [`003-general-hunter-deathtouch-beeline.md`](../experiments/003-general-hunter-deathtouch-beeline.md)).

**Use `split=0` when any hold:**

- `A // 2 < need + SPLIT_MARGIN` — the capture needs the army (opponent
  stacks, castles, the enemy general: always commit everything),
- `A < SPLIT_MIN_ARMY` — splitting produces unusable crumbs,
- source is our general tile — never split off the general in v1 (losing
  the general's stack loses the game; castle_builder already treats the
  general as a special reserve),
- the move consolidates own territory — BFS frontier-march steps and
  own-cell merges keep `split=0` so army concentrates toward the frontier
  (the expand_plus fallback is unchanged).

## Choke interaction (pointer, not scope)

Splitter does not detect chokes. One overlap matters for the shared
scoring vocabulary: a split capture *into* a corridor mouth leaves a
garrison on the entry cell, which is the two-deep corridor defense that
[`choke_control`](choke_control.md) wants explicitly. If both bots are
built, the choke spec's split policy takes precedence inside corridor
cells; splitter keeps the general rule elsewhere.

## Pseudocode

```python
def act(obs):
    best_score, best_move = -1.0, None

    for each cell s owned by me with army A >= 2:
        for each direction d -> dest t (in board, passable):
            if not capturable(t):            # visible neutral or opponent
                continue
            D = army(t)

            if A - 1 > D:                    # split=0 candidate
                cand = (0, s.r, s.c, d, 0)
                score = base(A - 1, t)
                keep if score > best_score

            if can_split(s, t):              # split=1 candidate
                # can_split: A >= SPLIT_MIN_ARMY,
                #            A // 2 >= D + 1 + SPLIT_MARGIN,
                #            s is not our general
                cand = (0, s.r, s.c, d, 1)
                score = base(A // 2, t) * split_bonus(s, t)
                keep if score > best_score

    if best_move is not None:
        return best_move
    return frontier_march(obs)               # expand_plus fallback, split=0
    # then any-valid-move, then PASS — unchanged
```

`frontier_march`, `_any_valid_move`, `PASS` stay byte-identical to
`bots/expand_plus/agent.py`.

## Edge cases

- **A = 2, D = 0**: `split=1` moves 1, captures, leaves 1. Legal but
  blocked by `SPLIT_MIN_ARMY`; acceptable — value is negligible.
- **A = 3**: split moves 1, leaves 2. Same guard applies.
- **Failed capture is not invalid**: `A // 2 <= D` still *executes* and
  donates the moved army. The `SPLIT_MARGIN` guard plus the strict `>`
  check must make this unreachable in proposed candidates.
- **Recapture after split=0**: source holds exactly 1; any adjacent enemy
  stack of 2+ takes it back next turn. This is the exact case
  `W_GARRISON` prices.
- **Opponent stack visible on destination**: force `split=0` (real combat
  needs full army); the bonus table never outweighs `base(A-1)` vs
  `base(A//2)` at equal multiplier, so this falls out of the scoring
  naturally — verify in unit tests rather than trusting the arithmetic.
- **Move-order side effect**: halved moving army wins the "smaller army
  resolves first" tie more often. Harmless in v1; note in the experiment
  log if a match turns on it.
- **50-turn land tick**: garrison remainders still grow (+1 per owned
  cell per 50 turns), so split garrisons recover — supports the strategy.
- **Fog**: only visible neutral (`owner 0`, type not 0/5) and opponent
  cells are capturable; splitting into fog is never proposed because fog
  tiles are not capture candidates.
- **Time budget**: scoring stays \(O(HW)\) per turn plus one BFS — far
  under the 150 ms limit (RULES.md section 08).

## Experiment hypothesis (008)

File: `docs/research/experiments/008-splitter-half-army-fronts.md`
(write after implementation, per
[`experiment-protocol.md`](../experiment-protocol.md)).

- **Change**: split-flag decision layer on `expand_plus`. One change only.
- **Claim**: multi-front pressure and garrison remainders let splitter
  reach the enemy general more often than `expand_plus` on the same maps,
  converting some 1200-turn draws into wins, without increasing faults or
  losses.
- **Grid**: opponents `smoke`, `expand_plus`; seeds 0, 1, 2 (matches the
  001 grid for comparability); `--mode competition`; store all games under
  `data/games/` before any rating update.
- **Metrics**: winrate, draw rate, mean turns, faults. Known limitation
  (same as 001): the game-record schema stores winner/turns/terminated/
  truncated only, so land/army growth from efficient capture is not
  directly visible. If all games draw again, unit-test the split
  arbitration on synthetic boards (two-front fork, contested source,
  cheap-target overkill) and record "no regression, mechanism verified"
  exactly as 001 did. A follow-up schema extension for final land/army is
  a separate arena task — out of scope here.
- **Decision rule**: keep if no loss/fault regression; revert if losses
  or faults appear on any seed.

## Verification gate (when implemented)

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/splitter/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```

Match must finish (win, loss, or draw at 1200). Bot code is out of scope
for this spec change; only docs are written now.

## Parameter revision 1

### Measurement Context

Round 1 benchmark results ([`round1.md`](../measurements/round1.md),
58 games, grids `new_vs_smoke` / `new_vs_expand_plus` / `new_round_robin`):

- Winrate: 0.0% (0 wins, 3 losses, 7 draws out of 10 games).
- Mean turns: 982.5. Elo: 1455.5 (Rank 9 of 11).
- All three losses went to concentrated single-stack bots: `army_convey`
  in **341 turns** (seed 0, the fastest decisive game of round 1),
  `fog_scout` in 504, `late_rush` in 580. Halved split armies lost every
  direct fight.
- Both `smoke` games drew at 1200. Splitter never threatened the smoke
  general; the deathtouch runner-split gate (`turn >= 800` and enemy
  general known) never fired because splitter has no sighting.
- Round 1 open question "which new bots beat smoke on both seeds 0 and
  1": splitter beat smoke on **neither** seed.

### What to Keep Unchanged

- **Core identity**: split-aware multi-front pressure as one split-flag
  decision layer on the `expand_plus` baseline. No choke detection —
  corridor policy stays with [`choke_control`](choke_control.md).
- **Engine semantics**: `split=1` moves `A // 2`, leaves the ceil;
  capture needs moved army strictly greater than the defender.
- **Scoring shape**: `base(moved, dest)` times `split_bonus(s, dest)`
  with named weights; exact ties resolve to `split=0`.
- **The three split triggers**: `extra_fronts`, `contested`, `overkill`
  on neutral destinations.
- **`split=0` commitments**: opponent stacks, castles, the enemy general,
  any source that is our general, and all BFS consolidation steps.

### What to Tune

1. **Threat-gated split suppression** (new guard constant):
   - Never propose `split=1` when a visible opponent stack with army
     `>= A // 2` sits within `SPLIT_THREAT_RADIUS = 2` passable-BFS steps
     of the source or the destination.
   - Rationale: the 341-turn loss to `army_convey` shows splits fired
     near a concentrated enemy stack donate half armies piecemeal.
     `W_GARRISON` prices recapture by 2-army pokes; it does not price an
     adjacent 40-army stack.
2. **Survivable halves**:
   - `SPLIT_MIN_ARMY` 8 → 16 and `SPLIT_MARGIN` 2 → 4.
   - Rationale: under the v1 constants a legal half is 4 army; it cannot
     hold a captured cell against a one-step enemy poke, and all three
     losses show split fronts collapsing before turn 600. With 16/4 each
     half starts at 8 and captures with margin 4.
3. **Late probe-runner gate** (win-condition conversion):
   - Keep the existing runner split (`turn >= 800`, enemy general known).
     Add a second gate: at `turn >= PROBE_SPLIT_TURN = 1000` with the
     enemy general unknown, the largest frontier stack with
     `A >= SPLIT_MIN_ARMY` may split into two runners aimed at the two
     frontier cells farthest from the own general (passable BFS),
     preferring cells adjacent to fog; each runner keeps
     `A // 2 >= SPLIT_MIN_ARMY // 2`. One runner pair at a time.
   - Rationale: 0% winrate with 70% draws. The known-general gate
     requires a sighting splitter never gets (both smoke games drew at
     1200). Two independent runners preserve the multi-front identity and
     create the contact threats that `army_convey`, `late_rush`, and
     `fog_scout` used to convert wins. This is a conversion gate, not a
     scouting system.
