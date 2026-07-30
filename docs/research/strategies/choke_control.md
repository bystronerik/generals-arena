# choke_control — corridor control strategy spec

Status: **spec only, not implemented**. Target bot path:
`bots/choke_control/` (fork `bots/expand_plus/`; keep `main.py` /
`run.sh` protocol intact). Baseline: `expand_plus`. One change only:
choke-aware capture scoring plus a hold/deny rule for owned choke cells.

## Goal

Claim and hold narrow corridors so the opponent's expansion is funneled
or stopped:

1. **Claim** — prefer capturing corridor cells over open-plain cells of
   equal army cost, because corridors are the cells the opponent *must*
   pass through to reach us.
2. **Hold** — once a choke cell is owned, do not drain its army on
   sideways captures; keep a stack that the opponent cannot cheaply push
   through.
3. **Deny** — reinforce a held choke instead of attacking through it,
   unless the attack is overwhelming. Channeled attacks through a 1-wide
   corridor die piecemeal against a stacked defender.

Maps make this relevant: competition mountain density is 24–26%
([`map-generation.md`](../../competition/map-generation.md)), so random
boards are rich in accidental corridors and pockets.

## Choke detection heuristic (core)

Three layers, computed per observation. v1 implements layers 1 and 2;
layer 3 is the frontier relevance weight.

### Layer 1 — narrowness (static geometry)

From `type_grid`: impassable = mountain (2), structure-in-fog (5), or
off-board. For every passable cell `x`:

```
narrow(x) = count of impassable orthogonal neighbors   # 0..4
```

- `narrow >= 2` marks corridor candidates.
- `narrow == 3` marks pocket/dead-end interiors; the *mouth* of the
  pocket (the passable neighbor chain leading out) is the hold point.
- Refinement for `narrow == 2`: if the two passable neighbors are
  opposite (up+down, left+right) the cell is a straight corridor segment;
  if adjacent (e.g. up+left) it is a bend — weight bends at half.

### Layer 2 — gate test (local cut)

A corridor cell only matters if traffic actually funnels through it.
Cheap articulation proxy: for candidate `x`, take its passable orthogonal
neighbors; check whether they are mutually connected *without* stepping
on `x` (two neighbors are connected if orthogonally adjacent to each
other, transitively, within the neighbor set only).

```
gate(x) = passable_neighbors(x) split into >= 2 components when x is removed
```

Example: passable neighbors up and down, not adjacent to each other →
removing `x` disconnects them → `x` is a gate. Open-plain cells (4
mutually adjacent neighbors) fail the test. This is \(O(1)\) per cell.

### Layer 3 — frontier relevance

A gate behind our lines is worthless. Weight by position relative to the
expansion frontier:

```
choke_score(x) = (narrow_weight(x) if gate(x) else 0)
               * (1 + W_OPPRESS if opponent cells visible within K BFS steps beyond x else 1... )
```

Simpler v1 form: use the expand_plus multi-source BFS field (seeded at
capturable tiles). A gate cell `x` is **frontier-relevant** if at least
one neighbor across the gate has `frontier_dist` smaller than the owned
side — i.e. the gate separates owned space from capturable space.
`K = 6` for the opponent-proximity boost `W_OPPRESS`.

### Fog handling

Structure-in-fog (5) reads as impassable, so unexplored map looks like
walls and fog-edge cells overestimate `narrow`. Mitigation: compute
`gate(x)` and `narrow(x)` only over **visible** neighbors, and halve
`choke_score(x)` if any of the 8 surrounding cells is fog (type 0). As
fog lifts, scores correct themselves.

## Features (per candidate move / per owned cell)

| Feature | Meaning |
| --- | --- |
| `narrow(x)`, `gate(x)` | layers 1–2 above |
| `choke_score(x)` | fog-decayed, frontier-weighted choke value |
| `held_choke(x)` | owned gate cell with capturable or opponent space beyond it |
| `threat(x)` | largest visible opponent army within `K` BFS steps beyond choke `x` |
| `frontier_dist` | expand_plus BFS field, reused for layer 3 and the march fallback |
| `forward(s, t)` | move `s -> t` crosses the gate line (increases `frontier_dist` on our side) |

## Scoring

Capture scoring extends expand_plus; hold/deny modifies source
eligibility.

```
# Capture destinations (visible neutral or opponent, A - 1 > D):
score = base(A - 1, dest) * (1 + W_CHOKE * choke_score(dest))
base  = (A - 1) * 10 * (2 if dest is opponent-owned else 1)

# Own-merge destinations (reinforce):
score = REINFORCE_BONUS * choke_score(dest)   # only held_choke dests
```

**Hold rule (source restriction):** a move *from* a `held_choke` cell `s`
is allowed only if

- the destination is also a choke/gate cell on the same corridor line
  (shifting the hold forward), or
- the attack is overwhelming: `A - 1 >= OVERWHELM_MULT * threat(s)`, or
- `A < HOLD_MIN_ARMY` (stack too small to matter; let it fight rather
  than idle).

**Deny rule (destination preference):** when `held_choke(d)` and
`threat(d) > 0`, reinforcing `d` from the largest adjacent/interior stack
outranks non-overwhelming attacks through `d` — implement by giving the
reinforce candidate a score above any non-overwhelming forward capture
from `d`'s side of the corridor.

Initial constants (tune by experiment):

| Constant | Value | Guard |
| --- | --- | --- |
| `W_CHOKE` | 0.5 | per unit `choke_score` on capture destinations |
| `W_OPPRESS` | 1.0 | opponent-presence boost in layer 3 |
| `K` | 6 | BFS radius for threat/opponent proximity |
| `HOLD_MIN_ARMY` | 10 | minimum stack that counts as a hold |
| `OVERWHELM_MULT` | 3 | attack-through-choke force ratio |
| `REINFORCE_BONUS` | 15.0 | flat weight for garrison merges into held chokes |
| `FOG_DECAY` | 0.5 | choke_score multiplier near fog |

## When to use `split=1` vs `split=0`

Default is `split=0` everywhere (inherited). One exception:

- **Two-deep hold**: a large stack moves *into* a corridor mouth with
  `split=1` when `A // 2 > D` and the source cell is itself on the
  corridor line — the entry cell keeps half, the mouth gets half, and the
  opponent must chew through two stacked cells in a 1-wide corridor.
- Never split when attacking *through* a choke (combat needs full army),
  and never split below `SPLIT_MIN_ARMY = 8` (shared vocabulary with
  [`splitter`](splitter.md)).

## Pseudocode

```python
def act(obs):
    narrow, gate = choke_layers(obs)          # layers 1-2, O(H*W)
    dist = frontier_bfs(obs)                  # expand_plus field
    chokes = {x: choke_score(x) for x in cells if gate(x)}
    held  = {x for x in chokes if owned(x) and frontier_beyond(x, dist)}

    best_score, best_move = -1.0, None
    for each owned cell s with army A >= 2:
        for each direction d -> t (in board, passable):
            if held.contains(s) and not allowed_from_hold(s, t, obs):
                continue                       # hold rule
            if capturable(t) and A - 1 > army(t):
                score = base(A - 1, t) * (1 + W_CHOKE * chokes.get(t, 0))
                split = 1 if two_deep_hold(s, t) else 0
                keep (0, s.r, s.c, d, split) if score > best_score
            elif owned(t) and t in held:
                score = REINFORCE_BONUS * chokes[t]   # deny rule
                keep (0, s.r, s.c, d, 0) if score > best_score

    if best_move is not None:
        return best_move
    return frontier_march(obs)                # unchanged expand_plus
    # fallback: prefer march steps that end adjacent to a held choke
    # when threat > 0; then any-valid-move, then PASS — unchanged
```

## Edge cases

- **Fog misreads**: unknown map looks like walls → phantom chokes at the
  fog edge. `FOG_DECAY` plus the visible-neighbors-only rule caps the
  damage; verify on synthetic fogged boards in unit tests.
- **Chokes everywhere**: at 24–26% mountains, raw `narrow >= 2` fires
  often. The `gate` test plus frontier relevance must do the real
  filtering; if capture scoring saturates toward corridors only, lower
  `W_CHOKE`.
- **Useless holds**: a gate the opponent already passed, or one far from
  any opponent, is not worth army. `held_choke` requires capturable or
  opponent space beyond; re-check every turn — a hold can become useless
  as the frontier moves.
- **Series / parallel corridors**: holding one of two parallel corridors
  does not deny anything. Layer 3 only weights a gate if capturable space
  lies beyond *it*; v1 accepts the approximation, v2 could test regional
  connectivity with the cell removed (windowed articulation).
- **Starvation while holding**: a garrison that never moves still grows
  (+1 per 50 turns land tick), and the general/castles keep producing —
  the hold is not an economic dead end, but do not let `held` cells
  exempt so much army that expansion stalls. `HOLD_MIN_ARMY` caps the
  exemption to cells that matter.
- **Deathtouch era (turn >= 800)**: holding chokes does not end games.
  If the enemy general is sighted, a corridor hold within its pocket
  becomes the strongest execute-block; pushing through a defended choke
  stays gated by `OVERWHELM_MULT` even late. Composition with a
  general-hunter-style beeline is out of v1 scope.
- **Castle synergy (note only)**: a castle built on a held choke produces
  army directly on the hold point — strongest defense in the ruleset.
  Funding a build needs castle_builder's reserve logic; out of v1 scope,
  record as a composition candidate.
- **Own general in a pocket**: if our spawn sits in a dead-end, the mouth
  cell is the single most important cell on the board — `held_choke`
  covers it automatically once owned; check spawn-pocket maps in tests.
- **Time budget**: layers 1–2 are \(O(HW)\) with tiny constants; one BFS
  per turn. Far under the 150 ms limit (RULES.md section 08).

## Experiment hypothesis (009)

File: `docs/research/experiments/009-choke-control-corridor-hold.md`
(write after implementation, per
[`experiment-protocol.md`](../experiment-protocol.md)).

- **Change**: choke-aware scoring + hold/deny rules on `expand_plus`.
  One change only.
- **Claim**: on corridor-rich maps, choke_control loses less land to
  opponent expansion and converts opponent attacks into failed pushes,
  showing as no-worse-than-baseline results on the standard grid with
  equal or fewer losses; on open maps it must behave identically to
  `expand_plus` (choke scores ~0).
- **Grid**: opponents `smoke`, `expand_plus` (add `splitter` if it exists
  by run time); seeds 0, 1, 2; `--mode competition`; store all games
  under `data/games/` before any rating update.
- **Metrics**: winrate, draw rate, mean turns, faults. Expectation:
  holding behavior likely *raises* draw rate (matches the all-draws
  pattern of experiments 001–003); decisive value needs final land/army
  telemetry, which the current game-record schema does not store — same
  known gap as 001/008, flagged as a separate arena follow-up.
- **Mechanism tests** (primary evidence if games all draw): unit-test the
  detector on synthetic boards — straight corridor, bend, open plain,
  pocket with mouth, fogged edge — and unit-test the hold rule (a held
  choke never emits a sideways capture below `OVERWHELM_MULT`).
- **Decision rule**: keep if no loss/fault regression and detector tests
  pass; revert or retune `W_CHOKE` if losses appear or if the bot visibly
  stalls expansion (mean turns at cap with near-zero land growth —
  requires the telemetry follow-up to confirm).

## Verification gate (when implemented)

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/choke_control/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```

Match must finish (win, loss, or draw at 1200). Bot code is out of scope
for this spec change; only docs are written now.
