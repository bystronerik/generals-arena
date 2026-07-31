# Strategy spec — metro

Bot: `bots/metro/`. Migrated from the generals-bot repo as a grid-native
**redesign**: the source captured neutral cities, which do not exist in
the arena, so the production programme now **builds** castles and hunts
**enemy** castles. Grounded idea: **castle network + pressure — army spent
on land is spent once; army spent on production keeps paying.** Baselines
for comparison: `castle_builder`, `castle_rush`, `phase_switch`,
`expand_plus`.

Rule references: [`RULES.md`](../../../RULES.md) sections 03 (build castles),
04 (army growth). Source-tuned values transfer as **hypotheses**.

## 1. Priority ladder (every turn)

| Priority | Trigger | Behavior |
| --- | --- | --- |
| Killing blow | enemy general adjacent and affordable | Capture (deathtouch-aware; general allowed as source — the game ends). |
| Defence | enemy stack within `DEFEND_RADIUS = 8` of the general | Kill it where it stands, or gather home. Pessimistic: garrison vs biggest stack, no attrition credit. |
| Cash in | visible enemy castle affordable from an adjacent cell | Capture, cheapest first — a **double** production swing. |
| Claim / raid | opening (< 50) or losing the land race | Free adjacent neutrals; cheap enemy border cells (castles excluded). |
| Expansion beat | 1 turn in `EXPAND_EVERY = 3` | Keeps the land base from rotting while saving. |
| Castle programme | turn ≥ 60 and castles < `MAX_CASTLES = 3` | Choose a site, collection-walk the main stack onto it, build. |
| Pressure | from turn `PRESSURE_FROM = 140` (or army ratio) | Always keep a wave in the field. |
| Fallbacks | nothing else | Expand; consolidate onto the poorest castle. |

## 2. The castle programme (the redesign)

- **Site scoring** (lower is better): `build_cost + FORWARD_WEIGHT × dist
  from the enemy anchor`, over owned plain cells, never within
  `FRONT_SAFETY = 6` of a visible enemy. The crowding surcharge in
  `build_cost` makes ≥7 spacing emerge naturally; paying surcharge for a
  genuinely forward cell is allowed — forward castles feed the waves.
- **Funding**: the main stack (≥ `WALK_MIN = 6`) collection-walks onto the
  site, absorbing surplus en route (dragging every frontier cell to a
  rally strips the frontier to 1 army — the source's live traces showed
  exactly that failure). The build fires when the site holds
  `cost + BUILD_KEEP = 3`; a project that has not funded itself in
  `GIVE_UP_TICKS = 80` is abandoned.
- **What was deleted**: chipping (`chip_min/chip_wait/chip_fraction`) —
  neutral-city damage was permanent; enemy castle armies regrow, so
  partial attacks are donations. City-flip tracking went with it.

## 3. Garrisons and war chests (ported unchanged)

An owned castle within `CASTLE_SAFE_DIST = 6` of a visible enemy is a
front-line bastion: locked as a source (a move sends all-but-one, so
draining it flips it). Castles holding more than `CASTLE_GARRISON = 8` are
war chests, reserved from expansion. The general locks below
`GENERAL_FREE_DIST = 8` enemy distance and only releases with
`MIN_BANK = 8` aboard.

## 4. Pressure (ported nearly unchanged)

From turn 140 there is always a wave: the spearhead launches at
`max(min_push, min(biggest enemy stack + 5, 0.55 × our army))`, keeps
pushing on the lower `MIN_PUSH = 18` bar, and is only taken over by a
stack ≥ `WAVE_TAKEOVER = 1.6×` bigger (a regenerating home castle must not
hijack an advance). Target: the believed general, else the enemy anchor.
This wave is what keeps metro out of the draw-prone home-builder cluster.

## 5. Arena-rule adaptations (vs the generals-bot source)

| Rule | Adaptation |
| --- | --- |
| No neutral cities | City purchases → castle builds; chipping deleted; scout-bias toward fogged obstacles deleted (type-5 cells are almost always mountains before enemies build). |
| Enemy castles exist | New top-priority cash-in channel. |
| Deathtouch (800) | Killing blow accepts any 2-army neighbour; `chase_defence` from 780. |
| Draw at 1200 | Pressure-from-140 is the anti-draw mechanism. |

## 6. Diversity check (vs the castle-economy cluster)

Per [`diversity-constraints.md`](diversity-constraints.md) the failure
case is a fourth rested-general home builder. Metro differs on every axis:
**funding** (gather-walked field army, never a rested general — the
general is a locked bank, not a builder), **placement** (forward-scored
sites, not cheapest-near-home), **defence of investment** (garrisoned
bastions), **aggression channel** (enemy-castle capture priority — no
existing bot has it), and **mandatory pressure waves from turn 140**
(the cluster's draws come from never pushing).

## 7. Experiment hypothesis

**Hypothesis:** three forward castles + one general out-produce the pool
by mid-game, and the pressure wave converts that into captures rather than
draws. Verification games: beat `smoke` (378), `expand_plus` (382), and —
the cluster test — `castle_builder` by **capture** at turn 382 (no draw),
building 3 castles in each game.

Seed grid: opponents `smoke`, `expand_plus`, `castle_builder`,
`castle_rush`; seeds 0–2; `--mode competition`. Metrics: W-L-D, castles
built (target 3), draws vs the castle cluster (primary diversity signal —
must not regress to all-draws), faults.

See [`014-metro-castle-network.md`](../experiments/014-metro-castle-network.md).
