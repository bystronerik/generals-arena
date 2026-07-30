# Generals Competition — Game rules

Processed from [https://www.generals.bot/rules](https://www.generals.bot/rules) (fetched 2026-07-31 with a browser User-Agent; plain fetch returned 403).

Cross-checked against `GeneralsEnv(mode="competition")` in `competition-module/generals/core/env.py`, plus modifiers `build_castles.py` and `deathtouch.py`. See **Code cross-check** at the end.

---

## 01 — The board

The game uses a rectangular grid with four tile types:

- **Plains** — empty passable tiles. Moving onto a neutral plain makes it yours.
- **Mountains** — impassable and permanent. Moves into a mountain are invalid.
- **Castles** — player-built structures. A standing castle produces army for its owner like a general and can be captured in combat. Maps start with **no castles**. (Type `3` in the bot observation.)
- **Generals** — each player's home base. Lose yours and the game is over.

### Map generation

| Parameter | Value |
| --- | --- |
| Board size | rectangular — each side drawn independently in 18–21 per game |
| Mountains | about 19–23% of the board (65–105 tiles) on the official page |
| Castles | none at the start — players build them |
| Generals | at least **17 BFS steps** apart over plain tiles |
| Connectivity | a passable path between the generals is guaranteed |

Maps are random per game. Official note: seeds are not reused on the competition infra, so bots cannot tune to known layouts. Local `matchup.py` still accepts `--seed` for reproducible research.

---

## 02 — Turns and moves

- Each turn, each player issues one action: **move**, **build a castle**, or **pass**.
- A move pushes army from one of your cells into an adjacent cell (up, down, left, or right). You send either **half** the cell's army or **all but one**. You always leave at least one unit behind.
- Moving between your own cells merges armies — no combat.
- Invalid moves are a silent pass (cell you do not own, mountain, off board, or nothing to send).
- Each turn, both players' moves and fights resolve first; army growth applies afterward.

### Move order (same turn)

Both moves happen on the same turn, but one resolves fully before the other. Priority, highest first — a tie falls through to the next:

1. **Chasing** — your move targets the cell the opponent's move starts from.
2. **Reinforcing** — your move lands on a cell you own.
3. **Smaller army** — otherwise, the move from the smaller cell resolves first.

Both generals captured on the same turn → the match is a **draw**.

Note: a recent generals.io update resolves simultaneous general captures by swapping land and armies. This competition keeps the older rule — it is a draw.

---

## 03 — Building castles

There are no neutral castles to capture — you **build** your own. A build is a full action (it replaces your move that turn) and permanently turns one of your plain cells into a castle.

- **Where:** any plain cell you own — not a mountain, not a general, not an existing castle.
- **Cost:** paid from the army on that cell. Base price **35**, plus a crowding surcharge for each structure you already own (your general and every castle of yours) near the target: **max(0, 14 − 2 × distance)**, Manhattan. Structures 7+ tiles away add nothing.
- **Remainder stays:** build on 50 army at price 47 and the new castle keeps 3. Build with the exact price and the castle sits at **0 army**.
- **Production starts immediately:** a built castle generates one army every other turn, like your general.
- **Invalid builds are a silent pass.**

| Distance to your nearest structure | Surcharge | Price |
| --- | --- | --- |
| adjacent | +12 | 47 |
| 2 | +10 | 45 |
| 3 | +8 | 43 |
| 4 | +6 | 41 |
| 5 | +4 | 39 |
| 6 | +2 | 37 |
| 7 or more | +0 | 35 |

Surcharges stack. Example: general and another castle both at distance 2 → 35 + 10 + 10 = 55. Prices are live — every castle you gain (built or captured) raises your own prices near it. Enemy structures never affect what you pay.

Corner cases:

- Builds resolve **before** either player's move each turn.
- Players build only on their own cells; prices depend only on own structures — builds never conflict between players.
- A castle captured from the enemy is yours: it produces for you and counts toward your build prices.
- Castles are permanent — they never revert to plains; they only change owners.

### Extended action space

One line of five integers: `<pass> <row> <col> <dir> <split>`

- `0 r c d s` — **move** from (r, c) in direction d
- `1 0 0 0 0` — **pass**
- `2 r c 0 0` — **build** a castle at (r, c); last two fields ignored

---

## 04 — Army growth

- **Every other turn**, your general and every castle you own (built or captured) each generate one army.
- **Every 50 turns**, every cell you own gains one army.

---

## 05 — Combat and capture

Moving onto an opponent-owned cell starts combat: armies subtract; the attacker takes the cell only with **strictly more** army and keeps the difference. On an exact tie the defender keeps the cell.

Castles defend like any owned cell — there are no neutral garrisons. Capture an enemy castle and it produces for you.

**Capturing the enemy general wins instantly.**

---

## 06 — Visibility

The competition uses **fog of war**, like classic generals.io. Each bot sees only cells next to tiles it owns. Scouted tiles fade back into fog when you move away. You cannot see an enemy's army or moves outside your vision.

---

## 07 — How a game ends

- **Win** — capture the opponent's general.
- **Deathtouch, from turn 800** — any move that *executes* onto the enemy general's tile wins instantly, no matter how large the defending army is. One unit is lethal. Defense is a *chase*: capture the attack's source cell from a third tile that same turn and the touch never executes. Counter-attacking from the general itself is a head-on clash — the attacker wins it.
- **Draw** — neither general falls within **1200 turns**. The cap is a hard draw regardless of territory or army counts.

---

## 08 — Match constraints (competition infrastructure)

- **Time per move:** 150 ms. First move gets a 10 s grace period.
- **Fault budget:** late, missing, or malformed reply → pass + one fault. **50 faults** in one game forfeits. Crash or exit forfeits immediately. Invalid but correctly formatted game actions are silent passes and do not add a runner fault.
- **Hardware:** one dedicated CPU core and a hard 2 GB memory cap per bot. Engine on its own core. No GPU at match time.
- **Network:** disabled during matches.
- **Build step:** `build.sh` runs once before games, with no network. Artifacts must live in the zip or the competition environment.

Bot interface and submit docs: [https://www.generals.bot/docs](https://www.generals.bot/docs).

---

## Code cross-check

Pinned by `GeneralsEnv(mode="competition")`:

| Rule | Official page | Engine preset / modifiers |
| --- | --- | --- |
| Board size | sides 18–21 | `min_grid_size=18`, `max_grid_size=21`, `pad_to=21` |
| Fog | on | `perfect_info=False` |
| Castles | none pre-placed; players build | `build_castles=True`; `strip_neutral_castles` |
| Build cost | 35 + max(0, 14−2d) per own structure | `BASE_COST=35`, `PROXIMITY_PENALTY=14`, `PROXIMITY_DECAY=2` |
| Deathtouch | from turn 800 | `deathtouch_turn=800` |
| Cap | 1200 turns | `truncation=1200` |
| Spawn distance | ≥ 17 BFS steps | `min_generals_distance=17` |
| Mountains | page says ~19–23% (65–105 tiles) | preset `mountain_density_range=(0.24, 0.26)` |

**Discrepancy:** official mountain density text (~19–23%) does not match the competition preset (24–26%). Prefer the env preset for local `--mode competition` matches until the site and code agree.

`pad_to=21` is an engine detail (mountain-pad smaller boards for fixed observation size). It is not on the rules page.
