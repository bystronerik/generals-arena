# Replay analysis

`arena/instrument/replay/`, driven by `scripts/replay.py`. Turns one scraped
[leaderboard replay](leaderboard-replays.md) into something an agent can read:
where the economies diverged, what happened and when, whether a side ever
learned where the enemy general was, and whether it ever gathered an army and
went at it.

Strictly read-only over `competition-replays/`. Nothing here writes anywhere,
and nothing here may feed `data/games/`, `data/ratings/` or
`data/remote_games/` — these are observational games played under rules we did
not run.

---

## CLI

```bash
python scripts/replay.py full erik.bystron 15932
```

| command | sections | example |
| --- | --- | --- |
| `summarize` | timeline, verdict | `python scripts/replay.py summarize erik.bystron 15932 --every 32` |
| `events` | phases, event log | `python scripts/replay.py events erik.bystron 15932` |
| `flow` | information vs action, verdict | `python scripts/replay.py flow erik.bystron 15932` |
| `full` | all four | `python scripts/replay.py full erik.bystron 15932` |
| `batch` | one line per replay, then flaw aggregates | `python scripts/replay.py batch erik.bystron --outcome lose` |

Every command takes `--json`; `summarize` and `full` take `--every N` for the
timeline sample rate (default 16). Exit codes: `2` no such replay (or no played
replay in the folder), `3` the game is a forfeit.

Cost is roughly 0.1–0.2 s per replay. `batch` loads one file at a time and
drops the second grid pass, so a whole folder never sits in memory.

## Sections

| section | what it says |
| --- | --- |
| header | ids, board, seats, which seat is the queried player, the player's own `outcome`, the `folder` it was filed under, winner |
| timeline | every `N`th tick per seat: army, tiles, largest stack and where it stood, `dgen` (Manhattan distance from that stack to the enemy general), `near` (closest owned tile to it) |
| phases | `expansion` until first contact, `contest` after it, `collapse` once the loser stops recovering to 75% of its own peak land |
| events | the chronological log, each tagged with its phase |
| information vs action | per seat: when the enemy general became knowable, what the largest stack did about it, every gather wave and where it ended |
| verdict | three to six sentences, each one a restatement of the numbers above |

`dgen` and `near` are measured against the enemy general's **true** cell, which
a side may never have seen. That is deliberate: they say how close a side got,
whether or not it knew. Vision is the separate question the flow section
answers.

## Events

Replays store state, not actions, so every event is a reading of what changed
between two frames.

| kind | inferred from |
| --- | --- |
| `first_contact` | the first tick two opposing tiles are orthogonally adjacent |
| `first_capture` | the first owner change from one player to the other |
| `big_capture` | a captured cell that held >= 10 army |
| `castle_built` | an owned cell that gained exactly +1 off a growth tick, no neighbour losing army, confirmed >= 2 times |
| `castle_captured` | that cell changing hands afterwards |
| `first_general_sight` | the enemy general's cell entering a seat's vision |
| `gather_wave` | the largest stack moving *and* growing for >= 5 consecutive ticks |
| `tile_loss_streak` | >= 20 ticks without a single tile gain, >= 5 tiles lost |
| `expansion_stall` | > 50 ticks never rising above the starting tile count while the opponent gained >= 10 |
| `general_captured`, `game_end` | the general cell changing hands; the last tick |

## Definitions

- **Vision** — a seat sees every cell within Chebyshev distance 1 of a cell it
  owns (`generals/core/game.py: get_visibility`, a 3x3 stamp).
- **Knowability** — a general cannot move, so from the tick its cell first
  enters vision it is treated as known for the rest of the game. Everything
  downstream is measured against that window, not against frame-by-frame
  sight, so a bot is never blamed for looking away.
- **Gather wave** — the signal that a bot is assembling an attack rather than
  trading land: the largest stack grows on every tick of a run in which it also
  moves. Their absence is reported explicitly.
- **Expansion stall** — a long stretch capped at the tile count it started
  with, while the opponent grew. It fires on a collapse too: the wording is
  `<= N tiles`, not "stuck at exactly N".
- **Toward / away fraction** — of the largest stack's *moves* (holds and jumps
  are excluded), the share that closed on the current target. The target is the
  enemy general once it has ever been seen, otherwise the nearest enemy tile
  currently in vision, otherwise nothing — and a move with no target counts in
  neither fraction (`blind_moves`). Both cells are measured against the *same*
  target, so a target that changes cannot manufacture progress. A one-step move
  always changes Manhattan distance by exactly 1, so there is no third bucket:
  a directed move either closed or it did not.
- **Castle detection lag** — the build itself is invisible in the grids, and
  the detector needs two productions to confirm, so `castle_built` is stamped
  at the *first observed production*, a couple of ticks after the action. A
  tick on which more than eight cells each gain +1 is read as growth, not as
  eight castles, and contributes no confirmations at all.
  Competition maps start with no castles at all
  ([`../competition/build-castles.md`](../competition/build-castles.md)), and
  the raw `castles` field is empty, so this inference is the only source.

## Caveats

- **Forfeits are skipped.** `total_ticks <= 1` is a scoring artefact, not a
  played game: the single-replay commands exit `3`, and `batch` counts them
  separately instead of averaging them in.
- **The final frame of a decided game is useless as a score.** Capturing a
  general hands over every tile the loser owned, so the last timeline row reads
  everything-to-nothing. `batch` reports the last frame where both seats still
  held land; read `full`'s timeline the same way.
- **"Toward %" is not progress toward the general.** Each target is labelled by
  kind — `enemy_general`, `visible_enemy_tile`, `none` — and a seat that never
  sighted the general can still show 80% toward, meaning it chased whatever
  enemy tile was nearest. Read the fraction next to
  `gather_waves_at_general` and `ever_moved_at_known_general`.
- **The outcome folder is not the queried player's result — so nothing here
  reads it.** The scraper files a game under the *list row's*
  `win`/`lose`/`draw`, which is side A's result, and side A is not always the
  player whose folder it is: 846 of the 1680 replays on disk are side B, and
  835 of those sit in a folder stating the opposite of what happened to them
  (30 of 324 for `erik.bystron`: 22 wins filed under `lose/`, 8 losses under
  `win/`). `Replay.outcome`, the header's `outcome`, every `GameLine`, the
  `--outcome` filter and `by_outcome` are all derived from the replay's
  `winner` and the name-resolved seat. The directory survives as
  `folder` (with `folder_disagrees`), provenance only; `full` prints
  `[filed under lose/, side A's result]` when the two differ. Because the
  filter is derived, `batch --outcome lose` still globs all three directories.
- **Self-matches.** Both seats would be the same account, so a name cannot
  resolve the seat and the metadata's `a_side` decides. The current pull
  contains none — the "30 self-matches" once reported for `erik.bystron` were
  its 30 side-B games, misread.

## Related

- [`leaderboard-replays.md`](leaderboard-replays.md) — where the replays come
  from, the raw format, and why they are not arena matches
- [`../arena/trajectories.md`](../arena/trajectories.md) — the same kind of
  question asked of *our* matches, from per-turn recordings
