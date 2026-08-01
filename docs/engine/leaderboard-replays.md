# Leaderboard replays (generals.bot)

Finished **competition-rules** games played by real leaderboard entrants,
downloaded from the `generals.bot` API by the `competition-scraper` submodule.

They are observational data for analysis and evaluation targets. They are not
arena matches: they never enter `data/games/`, and no rating is fit from them
(see [`../arena/ratings.md`](../arena/ratings.md)).

---

## Fetching

```bash
python scripts/scrape_replays.py                    # default player: erik.bystron
python scripts/scrape_replays.py erik.bystron prady --concurrency 8
```

The wrapper fixes `--out` to `competition-replays/` and reports on-disk
inventory after the run. It needs `httpx`. Raw form, if you want the submodule
CLI directly:

```bash
python competition-scraper/scrape.py erik.bystron --out competition-replays
```

Runs are incremental: a match whose `<id>.json` already exists is skipped, and
replays are written through a `.tmp` rename, so an interrupted run leaves no
truncated file.

## Layout

```
competition-replays/            # gitignored except .gitkeep
└── erik.bystron/
    ├── win/
    │   ├── 10858.json          # replay, verbatim from the API
    │   └── 10858.meta.json     # match row from the list endpoint
    ├── lose/
    └── draw/
```

`win` / `lose` / `draw` come from the list row's `winner` field, which is
**side A's** result. The queried player is *not* always side A, so the folder
is not their outcome — see the caveat below. Treat it as provenance and derive
the real result from the replay's `winner` and the player's seat.

## Match metadata (`<id>.meta.json`)

| field | meaning |
| --- | --- |
| `id` | match id; also the replay filename |
| `a_name` / `b_name` | the two accounts; **either one** can be the queried player |
| `a_side` | side A's seat index — agreed with the name-resolved seat in 1680/1680 replays |
| `created_at` | ISO timestamp |
| `seed` | map seed |
| `turns` | game length in ticks |
| `winner` | `A` / `B` / `D` |

## Replay (`<id>.json`)

| field | shape | meaning |
| --- | --- | --- |
| `version` | int | replay format version (`1` as of 2026-08-01) |
| `dims` | `{rows, cols}` | board size |
| `players` | `[name, name]` | index = owner id used in `ticks` |
| `seed` | int | map seed, matches the metadata row |
| `mountains` | `[[row, col], …]` | impassable cells |
| `castles` | `[[row, col], …]` | neutral castles at spawn |
| `generals` | `[[row, col], …]` | one per player, indexed like `players` |
| `ticks` | `[{armies, owners}, …]` | one frame per tick, each a `rows × cols` grid |
| `winner` | int | player index, or `-1` on a draw |
| `total_ticks` | int | `len(ticks) - 1`; tick `0` is the initial state |

`owners[r][c]` is `-1` for neutral, otherwise an index into `players`.
Coordinates are `[row, col]` throughout.

## Caveats when using these as data

- **The window is recent, not complete.** The list endpoint returns a fixed
  window (~300 matches for one player) with no pagination. History only
  accumulates by re-running the scrape periodically.
- **The folder is side A's outcome, and half the games are side B.** Across the
  1680 replay/meta pairs on disk at 2026-08-01, 846 have the queried player as
  side `B`, and 835 of those sit in a folder that states the opposite of what
  happened to them (`erik.bystron`: 30 side-B games — 22 real wins filed under
  `lose/`, 8 real losses under `win/`). Counting outcomes by directory is
  therefore wrong by roughly half the sample. Derive instead: the player's seat
  from `players` in the replay (or `a_side` when the names cannot decide), then
  compare with the replay's `winner`. `arena/instrument/replay/` does this and
  exposes the directory separately as `folder`.
- **No self-matches in the current pull.** An earlier version of this page
  reported "30 of 306 `erik.bystron` matches had the same account on both
  sides". That was the side-B count above, misread: no replay on disk has
  `players[0] == players[1]`. Should one appear, its name cannot resolve a
  seat and `a_side` has to.
- **Forfeits look like 1-tick wins.** 6 of 306 games ended at `total_ticks == 1`.
  Filter on length before treating a game as played.
- **`castles` was empty in all 306 replays** of that pull, so the field is not a
  reliable source of castle positions — read castle state from the tick grids.
- Size is roughly 0.6 MB per replay, ~330 MB for one player's window.

## Related

- [`replay-analysis.md`](replay-analysis.md) — reading one of these games:
  timeline, events, fog vs action, and the whole-folder flaw aggregate
- [`local-matchup.md`](local-matchup.md) — our own competition matches, which
  *do* feed `data/games/`
- [`remote-eval-heuristics.md`](remote-eval-heuristics.md) — live classic play,
  a different ruleset again
