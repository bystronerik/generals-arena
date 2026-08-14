# Leaderboard replays (generals.bot)

Finished **competition-rules** games played by real leaderboard entrants,
downloaded from the `generals.bot` API by the `competition-scraper` submodule.

They are observational data for analysis and evaluation targets. They are not
arena matches: they never enter `data/games/`, `data/ratings/`, or
`data/remote_games/`, and no rating is fit from them (see
[`../arena/ratings.md`](../arena/ratings.md)).

---

## Fetching

```bash
python scripts/scrape_replays.py                    # default player: erik.bystron
python scripts/scrape_replays.py erik.bystron prady --concurrency 8
```

The wrapper fixes `--out` to `competition-replays/` and reports on-disk
inventory after the run. It needs `httpx`, and the submodule:
`git submodule update --init competition-scraper`. Raw form, if you want the
submodule CLI directly:

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

`win` / `lose` / `draw` are the **queried player's** result. An early scraper
filed by the list row's `winner` field, which is *side A's* result, and the
queried player is side A only about half the time; that history has since been
repaired in place. Treat the folder as provenance anyway and derive the real
result from the replay's `winner` and the player's seat — see the caveat below.

### Repairing the sort

```bash
python scripts/scrape_replays.py --refile-only
```

Offline, idempotent, and downloads nothing: it re-derives each saved replay's
outcome from its `.meta.json` sidecar and moves mislabeled pairs. This is the
only way to heal history, because the list endpoint serves a recent window and
deleted players return nothing at all. With no player named it repairs every
directory on disk, which is the useful default — a misfiled folder is the one
you have not thought to name.

## Match metadata (`<id>.meta.json`)

| field | meaning |
| --- | --- |
| `id` | match id; also the replay filename |
| `a_name` / `b_name` | the two accounts; **either one** can be the queried player |
| `a_side` | side A's seat index — `a_name == players[a_side]` held in every replay sampled (3,000 of 13,088 at 2026-08-05) |
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
- **Derive the outcome; do not read it off the folder.** The folders are
  currently accurate — all 13,088 pairs across 14 players were verified filed
  correctly on 2026-08-05, twice over: once from each replay's own `winner`
  against the name-resolved seat, once from the `.meta.json` sidecars via
  `--refile-only` (13,088 checked, 0 refiled). That makes them accurate, not
  authoritative. An older scraper filed by side A's result and was wrong for
  roughly half the sample, so keep deriving: the player's seat from `players`
  in the replay (or `a_side` when the names cannot decide), then compare with
  the replay's `winner`. `arena/instrument/replay/` does this and exposes the
  directory separately as `folder`, so a future regression surfaces as a
  `folder_disagrees` count instead of quietly mislabeled training data.
- **No self-matches in the current pull.** An earlier version of this page
  reported "30 of 306 `erik.bystron` matches had the same account on both
  sides". That was the side-B count above, misread: no replay on disk has
  `players[0] == players[1]`. Should one appear, its name cannot resolve a
  seat and `a_side` has to.
- **Forfeits look like 1-tick wins.** 132 of the 13,088 games on disk at
  2026-08-05 ended at `total_ticks == 1`. Filter on length before treating a
  game as played.
- **`castles` was empty in all 306 replays** of that pull, so the field is not a
  reliable source of castle positions — read castle state from the tick grids.
- Size is roughly 0.6 MB per replay — 12 GB for the 13,088 replays across 14
  players held at 2026-08-05. Never commit them.

## Related

- [`sprint-replays.md`](sprint-replays.md) — whole generals.bot tournaments,
  same replay format but a different source: a results asset plus blob URLs, so
  a different scraper
- [`replay-analysis.md`](replay-analysis.md) — reading one of these games:
  timeline, events, fog vs action, and the whole-folder flaw aggregate
- [`local-matchup.md`](local-matchup.md) — our own competition matches, which
  *do* feed `data/games/`
- [`remote-eval-heuristics.md`](remote-eval-heuristics.md) — live classic play,
  a different ruleset again
