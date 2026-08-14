# Sprint replays (generals.bot tournaments)

Every game of a published `generals.bot` sprint — a bot-vs-bot tournament run
by the organizers over archived checkpoints, under **competition rules**.

Like [leaderboard replays](leaderboard-replays.md), these are observational
data. They are not arena matches: they never enter `data/games/`,
`data/ratings/`, or `data/remote_games/`, and no rating is fit from them.

---

## What a sprint asset is

`https://www.generals.bot/assets/<tourney-id>.json` is a tournament *report*,
not a leaderboard query. One file carries the run configuration, standings,
cross tables, a fault matrix, and one row per game. The replays are not served
by the leaderboard API at all — each row points at a gzipped blob on public
storage.

That is why the leaderboard scraper cannot fetch these: it knows one endpoint
(`/api/leaderboard`) and files by a queried player's `win`/`lose`/`draw`, and a
sprint row names two checkpoints and a winning *seat*, with no player to query.
Do not extend the `competition-scraper` submodule for sprints —
`scripts/scrape_sprint.py` owns this source.

The `sprint-2026-08-08` run, for reference: 15 archived checkpoints, a
qualifier of every pair over 5 maps (1050 games), then the top 6 replayed at 20
maps (600 games). Both stages appear in one `matches` list.

## Fetching

```bash
python scripts/scrape_sprint.py
```

```bash
python scripts/scrape_sprint.py https://www.generals.bot/assets/sprint-2026-08-08.json
```

Needs `httpx`, and the `competition-scraper` submodule for its pacing
primitives (`git submodule update --init competition-scraper`). Runs are
incremental and writes are atomic, so an interrupted or throttled run resumes by
re-running; `--dry-run` prints the plan offline from a local copy of the asset,
and `--limit N` caps a smoke test. Pacing reuses the scraper submodule's rate
limiter and shared throttle gate.

## Layout

```
competition-replays/_sprints/           # gitignored with the rest of competition-replays/
└── sprint-2026-08-08/
    ├── sprint.json                     # the asset, verbatim
    ├── bots.json                       # checkpoint key -> name, user_id, rank
    └── 160ac2f2cf4fc267_2dcb5f39ae3aca35/
        ├── 1816681097-0.json           # replay, decompressed
        └── 1816681097-0.meta.json      # the sprint row(s) that named this blob
```

One directory per **pair**, named by the two checkpoint keys in the row's `pair`
order; files are `<seed>-<side>`. Keys, not display names: keys are stable and
unique, names are neither. `bots.json` reads a key back as a bot.

The `_sprints/` prefix keeps these out of the player namespace —
`competition-replays/<player>/` — since nothing here is a player's history.
Nothing in `arena/instrument/replay/` walks the replay root, so the sibling
directory is inert to the leaderboard tooling.

## Match row (`<seed>-<side>.meta.json`)

The sidecar is `{source, replay_gz, stages, matches: [row, …]}`, with each row
verbatim from the asset:

| field | meaning |
| --- | --- |
| `pair` | `<p0_key>|<p1_key>` — the matchup, in a fixed order |
| `p0_key` / `p1_key` | checkpoint keys; **these swap between the two sides' rows** |
| `p0_name` / `p1_name` | display names for those keys |
| `seed` | map seed |
| `a_side` | which of `p0`/`p1` played side A |
| `winner` | `p0` / `p1` / `draw` |
| `turns` | game length in ticks |
| `faults` | per-player fault count (protocol violations, timeouts) |
| `suspect` / `forfeit` | organizer adjudication flags for that game |
| `stage` | `qualifier` or `final` |
| `replay_gz` | the blob URL this file came from |

The rows are nested under `matches` rather than spliced in at the top level on
purpose. A leaderboard sidecar's `winner` is `A`/`B`/`D` against
`a_name`/`b_name`; a sprint row's is `p0`/`p1`/`draw` against checkpoint keys.
Flattening would hand `arena.instrument.replay.Meta` a field it would read in
the wrong vocabulary, so it reads none of them and the outcome stays derived
from the replay's own `winner` — the one source that cannot be filed wrong.

## Replay (`<seed>-<side>.json`)

Byte-identical in schema to a leaderboard replay — `version`, `dims`,
`players`, `seed`, `mountains`, `castles`, `generals`, `ticks[{armies,
owners}]`, `winner`, `total_ticks`. Field meanings are in
[leaderboard-replays.md](leaderboard-replays.md#replay-idjson), and
`arena/instrument/replay/` loads these unchanged:

```python
from pathlib import Path
from arena.instrument.replay.loader import load_replay

replay = load_replay(path, queried_player="Kubic", folder="win")
replay.outcome  # derived from the replay's winner and the seat, not the folder
```

`folder` is a required argument there and means nothing for a sprint game; pass
the outcome you are filtering for, or ignore `folder_disagrees`.

## Caveats when using these as data

- **Rows outnumber replays.** In `sprint-2026-08-08`, 1650 rows resolve to 1500
  distinct blobs: the final round's first seeds repeat the qualifier's, so the
  same game is listed once per stage under one URL. The scraper groups by URL
  and keeps both rows in `stages`. Counting rows double-counts 150 games.
- **The result is the organizers', on their engine.** `run.engine_sha` pins
  their engine, and their own note says the pin post-dates the 2026-07-27
  move-order tiebreak flip and is *not comparable to pre-Jul-27 games*. Read
  `run.ruleset` before comparing anything: `sprint-2026-08-08` ran
  `move_timeout 0.15`, `first_move_timeout 10`, `truncation 1200`,
  `max_turns 2000`, `max_faults 50`, with 1 CPU per bot and 2 GB.
- **A win here is a win under their resource limits.** A bot that faults or
  times out on 1 core may not on ours, and the reverse. Check `faults`,
  `suspect`, and `forfeit` before reading a result as play quality; the asset's
  `suspect_games`, `flipped_games`, and `adjudication` blocks say which games
  the organizers themselves re-ran.
- **The bots are checkpoints, not accounts.** Two rows with the same display
  name can be different checkpoints; the key is the identity.
- **Never into `data/games/`.** Same rule as leaderboard replays, for the same
  reason: not produced by our runner, no bot version, no rating fit.
- Roughly 0.9 MB per decompressed replay — about 1.4 GB for the 1500 of
  `sprint-2026-08-08`. Gitignored; never commit them.

## Related

- [`leaderboard-replays.md`](leaderboard-replays.md) — the per-player scrape,
  the same replay format, and the sort-order history behind `win/lose/draw`
- [`replay-analysis.md`](replay-analysis.md) — reading one of these games
- [`local-matchup.md`](local-matchup.md) — our own competition matches, which
  *do* feed `data/games/`
