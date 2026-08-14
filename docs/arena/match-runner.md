# Match runner

`arena/matches/run_match.py` wraps `competition-module/competition/matchup.py` and writes one JSON record under `data/games/`.

## Rules

- Always uses `--mode competition`.
- Store the game **before** refitting ratings.
- The runner registers each bot's content hash in `data/bot_versions/` before
  the match, in this process. See [bot-version-registry.md](bot-version-registry.md).
- Prefer the project `.venv` (CPython 3.12). System Python 3.14 can fail on pygame / engine pins.

## Command

```bash
source .venv/bin/activate
python -m arena.matches.run_match \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0
```

Optional: `--round <name>` (default `adhoc`) sets both the record's `round` field
**and the directory it lands in** — `data/games/<round>/`. A record whose `round`
names a directory that does not exist is invisible to the rating layer, which is
how 102 loose files carrying 49 distinct round names accumulated under
`data/games/` and contributed to nothing.

Rating is a separate step, and there is no per-game option for it:

```bash
python -m arena.records.ratings
```

Ratings are fitted one round at a time, so a per-game refit would solve every
round in the store to publish one game — and a one-game round cannot clear the
30-game gate anyway. See [ratings.md](ratings.md#per-round-fits).

Thin smoke CLI:

```bash
python scripts/smoke_match.py --seed 0
```

## Recording a per-turn trajectory

```bash
python -m arena.matches.run_match bots/metro/run.sh bots/blitz/run.sh \
  --seed 13 --record
```

Off by default. With `--record` the match also writes
`data/trajectories/<round>/<game_id>.*`, and each seat whose bot carries a
`probe.py` is spawned through `arena.instrument.runner` instead of `run.sh` so
its internals are traced. The record then carries the reducer output for every
recorded series. Costs ~1.4% wall clock; the game itself is unchanged.
`--trajectories-dir` overrides where the files land.

See [trajectories.md](trajectories.md).

## Where the result comes from

- winner, turns, and `truncated` come from the loop's own outcome, not from
  parsed text;
- `castles_built_a` / `_b` are the engine's tally of castle births;
- `final_land_*` / `final_army_*` are the engine's terminal `GameInfo` — the
  bots are never asked, so a crashed seat is still scored;
- everything else in `metrics` needs `--record`.

See [game-record-schema.md](game-record-schema.md) for the schema v5 field list.

## Schema

See [game-record-schema.md](game-record-schema.md).

## Tournament

N seeds × bot pairs: [tournament.md](tournament.md).
