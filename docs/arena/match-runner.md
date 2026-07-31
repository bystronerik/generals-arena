# Match runner

`arena/run_match.py` wraps `competition-module/competition/matchup.py` and writes one JSON record under `data/games/`.

## Rules

- Always uses `--mode competition`.
- Store the game **before** updating ratings.
- Prefer the project `.venv` (CPython 3.12). System Python 3.14 can fail on pygame / engine pins.

## Command

```bash
source .venv/bin/activate
python -m arena.run_match \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0
```

Optional: `--update-ratings` applies Elo after the game file exists.

Thin smoke CLI:

```bash
python scripts/smoke_match.py --seed 0
```

## Result parsing

The runner reads matchup output lines:

- `player N captured the enemy general` → winner `a` (N=0) or `b` (N=1), `terminated=true`
- `truncated at ... turns (draw)` → `winner=draw`, `truncated=true`
- `[matchup] castles built: N (...) vs M (...)` → optional `castles_built_a` / `_b`
- `[telemetry] player=P ...` (last line per player on stderr) → optional final land/army and sighting metrics (schema v2)

See [game-record-schema.md](game-record-schema.md) for the full v2 field list.

## Schema

See [game-record-schema.md](game-record-schema.md).

## Tournament

N seeds × bot pairs: [tournament.md](tournament.md).
