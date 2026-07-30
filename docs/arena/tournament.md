# Tournament

`arena/tournament.py` runs N seeds × bot pairs under competition mode. Each match is stored, then rated.

Full competition games can run to **1200** turns (draw/truncation). Plan long timeouts for smoke vs expander grids.

## Command

```bash
source .venv/bin/activate
python arena/tournament.py \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --seeds 0-2

# thin CLI
python scripts/tournament.py \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --seeds 0
```

## Flags

| Flag | Meaning |
| --- | --- |
| `--seeds` | `0,1,2` or `0-3` (or mixed) |
| `--no-ratings` | store only |
| `--swap-sides` | also play B vs A |
| `--timeout` | per-match seconds limit |
| `--include-self` | every ordered pair including self-play |

## Verification

Every match uses `--mode competition` via `arena/run_match.py`. After the grid, check `data/games/` and `data/ratings/leaderboard.md`.
