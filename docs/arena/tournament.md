# Tournament

`arena/tournament.py` runs games-per-pair × bot pairs under competition mode
with a ProcessPool of in-process match workers. Each match is stored under
`data/games/<round>/`. Elo rebuilds once after the pool finishes (unless
`--no-ratings`).

Full competition games can run to **1200** turns (draw/truncation). Plan long
timeouts when you need hang detection.

## Command

```bash
source .venv/bin/activate
python -m arena.tournament \
  bots/smoke/run.sh \
  bots/expand_plus/run.sh \
  --round parallel-smoke \
  --games-per-pair 50 \
  --round-seed 1

# thin CLI
python scripts/tournament.py \
  bots/smoke/run.sh \
  bots/expand_plus/run.sh \
  --round my-round \
  --games-per-pair 2 \
  --round-seed 0 \
  --no-ratings
```

## Flags

| Flag | Meaning |
| --- | --- |
| `--round` | **required**; games store under `data/games/<round>/` |
| `--games-per-pair` | random map seeds per unordered pair (default: **50**) |
| `--round-seed` | RNG seed for map-seed generation (default: 0) |
| `--seeds` | optional fixed seed list/ranges; overrides random generation |
| `--jobs` | parallel workers (default: physical CPU cores; capped to that) |
| `--no-ratings` | store only; skip Elo rebuild |
| `--swap-sides` | also play B vs A (off by default; Rule C uses random seeds instead) |
| `--timeout` | per-match wall-clock seconds limit |
| `--include-self` | every ordered pair including self-play |
| `--games-dir` | override output directory |

## Parallelism

- Workers run [`arena/competition_match.py`](../../arena/competition_match.py)
  (in-process JAX + stdio bots), not a fresh `matchup.py` per match.
- Each worker pins BLAS/OpenMP/XLA to one thread so packing can reach one
  match per physical core.
- Default `--jobs` equals physical core count and is hard-capped there.
- Do not raise `--jobs` past physical cores: a CPU-starved bot can miss the
  150 ms move budget and look faulty.

## Round layout

```
data/games/<round>/
  manifest.json          # round_seed, assignments, jobs, bots
  <game_id>.json         # one file per match
```

`arena/records/ratings.py` rebuild scans `data/games/` recursively (skips
`manifest.json`). Legacy flat JSON at `data/games/*.json` still loads.

## Verification

Every match uses competition mode via the in-process runner. After the grid,
check `data/games/<round>/` and `data/ratings/leaderboard.md`.
