# Tournament

`arena/tournaments/competition.py` runs games-per-pair × bot pairs under competition mode
with a ProcessPool of in-process match workers. Each match is stored under
`data/games/<round>/`. Elo rebuilds once after the pool finishes (unless
`--no-ratings`).

Full competition games can run to **1200** turns (draw/truncation). Plan long
timeouts when you need hang detection.

## Command

```bash
source .venv/bin/activate
python -m arena.tournaments.competition \
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
| `--no-ratings` | store only; skip the refit |
| `--seat-policy` | `random` (default) or `alternate` — see below |
| `--strict-versions` | refuse to start if any bot's closure differs from HEAD |
| `--timeout` | per-match wall-clock seconds limit |
| `--include-self` | also play each bot against itself |
| `--games-dir` | override output directory |
| `--record` | also write per-turn trajectories under `data/trajectories/<round>/` |

## Seat policy

Seat is drawn per game from the round's seeded stream, not implied by roster
position. Under the old scheme "sits in seat A" and "appears earlier in the
roster" were the same variable, so the fitted seat advantage was aliased with
the strength parameters.

| Policy | What it does | Use it for |
| --- | --- | --- |
| `random` | one game per map seed, orientation drawn from the pair's stream | large exploratory and regeneration rounds — **zero extra games** |
| `alternate` | every map seed played in **both** orientations | decision arms — exactly 50/50, and map difficulty cancels within each matched pair |

`alternate` draws half as many distinct seeds for the same game count, and
rounds an odd `--games-per-pair` up to keep the split exact. Self-play pairs
play one game per seed under either policy.

`--swap-sides` is gone. It keyed the pair RNG on the *oriented* pair, so the
mirrored copy drew an entirely different seed list: 2× the games for an
unmatched sample. `--seat-policy alternate` is what it should have been.

## Parallelism

- Workers run [`arena/matches/competition.py`](../../arena/matches/competition.py)
  (in-process JAX + stdio bots), not a fresh `matchup.py` per match.
- Each worker pins BLAS/OpenMP/XLA to one thread so packing can reach one
  match per physical core.
- Default `--jobs` equals physical core count and is hard-capped there.
- Do not raise `--jobs` past physical cores: a CPU-starved bot can miss the
  150 ms move budget and look faulty.

## Round layout

```
data/games/<round>/
  manifest.json          # round_seed, seat_policy, assignments, jobs, bots, hashes
  <game_id>.json         # one file per match
```

With `--record`, alongside it:

```
data/trajectories/<round>/
  <game_id>.traj.jsonl.gz      # engine truth, per turn
  <game_id>.trace.{a,b}.jsonl.gz  # probe trace, per probed seat
```

The parent creates the round's trajectory directory once; each worker then
writes only its own game's files into it, the same pattern that makes
`save_game` pool-safe — no shared writer, no lock. Recording costs ~1.4% wall
clock and never changes a game's outcome. See [trajectories.md](trajectories.md).

The rating refit walks each `data/games/<round>/` directory and fits it
**independently** — there is no pooled fit. Eligibility still comes from each
record's own `mode`, `round`, `engine_version` and content hashes, never from the
directory it sits in; the directory decides which *fit* a record belongs to. Loose
games directly under `data/games/` are in no round and enter nothing. See
[ratings.md](ratings.md#per-round-fits).

## Verification

Every match uses competition mode via the in-process runner. After the grid,
check `data/games/<round>/` and this round's section of
`data/ratings/leaderboard.md`. Ratings in another round's section are on another
scale and are not comparable to it.
