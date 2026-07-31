# Ratings (elote)

`arena/ratings.py` rates bots with elote `EloCompetitor` (`beat` / `tied` / `lost_to`).

## Flow

1. Store every match under `data/games/<game_id>.json`.
2. Apply the stored result to the rating book.
3. Persist competitor state and leaderboard under `data/ratings/` (local only;
   JSON and Markdown snapshots are gitignored — rebuild from `data/games/`).

Do not update Elo from live stdout alone.

## Files

| Path | Contents |
| --- | --- |
| `data/ratings/competitors.json` | Elo state + rated game ids + W/L/D |
| `data/ratings/leaderboard.json` | Ranked snapshot |
| `data/ratings/leaderboard.md` | Same snapshot as Markdown |

Initial Elo for new bots: **1500**.

## Rebuild

```bash
source .venv/bin/activate
python -m arena.ratings --print
# or
python scripts/leaderboard.py
```

Rebuild walks all `data/games/*.json` in `finished_at` order.

## Incremental update

`arena/tournament.py` and `arena/run_match.py --update-ratings` call `rate_stored_game` after each store. Already-rated `game_id` values are skipped on incremental apply; rebuild resets from the game store.
