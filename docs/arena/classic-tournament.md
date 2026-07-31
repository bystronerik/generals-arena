# Classic tournament

`arena/classic_tournament.py` runs N seeds × bot pairs under the
classic-approximate harness (`arena/classic_match.py`). Each match is stored
under `data/classic_games/` — **not** `data/games/` and **not** Elo.

For a full grid plus Markdown/JSON report, use [`scripts/measure_classic.py`](../../scripts/measure_classic.py).

## Command

```bash
source .venv/bin/activate
python -m arena.classic_tournament \
  bots/classic_duel/run.sh \
  bots/smoke/run.sh \
  --seeds 0-2
```

Measurement CLI (stores games + writes `docs/research/measurements/<round>.md`):

```bash
python scripts/measure_classic.py \
  classic_duel smoke expand_plus \
  --seeds 0-4 --swap-sides --round classic-duel-stress
```

## Flags

| Flag | Meaning |
| --- | --- |
| `--seeds` | `0,1,2` or `0-3` (or mixed) |
| `--swap-sides` | also play B vs A |
| `--include-self` | every ordered pair including self-play |
| `--grid-size` | square board side (default 24) |
| `--truncation` | max turns (default 5000) |
| `--jobs` | parallel match workers (default 1) |
| `--games-dir` | override JSON store (default `data/classic_games/`) |

## Record schema

Each JSON file mirrors the competition record shape with `mode: "classic"` and
an extra `winner_player_id` field (0, 1, or -1 from the harness). See
[`game-record-schema.md`](game-record-schema.md) for the competition fields;
classic games omit castle telemetry.

## Verification

Matches use `run_classic_match` (neutral cities, no build, no deathtouch).
Confirm JSON under `data/classic_games/` and that nothing new appears under
`data/games/` or `data/ratings/`.

## Related

- [`../engine/classic-matchup.md`](../engine/classic-matchup.md) — single-match harness
- [`../research/strategies/human-95-plan.md`](../research/strategies/human-95-plan.md) — remote human-block goal
