# Classic-approximate local matchup

Practice classic generals.io rules locally without competition modifiers.
Results from this harness **do not** enter `data/games/` or Elo.

## When to use

- Tune `classic_duel` and other remote bots before live blocks.
- Verify neutral city behavior (`build_castles=False` keeps pre-placed cities).
- Compare bots under no build, no deathtouch, and no 1200-turn draw cap.

For competition verification, use [`local-matchup.md`](local-matchup.md) with
`--mode competition` instead.

## Command

From the repo root:

```bash
python -m arena.classic_match \
  bots/classic_duel/run.sh \
  bots/expand_plus/run.sh \
  --seed 0
```

Against smoke:

```bash
python -m arena.classic_match \
  bots/classic_duel/run.sh \
  bots/smoke/run.sh \
  --seed 0
```

## Environment settings

`arena/classic_match.py` builds `GeneralsEnv` with:

| Setting | Value | Effect |
| --- | --- | --- |
| `build_castles` | `False` | no build action; neutral cities stay on the map |
| `deathtouch_turn` | `None` | general capture needs strictly more army |
| `truncation` | 5000 (default) | no competition 1200-turn draw |
| `num_castles_range` | (3, 6) | pre-placed neutral cities |
| `castle_val_range` | (20, 40) | city garrison range (remeasure from live) |
| `grid_dims` | (24, 24) default | wider than competition 18–21 |

Override board size and truncation:

```bash
python -m arena.classic_match bots/classic_duel/run.sh bots/smoke/run.sh \
  --seed 1 --grid-size 20 --truncation 3000
```

## Batch grid

For N seeds × bot pairs with JSON storage under `data/classic_games/` (never
Elo), use the classic tournament runner:

```bash
python -m arena.classic_tournament \
  bots/classic_duel/run.sh \
  bots/smoke/run.sh \
  --seeds 0-4 --swap-sides
```

Or the measurement CLI, which also writes a report under
`docs/research/measurements/`:

```bash
python scripts/measure_classic.py \
  classic_duel smoke expand_plus \
  --seeds 0-4 --swap-sides --round classic-duel-stress
```

See [`../arena/classic-tournament.md`](../arena/classic-tournament.md).

## Relation to matchup.py

`competition-module/competition/matchup.py` only accepts `--mode competition`
from the CLI for graded matches. The classic harness reuses the same stdio loop
and `make_board` / `make_transition` helpers from `matchup.py`, but constructs
its own `GeneralsEnv` kwargs. This is a repo-side wrapper, not a submodule edit.

## Remote verification

After local classic matches, dry-run the remote adapter:

```bash
python scripts/remote_play.py --mode dry-run --bot classic_duel
```

See [`remote-play-setup.md`](remote-play-setup.md) for live credentials.

## Related

- [`../research/strategies/classic_duel.md`](../research/strategies/classic_duel.md) — champion spec
- [`../research/strategies/human-95-plan.md`](../research/strategies/human-95-plan.md) — Phase 3 plan
- [`../competition/vs-classic.md`](../competition/vs-classic.md) — rule deltas
