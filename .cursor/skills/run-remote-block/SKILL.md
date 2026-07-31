---
name: run-remote-block
description: >-
  Runs or resumes a 100-game human block on live generals.io through
  scripts/remote_play.py, counts eligible games with arena/remote_block.py, and
  publishes docs/research/measurements/remote-block<N>.md. Use for Phase 3
  human evaluation and the 95/100 gate ladder.
---

# Run remote block

## Model split

- Think model: stop/go at gates B–D; reads losing replays; chooses the next block hypothesis
- Composer: runs sessions, counts logs, drafts the block report

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

## Preconditions

1. Competition gate passed for the bot commit (`run-competition-match`).
2. Classic harness proxy acceptable (`run-classic-grid`).
3. `GENERALS_USER_ID` set (see [`docs/engine/remote-play-setup.md`](../../../docs/engine/remote-play-setup.md)).
4. Offline verify: `python scripts/remote_play.py --bot <name> --mode dry-run`.

## Commands

Dry-run (no network):

```bash
python scripts/remote_play.py --bot classic_duel --mode dry-run \
  --username "$GENERALS_USERNAME" --lobby-id "$GENERALS_LOBBY_ID"
```

Lobby gate A (controlled human):

```bash
python scripts/remote_play.py --bot classic_duel --mode lobby --max-games 10
```

Public 1v1 queue:

```bash
python scripts/remote_play.py --bot classic_duel --mode 1v1 --max-games 1
```

Count human-eligible games in the current log dir:

```python
from pathlib import Path
from arena.remote_block import count_human_block_games
count_human_block_games(Path("data/remote_games"))
```

Only rows with `opponent_is_bot is False` and `counts_toward_block` count.

## Outputs

| Path | Contents |
| --- | --- |
| `data/remote_games/*.json` | Per-game logs (gitignored) |
| `docs/research/measurements/remote-block<N>.md` | Block report (commit this) |

Remote games never enter `data/games/` or `data/ratings/`.

## Gate ladder

Follow [`docs/research/strategies/human-95-plan.md`](../../../docs/research/strategies/human-95-plan.md) §3.4.
Stop at the sixth loss in a block.

## Changelog

- 2026-07-31 — Initial skill (repo velocity review S1)
