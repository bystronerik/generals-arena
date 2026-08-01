"""CONTRACT STUB — the real search is written by the search owner.

This file defines only the boundary `agent.py` depends on. Replacing the body
must not change the signatures below.

    TacticalSearch(params).improve(obs, core_move, core, deadline) -> Outcome

- `obs` is the bot's own fogged `Observation` (`bots/_common/wire.py`): `H`,
  `W`, `turn`, `my_land`, `my_army`, `opp_land`, `opp_army`, and the three
  `H x W` int grids `type_grid` / `owner_grid` / `army_grid`. Cell types are
  0 fog, 1 plain, 2 mountain, 3 castle, 4 general, 5 structure-in-fog; owners
  are 0 neutral, 1 us, 2 them.
- `core_move` is the vendored core's own five-int action for this turn. It is
  already legal. Returning it unchanged is always a valid outcome.
- `core` is the live `BlitzCore`; `core.memory` carries the cross-turn belief
  (`memory.belief.enemy_general`, `memory.stack`, `memory.target`,
  `memory.phase`) and `core.memory.opp` the opponent model.
- `deadline` is an absolute `time.monotonic()` timestamp. **It must be checked
  inside the rollout loop, not only between iterations** — one rollout that
  runs long is enough to miss a turn, and §08 forfeits at 50 of those.
- The return is `Outcome(move, searched, iters)`. `move` may be `None` or
  `core_move` to decline. `searched` says whether the search actually ran (as
  opposed to being skipped by scoping), `iters` how many iterations completed.

The stub declines every turn, so a bot built on it plays exactly the vendored
core — which is also what `MACARIA_TUNE='{"mcts_enabled": false}'` gives, and
what the control arm of the measurement is.
"""
from __future__ import annotations

from typing import NamedTuple


class Outcome(NamedTuple):
    move: tuple[int, int, int, int, int] | None
    searched: bool
    iters: int


class TacticalSearch:
    def __init__(self, params) -> None:
        self.params = params

    def improve(self, obs, core_move, core, deadline: float) -> Outcome:
        return Outcome(move=None, searched=False, iters=0)
