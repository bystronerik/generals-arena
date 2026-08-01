"""macaria — the vendored blitz core, with a tactical search allowed to
override it inside a self-enforced wall-clock budget.

Structure, and why it is this way:

- `blitz_core` is a **copy** of `bots/blitz/agent.py`, not an import. blitz is
  the A arm of every contrast macaria is measured by; a copy makes the baseline
  unreachable from this bot by construction (`blitz_core`'s docstring has the
  hash it was taken at).
- `params` holds every behavioural constant of both halves.
- `search` owns the move search. It is handed a deadline and a move that is
  already legal, and its contract is to return *something playable* no matter
  what — see the budget discussion below.

**The turn budget.** RULES.md §08 gives 150 ms per move and forfeits a bot
that accumulates 50 late or malformed replies; a crash forfeits immediately.
macaria spends at most `mcts_latency_cap_ms` (100) on the whole move and keeps
the remaining 50 ms as reserve for interpreter jitter and the stdio round
trip. The heuristic core runs first and is *not* interruptible, so the search's
deadline is what is left of the cap after the core has been paid, further
capped by the search's own budget. Two consequences are deliberate:

- a turn where the core alone overruns the cap does no search at all rather
  than a truncated one;
- the search's deadline is an absolute `time.monotonic()` timestamp rather
  than a duration, so it cannot drift as it is passed down.

`act` never raises. A search bug that would crash the process forfeits the
whole game under §08, whereas the core's move is always legal and already
computed — so the fallback costs one turn of search and saves the match. That
is the one broad `except` in this bot, and it is load-bearing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_BOTS = Path(__file__).resolve().parent.parent
if str(_BOTS) not in sys.path:  # `arena.bot_api` imports agent.py directly
    sys.path.insert(0, str(_BOTS))

from blitz_core import BlitzCore
from params import blitz_config, load_params
from search import TacticalSearch

_MS = 1000.0


class Agent:
    """Arena entrypoint (contract: agent_class(player_id=..., H=..., W=...))."""

    def __init__(self, player_id: int, H: int, W: int) -> None:
        self.params = load_params()
        self._core = BlitzCore(
            player_id, H, W, config=blitz_config(self.params)
        )
        self._search = (
            TacticalSearch(self.params) if self.params.mcts_enabled else None
        )
        # Last-turn telemetry. Read by `probe.py` (arena-owned, never imported
        # from here) and by nothing else — no branch depends on these.
        self.searched = False
        self.overrode = False
        self.search_iters = 0
        self.move_ms = 0

    @property
    def phase(self) -> str:
        return self._core.phase

    def act(self, obs):
        started = time.monotonic()
        self.searched = False
        self.overrode = False
        self.search_iters = 0

        move = self._core.decide(obs)

        if self._search is not None:
            cap_at = started + self.params.mcts_latency_cap_ms / _MS
            now = time.monotonic()
            if now < cap_at:
                deadline = min(cap_at, now + self.params.mcts_budget_ms / _MS)
                try:
                    outcome = self._search.improve(
                        obs, move, self._core, deadline
                    )
                except Exception:  # noqa: BLE001 — see the module docstring
                    outcome = None
                if outcome is not None:
                    self.searched = bool(outcome.searched)
                    self.search_iters = int(outcome.iters)
                    if outcome.move is not None and outcome.move != move:
                        self.overrode = True
                        move = outcome.move

        self.move_ms = int(round((time.monotonic() - started) * _MS))
        return move
