"""Proteus — the adaptive composite (spec: docs/research/strategies/proteus.md).

Classifies the opponent every turn from one shared OpponentModel plus its own
home-pressure latch, switches between strategy cores with hysteresis (see
switcher.py), and delegates the move to the active core. The inactive core's
memory is kept warm each turn so a mid-game switch starts informed.

Two cores, not four. metro and aegis were constructed and warmed by every
previous version but `COUNTER` could never select either, and a 32-game
per-cell grid of all four cores against the whole roster says that was the
right answer for the wrong reason: both are dominated (0.596 and 0.603 pooled,
against blitz 0.788 and boom 0.816) and uniquely best against nothing. Warming
them cost two `observe()` calls a turn to reach a branch that did not exist.

Sibling-core imports (``from blitz.agent import BlitzCore`` etc.) resolve
because ``bots/`` is on ``sys.path`` in both stdio mode (main.py) and
in-process mode (arena.bot_api.load_strategy_class).
"""
from __future__ import annotations

from _common.oppmodel import OpponentModel
from blitz.agent import BlitzCore
from boom.agent import BoomCore
from proteus.classifier import classify
from proteus.signals import HomePressure
from proteus.switcher import Switcher

DEFAULT_STRATEGY = "blitz"
"""The spine (source calibration, re-measured): played from turn 0, because
mid-game switches into blitz cannot replay its opening and a blitz that never
acted holds no chain and no strike stack. Always-boom beats it on the mean
(0.815 to 0.787) and loses to it on the floor (0.36 to 0.50)."""


class Agent:
    """Strategy-switching bot built from the migrated strategy cores."""

    def __init__(self, player_id: int, H: int, W: int,
                 default: str = DEFAULT_STRATEGY) -> None:
        # One shared model: Agent.act updates it once per turn; each core's
        # observe() is idempotent per turn, so warming never double-counts.
        self.model = OpponentModel()
        # Proteus-owned, because it joins two facts the shared model keeps
        # apart — stack size and distance from home, at the same instant.
        self.pressure = HomePressure()
        self.switcher = Switcher(default=default)
        self.cores = {
            "blitz": BlitzCore(player_id, H, W, model=self.model),
            "boom": BoomCore(player_id, H, W, model=self.model),
        }

    def act(self, obs):
        self.model.update(obs)
        self.pressure.update(obs)
        cls = classify(self.model, obs.turn, self.pressure)
        active = self.switcher.update(cls, obs.turn)

        # Keep the inactive core's beliefs/latches current for a warm handover.
        for name, core in self.cores.items():
            if name != active:
                core.observe(obs)

        return self.cores[active].decide(obs)

    @property
    def label(self) -> str:
        """Current opponent classification label (for traces/tests)."""
        return self.switcher.label

    @property
    def active_strategy(self) -> str:
        return self.switcher.current
