"""Yankee — a tuned, search-augmented proteus (spec: docs/research/strategies/yankee.md).

Forked from `bots/proteus` at content hash `7ae237e99d34` and kept structurally
identical at seq 1 so that the fork itself is measurable: same classifier, same
switcher, same two cores, only the module names changed. Every later increment
is a separate registered step with its own contrast against proteus, which
stays byte-identical as the A arm.

Classifies the opponent every turn from one shared OpponentModel plus its own
home-pressure latch, switches between strategy cores with hysteresis (see
switcher.py), and delegates the move to the active core. The inactive core's
memory is kept warm each turn so a mid-game switch starts informed.

Everything yankee changes about the cores lives in yankee-local config objects
(`BlitzConfig` / `BoomConfig` instances constructed here) or in yankee-owned
modules. `bots/blitz`, `bots/boom` and `bots/_common` are never edited: they sit
in proteus's source closure, so a change there would move the baseline under the
comparison (docs/arena/decision-rule.md, "connectivity").

Sibling-core imports (``from blitz.agent import BlitzCore`` etc.) resolve
because ``bots/`` is on ``sys.path`` in both stdio mode (main.py) and
in-process mode (arena.bot_api.load_strategy_class).
"""
from __future__ import annotations

from _common.oppmodel import OpponentModel
from blitz.agent import BlitzCore
from boom.agent import BoomCore
from yankee.classifier import classify
from yankee.signals import HomePressure
from yankee.switcher import Switcher

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
