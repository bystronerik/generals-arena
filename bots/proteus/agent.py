"""Proteus — the adaptive composite (spec: docs/research/strategies/proteus.md).

Classifies the opponent every turn from one shared OpponentModel, switches
between the four migrated strategy cores (with hysteresis, see
switcher.py), and delegates the move to the active core. Inactive cores'
memories are kept warm each turn so a mid-game switch starts informed.

Sibling-core imports (``from blitz.agent import BlitzCore`` etc.) resolve
because ``bots/`` is on ``sys.path`` in both stdio mode (main.py) and
in-process mode (arena.bot_api.load_strategy_class).
"""
from __future__ import annotations

from _common.oppmodel import OpponentModel
from aegis.agent import AegisCore
from blitz.agent import BlitzCore
from boom.agent import BoomCore
from metro.agent import MetroCore
from proteus.classifier import classify
from proteus.switcher import Switcher

DEFAULT_STRATEGY = "blitz"
"""The spine (source calibration): played from turn 0; mid-game switches
into blitz cold-start its opening and measurably lose games."""


class Agent:
    """Strategy-switching bot built from the migrated strategy cores."""

    def __init__(self, player_id: int, H: int, W: int,
                 default: str = DEFAULT_STRATEGY) -> None:
        # One shared model: Agent.act updates it once per turn; each core's
        # observe() is idempotent per turn, so warming never double-counts.
        self.model = OpponentModel()
        self.switcher = Switcher(default=default)
        self.cores = {
            "blitz": BlitzCore(player_id, H, W, model=self.model),
            "boom": BoomCore(player_id, H, W, model=self.model),
            "metro": MetroCore(player_id, H, W, model=self.model),
            "aegis": AegisCore(player_id, H, W, model=self.model),
        }

    def act(self, obs):
        self.model.update(obs)
        cls = classify(self.model, obs.turn)
        active = self.switcher.update(cls, obs.turn)

        # Keep inactive cores' beliefs/latches current for a warm handover.
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
