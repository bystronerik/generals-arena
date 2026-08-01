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

The cores are **vendored, not imported**: `blitz_core.py` and `boom_core.py`
are yankee's own copies of `bots/blitz/agent.py` and `bots/boom/agent.py`.
Those two files sit in proteus's source closure, so editing them would move
the baseline under every comparison here (docs/arena/decision-rule.md,
"connectivity") — and the measured cm_hunter failure is in core *logic* that
no `BlitzConfig` value reaches. After the fork yankee's closure holds nothing
another bot depends on except `_common/`, which yankee also never edits, so
nothing yankee does can move another entity's hash.

Values live in `params.py`; behaviour lives in the vendored cores.
"""
from __future__ import annotations

from _common.oppmodel import OpponentModel
from yankee.blitz_core import BlitzCore
from yankee.boom_core import BoomCore
from yankee.classifier import classify
from yankee.params import YankeeParams, blitz_config, boom_params, load_params
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
                 default: str = DEFAULT_STRATEGY,
                 params: YankeeParams | None = None) -> None:
        self.params = load_params(params)
        # One shared model: Agent.act updates it once per turn; each core's
        # observe() is idempotent per turn, so warming never double-counts.
        self.model = OpponentModel()
        # Yankee-owned, because it joins two facts the shared model keeps
        # apart — stack size and distance from home, at the same instant.
        self.pressure = HomePressure(radius=self.params.pressure_radius,
                                     duel_army=self.params.duel_army)
        self.switcher = Switcher.from_params(self.params, default)
        self.cores = {
            "blitz": BlitzCore(player_id, H, W, model=self.model,
                               config=blitz_config(self.params)),
            "boom": BoomCore(player_id, H, W, model=self.model,
                             params=boom_params(self.params)),
        }

    def act(self, obs):
        self.model.update(obs)
        self.pressure.update(obs)
        cls = classify(self.model, obs.turn, self.pressure, self.params)
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
