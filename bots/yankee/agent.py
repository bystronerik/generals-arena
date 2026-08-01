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

import time

from _common.oppmodel import OpponentModel
from yankee.blitz_core import BlitzCore
from yankee.boom_core import BoomCore
from yankee.classifier import classify
from yankee.deathtouch import DeathtouchCore
from yankee.params import (
    YankeeParams,
    blitz_config,
    boom_params,
    deathtouch_config,
    load_params,
)
from yankee.search import TacticalSearch
from yankee.signals import HomePressure
from yankee.switcher import DEATHTOUCH, Switcher

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
        if self.params.deathtouch_enabled:
            # Constructed only when reachable. proteus's §2 records what
            # happens otherwise: it warmed four cores the switcher could select
            # two of, and paid two `observe()` calls a turn to reach a branch
            # that did not exist.
            self.cores[DEATHTOUCH] = DeathtouchCore(
                player_id, H, W, model=self.model,
                config=deathtouch_config(self.params),
            )
        # §02's move-order tie-break goes to player 0, and the owner codes in
        # the observation are perspective-relative, so the search needs the
        # real seat handed to it here or its forward model breaks ties by
        # coin flip where the engine breaks them by fact.
        self.search = TacticalSearch(player_id, self.params)
        self.last_move_ms = 0.0

    def act(self, obs):
        started = time.perf_counter()
        self.model.update(obs)
        self.pressure.update(obs)
        cls = classify(self.model, obs.turn, self.pressure, self.params)
        active = self.switcher.update(cls, obs.turn)

        # Keep the inactive core's beliefs/latches current for a warm handover.
        for name, core in self.cores.items():
            if name != active:
                core.observe(obs)

        # The blitz core's standing guard keys off "this opponent has walked a
        # sized stack at our general", which is exactly `HomePressure`'s latched
        # conjunction and is not derivable from the shared OpponentModel (which
        # latches "big" and "near" independently — the defect proteus's signals
        # module exists to fix). Push it down rather than teach the core to
        # recompute it.
        self.cores["blitz"].memory.fist_seen = self.pressure.duel_turn is not None

        core = self.cores[active]
        move = core.decide(obs)

        # The search is a filter on a move that already exists, never a
        # replacement for computing one: whatever it does or fails to do in
        # its budget, `move` is already a legal reply (RULES.md §08 — a late
        # or missing reply is a fault, and 50 forfeit the game).
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        move = self.search.improve(
            obs, move, self._my_general(core), self._enemy_general(core),
            elapsed_ms, active,
        )
        self.last_move_ms = (time.perf_counter() - started) * 1000.0
        return move

    @staticmethod
    def _my_general(core):
        return getattr(core.memory, "my_general", None)

    @staticmethod
    def _enemy_general(core):
        belief = getattr(core.memory, "belief", None)
        return getattr(belief, "enemy_general", None) if belief else None

    @property
    def label(self) -> str:
        """Current opponent classification label (for traces/tests)."""
        return self.switcher.label

    @property
    def active_strategy(self) -> str:
        return self.switcher.current
