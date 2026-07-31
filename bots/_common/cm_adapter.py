"""
Bridge stdio wire observations to competition-module JAX agents.

Benchmark bots under ``bots/cm_*`` instantiate upstream ``generals.agents``
classes without copying decision logic. Do not retune upstream parameters here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Type

_CM_PATHS_READY = False


def ensure_cm_paths() -> None:
    """Put ``bots/`` and ``competition-module/`` on ``sys.path`` for imports."""
    global _CM_PATHS_READY
    if _CM_PATHS_READY:
        return
    bots_dir = Path(__file__).resolve().parent.parent
    cm_root = bots_dir.parent / "competition-module"
    for entry in (bots_dir, cm_root):
        path = str(entry)
        if path not in sys.path:
            sys.path.insert(0, path)
    _CM_PATHS_READY = True


ensure_cm_paths()

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.core.observation import Observation as CmObservation

from _common.wire import Observation as WireObservation


def wire_to_cm_obs(obs: WireObservation) -> CmObservation:
    """Map competition stdio wire grids to ``generals.core.observation.Observation``."""
    type_g = np.asarray(obs.type_grid, dtype=np.int32)
    owner_g = np.asarray(obs.owner_grid, dtype=np.int32)
    armies = np.asarray(obs.army_grid, dtype=np.int32)

    fog = type_g == 0
    structures_in_fog = type_g == 5
    visible = ~fog

    return CmObservation(
        armies=jnp.array(armies),
        generals=jnp.array(type_g == 4),
        castles=jnp.array(type_g == 3),
        mountains=jnp.array(type_g == 2),
        neutral_cells=jnp.array((owner_g == 0) & visible & ~structures_in_fog),
        owned_cells=jnp.array(owner_g == 1),
        opponent_cells=jnp.array(owner_g == 2),
        fog_cells=jnp.array(fog),
        structures_in_fog=jnp.array(structures_in_fog),
        owned_land_count=jnp.int32(obs.my_land),
        owned_army_count=jnp.int32(obs.my_army),
        opponent_land_count=jnp.int32(obs.opp_land),
        opponent_army_count=jnp.int32(obs.opp_army),
        timestep=jnp.int32(obs.turn),
    )


def cm_action_to_wire(action) -> tuple[int, int, int, int, int]:
    """Convert upstream ``[pass, row, col, dir, split]`` to stdio action tuple."""
    arr = np.asarray(action, dtype=np.int32)
    return (int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]), int(arr[4]))


def make_cm_agent(upstream_class: Type, *, agent_id: str | None = None) -> Type:
    """
    Factory for a stdio ``Agent`` class that delegates to one upstream JAX agent.

    ``upstream_class`` must be a ``generals.agents.Agent`` subclass whose
    ``act(observation, key)`` returns a five-int action array.
    """

    class Agent:
        def __init__(self, player_id: int, H: int, W: int):
            self.player_id = player_id
            self.H = H
            self.W = W
            self._key = jrandom.PRNGKey(player_id)
            init_kwargs = {"id": agent_id} if agent_id is not None else {}
            self._upstream = upstream_class(**init_kwargs)

        def act(self, obs: WireObservation) -> tuple[int, int, int, int, int]:
            self._key, subkey = jrandom.split(self._key)
            cm_obs = wire_to_cm_obs(obs)
            action = self._upstream.act(cm_obs, subkey)
            return cm_action_to_wire(action)

        def reset(self) -> None:
            if hasattr(self._upstream, "reset"):
                self._upstream.reset()

    Agent.__name__ = f"Cm{upstream_class.__name__}"
    Agent.__qualname__ = Agent.__name__
    return Agent


def benchmark_agent(upstream_class: Type, agent_id: str | None = None) -> Callable[[], Type]:
    """Return a zero-arg callable that builds the stdio Agent class (for agent.py)."""

    def _factory() -> Type:
        return make_cm_agent(upstream_class, agent_id=agent_id)

    return _factory
