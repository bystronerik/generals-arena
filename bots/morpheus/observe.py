"""Bundle-safe fogged observation emission from GameState.

Mirrors competition-module ``get_visibility`` + ``get_observation`` + wire
encoding without importing the submodule.
"""
from __future__ import annotations

import numpy as np

from _common.wire import Observation
from memory import (
    TYPE_CASTLE,
    TYPE_FOG,
    TYPE_GENERAL,
    TYPE_MOUNTAIN,
    TYPE_PLAIN,
    TYPE_STRUCTURE_FOG,
)
from state import GameState
from transition import get_info

Array = np.ndarray


def visibility_mask(ownership: Array) -> Array:
    """Chebyshev-1 (3×3) visibility around owned cells."""
    ownership = np.asarray(ownership, dtype=bool)
    H, W = ownership.shape
    padded = np.pad(ownership.astype(np.float32), 1, mode="constant")
    stacked = np.stack(
        [
            padded[0:H, 0:W],
            padded[0:H, 1 : W + 1],
            padded[0:H, 2 : W + 2],
            padded[1 : H + 1, 0:W],
            padded[1 : H + 1, 1 : W + 1],
            padded[1 : H + 1, 2 : W + 2],
            padded[2 : H + 2, 0:W],
            padded[2 : H + 2, 1 : W + 1],
            padded[2 : H + 2, 2 : W + 2],
        ],
        axis=0,
    )
    return stacked.max(axis=0) > 0


def emit_observation(state: GameState, player_idx: int) -> Observation:
    """Perspective-relative wire observation with competition fog."""
    player_idx = int(player_idx)
    opponent_idx = 1 - player_idx
    visible = visibility_mask(state.ownership[player_idx])
    invisible = ~visible
    info = get_info(state)

    H, W = state.armies.shape
    fog = invisible & ~(state.mountains | state.castles)
    structures_fog = invisible & (state.mountains | state.castles)

    type_grid = np.full((H, W), TYPE_PLAIN, dtype=np.int32)
    type_grid[fog] = TYPE_FOG
    type_grid[structures_fog] = TYPE_STRUCTURE_FOG
    # Visible terrain overlays fog defaults.
    type_grid[visible & state.mountains] = TYPE_MOUNTAIN
    type_grid[visible & state.castles] = TYPE_CASTLE
    type_grid[visible & state.generals] = TYPE_GENERAL

    owner_grid = np.zeros((H, W), dtype=np.int32)
    owner_grid[visible & state.ownership[player_idx]] = 1
    owner_grid[visible & state.ownership[opponent_idx]] = 2

    army_grid = np.where(visible, state.armies, 0).astype(np.int32)

    return Observation(
        H=H,
        W=W,
        turn=int(state.time),
        my_land=int(info.land[player_idx]),
        my_army=int(info.army[player_idx]),
        opp_land=int(info.land[opponent_idx]),
        opp_army=int(info.army[opponent_idx]),
        type_grid=type_grid.tolist(),
        owner_grid=owner_grid.tolist(),
        army_grid=army_grid.tolist(),
    )


def observation_grids(obs: Observation) -> tuple[Array, Array, Array]:
    """Return ``(types, owners, armies)`` as int32 arrays."""
    return (
        np.asarray(obs.type_grid, dtype=np.int32),
        np.asarray(obs.owner_grid, dtype=np.int32),
        np.asarray(obs.army_grid, dtype=np.int32),
    )


def observations_match(simulated: Observation, real: Observation) -> bool:
    """Exact match on visible cells, types, owners, armies, turn, public totals."""
    if (
        int(simulated.H) != int(real.H)
        or int(simulated.W) != int(real.W)
        or int(simulated.turn) != int(real.turn)
        or int(simulated.my_land) != int(real.my_land)
        or int(simulated.my_army) != int(real.my_army)
        or int(simulated.opp_land) != int(real.opp_land)
        or int(simulated.opp_army) != int(real.opp_army)
    ):
        return False
    st, so, sa = observation_grids(simulated)
    rt, ro, ra = observation_grids(real)
    return (
        np.array_equal(st, rt)
        and np.array_equal(so, ro)
        and np.array_equal(sa, ra)
    )


def observation_hash(obs: Observation) -> bytes:
    """Stable hash key for information-set / enemy-info deduplication."""
    types, owners, armies = observation_grids(obs)
    payload = np.concatenate(
        [
            np.asarray(
                [
                    obs.H,
                    obs.W,
                    obs.turn,
                    obs.my_land,
                    obs.my_army,
                    obs.opp_land,
                    obs.opp_army,
                ],
                dtype=np.int32,
            ).ravel(),
            types.ravel(),
            owners.ravel(),
            armies.ravel(),
        ]
    )
    return payload.tobytes()
