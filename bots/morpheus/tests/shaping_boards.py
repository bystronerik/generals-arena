"""Shared boards for prior-shaping tests (Part 17).

Each builder returns ``(obs, memory)``. The set covers both shaping phases:
pre-contact fog expansion and post-contact seek. ``golden_cases`` pairs each
board with a deterministic network prior so a parity fixture can be regenerated
without touching the shaping code.
"""
from __future__ import annotations

import numpy as np

from memory import empty_memory, update_memory
from observe import emit_observation
from state import create_initial_state


def corridor_board():
    """Pre-contact: three-tile north corridor with army stacked off the tip."""
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 7] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    ownership[0, 5, 0] = True
    ownership[0, 6, 0] = True
    armies[7, 0] = 3
    armies[6, 0] = 40
    armies[5, 0] = 2
    state = state._replace(armies=armies, ownership=ownership)
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(8, 8), obs)


def idle_structure_board():
    """Pre-contact: fat idle pile on the general, thin corridor beside it."""
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 7] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    ownership[0, 6, 0] = True
    ownership[0, 5, 0] = True
    armies[7, 0] = 50
    armies[6, 0] = 2
    armies[5, 0] = 2
    state = state._replace(armies=armies, ownership=ownership, time=10)
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(8, 8), obs)


def contact_board():
    """Post-contact: own cluster adjacent to one enemy tile."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[9, 0] = 1
    grid[0, 9] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r, c in ((5, 5), (5, 6), (6, 5)):
        ownership[0, r, c] = True
        neut[r, c] = False
        armies[r, c] = 5
    armies[5, 6] = 30
    ownership[1, 5, 7] = True
    neut[5, 7] = False
    armies[5, 7] = 2
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=50
    )
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(10, 10), obs)


def march_board():
    """Post-contact: long own column with a king stack facing an enemy tile."""
    grid = np.zeros((12, 12), dtype=np.int32)
    grid[11, 5] = 1
    grid[0, 5] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r in range(4, 12):
        ownership[0, r, 5] = True
        neut[r, 5] = False
        armies[r, 5] = 2
    armies[8, 5] = 40
    ownership[1, 3, 5] = True
    neut[3, 5] = False
    armies[3, 5] = 2
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=60
    )
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(12, 12), obs)


def latched_general_board():
    """Post-contact with a latched enemy general behind fog."""
    grid = np.zeros((12, 12), dtype=np.int32)
    grid[11, 5] = 1
    grid[0, 5] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r in range(4, 12):
        ownership[0, r, 5] = True
        neut[r, 5] = False
        armies[r, 5] = 2
    armies[8, 5] = 60
    ownership[1, 3, 5] = True
    neut[3, 5] = False
    armies[3, 5] = 2
    kg = np.zeros((12, 12), dtype=bool)
    kg[0, 5] = True
    og = np.zeros((12, 12), dtype=bool)
    og[11, 5] = True
    mem = empty_memory(12, 12)._replace(known_enemy_general=kg, own_general=og)
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=90
    )
    obs = emit_observation(state, 0)
    mem = update_memory(mem, obs)._replace(known_enemy_general=kg)
    return obs, mem


BOARDS = {
    "corridor": corridor_board,
    "idle_structure": idle_structure_board,
    "contact": contact_board,
    "march": march_board,
    "latched_general": latched_general_board,
}

# Deterministic network priors exercised per board. "uniform" is flat over the
# play mask; "skewed" is a fixed Dirichlet-free ramp that puts most mass on a
# handful of actions, so the blend is tested away from the degenerate flat case.
PRIOR_KINDS = ("uniform", "skewed")


def make_prior(kind: str, mask, *, seed: int = 0):
    """Deterministic legal-normalized prior of the requested shape."""
    mask_a = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(mask_a.sum())
    prior = np.zeros(mask_a.shape, dtype=np.float64)
    if n <= 0:
        return prior
    if kind == "uniform":
        prior[mask_a] = 1.0 / float(n)
        return prior
    if kind == "skewed":
        rng = np.random.default_rng(seed)
        weights = rng.random(n) ** 4 + 1e-9
        prior[mask_a] = weights / float(weights.sum())
        return prior
    raise ValueError(f"unknown prior kind: {kind}")


def golden_cases():
    """Yield ``(case_name, obs, memory, mask, prior)`` for every fixture case."""
    from tactics import play_mask

    for board_name, builder in BOARDS.items():
        obs, mem = builder()
        mask = np.asarray(play_mask(obs, mem), dtype=bool)
        for i, kind in enumerate(PRIOR_KINDS):
            prior = make_prior(kind, mask, seed=i)
            yield f"{board_name}:{kind}", obs, mem, mask, prior
