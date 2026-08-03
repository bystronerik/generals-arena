"""Hand-crafted 21×21 competition-state fixtures for castle build and deathtouch.

States are padded with mountains to the competition `pad_to=21` shape so they
match `GeneralsEnv(mode="competition")` pool / step shapes.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from generals.core.game import GameState, create_initial_state
from generals.modifiers import build_castles as bc

PAD = 21
PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
RIGHT, LEFT, DOWN, UP = 3, 2, 1, 0


class ParityFixture(NamedTuple):
    name: str
    state: GameState
    actions: jnp.ndarray  # (2, 5)


def _open_padded(playable: int = 6) -> jnp.ndarray:
    """Mountain-padded pad_to board with an open playable corner."""
    grid = jnp.full((PAD, PAD), -2, dtype=jnp.int32)
    grid = grid.at[:playable, :playable].set(0)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[0, playable - 1].set(2)
    return grid


def _give(state: GameState, player: int, ij: tuple[int, int], army: int) -> GameState:
    i, j = ij
    return state._replace(
        armies=state.armies.at[i, j].set(army),
        ownership=state.ownership.at[player, i, j].set(True),
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
    )


def _move(i: int, j: int, d: int) -> jnp.ndarray:
    return jnp.array([0, i, j, d, 0], dtype=jnp.int32)


def _build(i: int, j: int) -> jnp.ndarray:
    return jnp.array([bc.BUILD, i, j, 0, 0], dtype=jnp.int32)


def castle_build_fixture() -> ParityFixture:
    """P0 builds a castle on owned land with enough army to pay the cost."""
    state = create_initial_state(_open_padded(6))
    # Cell (0, 3) is d=3 from the general → cost 43. Leave a remainder.
    state = _give(state, 0, (0, 3), 60)
    actions = jnp.stack([_build(0, 3), PASS])
    return ParityFixture("castle_build", state, actions)


def deathtouch_fixture() -> ParityFixture:
    """P0 touches P1's general on turn 800 with a small stack (combat would lose)."""
    state = create_initial_state(_open_padded(6))
    state = state._replace(time=jnp.int32(800))
    state = _give(state, 0, (0, 4), 5)
    state = state._replace(armies=state.armies.at[0, 5].set(50))
    actions = jnp.stack([_move(0, 4, RIGHT), PASS])
    return ParityFixture("deathtouch", state, actions)


def deathtouch_chase_defense_fixture() -> ParityFixture:
    """Chase defense kills the touch source before deathtouch resolves."""
    state = create_initial_state(_open_padded(6))
    state = state._replace(time=jnp.int32(800))
    state = _give(state, 0, (0, 4), 5)
    state = _give(state, 1, (1, 4), 10)
    actions = jnp.stack([_move(0, 4, RIGHT), _move(1, 4, UP)])
    return ParityFixture("deathtouch_chase_defense", state, actions)


def pass_noop_fixture() -> ParityFixture:
    """Both seats pass once; exercises growth and time advance only."""
    state = create_initial_state(_open_padded(6))
    actions = jnp.stack([PASS, PASS])
    return ParityFixture("pass_noop", state, actions)


def all_parity_fixtures() -> list[ParityFixture]:
    return [
        castle_build_fixture(),
        deathtouch_fixture(),
        deathtouch_chase_defense_fixture(),
        pass_noop_fixture(),
    ]
