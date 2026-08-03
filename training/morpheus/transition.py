"""Training adapter over ``GeneralsEnv(mode="competition")``.

Fixed-shape batched states keep ``jax.jit``, ``jax.vmap``, and ``jax.lax.scan``.
Never imported by ``bots/morpheus/`` — the deployment kernel is pure NumPy.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from generals import GeneralsEnv
from generals.core.game import GameInfo, GameState

from training.morpheus.jax_preflight.transition import (
    DEFAULT_NUM_ENVS,
    DEFAULT_POOL_SIZE,
    DEFAULT_SCAN_STEPS,
    compile_scan_transition,
    make_action_sequence,
    make_competition_env,
    make_pool_and_states,
    step_one,
)

# Re-export preflight helpers so Part 02 callers have one import surface.
__all__ = [
    "DEFAULT_NUM_ENVS",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_SCAN_STEPS",
    "compile_batched_transition",
    "compile_scan_transition",
    "make_action_sequence",
    "make_competition_env",
    "make_competition_transition",
    "make_pool_and_states",
    "step_one",
]


def make_competition_transition(
    env: GeneralsEnv | None = None,
) -> Callable[[GameState, jnp.ndarray], tuple[GameState, GameInfo]]:
    """Same composition as ``matchup.make_transition`` for competition mode.

    Builds resolve first; deathtouch wraps the base step. Truncation stays at
    the driver (``env.truncation``), not inside this callable.
    """
    if env is None:
        env = make_competition_env()
    from generals.core import game
    from generals.modifiers import build_castles as bc
    from generals.modifiers import deathtouch as dt

    build_castles = bool(env.build_castles)
    deathtouch_turn = env.deathtouch_turn

    def transition(state: GameState, actions: jnp.ndarray) -> tuple[GameState, GameInfo]:
        if build_castles:
            state, actions = bc.apply_build_actions(state, actions)
        if deathtouch_turn is not None:
            return dt.step(state, actions, deathtouch_turn)
        return game.step(state, actions)

    return transition


def compile_batched_transition(
    env: GeneralsEnv | None = None,
    *,
    seed: int = 0,
    num_envs: int = DEFAULT_NUM_ENVS,
) -> tuple[
    Callable[[GameState, jnp.ndarray], tuple[GameState, Any]],
    GameState,
    GameState,
]:
    """Return ``(scan_fn, pool, init_states)`` for fixed 21×21 competition boards.

    ``scan_fn(states, actions_seq)`` runs ``jax.lax.scan`` over a sequence of
    batched joint actions with shape ``(T, N, 2, 5)``.
    """
    if env is None:
        env = make_competition_env()
    pool, init_states = make_pool_and_states(env, seed=seed, num_envs=num_envs)
    scan_fn = compile_scan_transition(env, pool)
    return scan_fn, pool, init_states
