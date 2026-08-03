"""Compile a competition GeneralsEnv.step under jax.vmap + jax.lax.scan."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv
from generals.core.game import GameState

# Small pool keeps reset cheap; shapes still match competition pad_to=21.
DEFAULT_POOL_SIZE = 32
DEFAULT_NUM_ENVS = 64
DEFAULT_SCAN_STEPS = 50
PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)


def make_competition_env(pool_size: int = DEFAULT_POOL_SIZE) -> GeneralsEnv:
    return GeneralsEnv(mode="competition", pool_size=pool_size)


def make_pool_and_states(
    env: GeneralsEnv,
    *,
    seed: int,
    num_envs: int,
) -> tuple[GameState, GameState]:
    """Return (pool, batched init states) for fixed 21×21 competition boards."""
    key = jrandom.PRNGKey(seed)
    key, pool_key, init_key = jrandom.split(key, 3)
    pool, _ = env.reset(pool_key)
    init_keys = jrandom.split(init_key, num_envs)
    states = jax.vmap(env.init_state)(init_keys)
    return pool, states


def make_action_sequence(
    *,
    seed: int,
    num_envs: int,
    num_steps: int,
) -> jnp.ndarray:
    """Deterministic pass-only action tensor of shape (num_steps, num_envs, 2, 5).

    Pass actions keep the scan focused on transition cost rather than agent
    policy. Fixtures exercise castle build and deathtouch separately.
    """
    del seed  # reserved so callers can later inject seeded non-pass sequences
    one = jnp.stack([PASS, PASS])
    return jnp.tile(one, (num_steps, num_envs, 1, 1))


def compile_scan_transition(
    env: GeneralsEnv,
    pool: GameState,
) -> Callable[[GameState, jnp.ndarray], tuple[GameState, Any]]:
    """Return a jitted (states, actions_seq) -> (final_states, infos) scan.

    Uses `GeneralsEnv.step(state, actions, pool)` with the pool closed over,
    matching competition-module/examples/vectorized_example.py.
    """

    def body(states, actions):
        timesteps, new_states = jax.vmap(
            lambda s, a: env.step(s, a, pool)
        )(states, actions)
        return new_states, timesteps.info

    @jax.jit
    def scan_fn(states, actions_seq):
        return jax.lax.scan(body, states, actions_seq)

    return scan_fn


def step_one(
    env: GeneralsEnv,
    pool: GameState,
    state: GameState,
    actions: jnp.ndarray,
):
    """Single env.step; returns (last_state, info) before auto-reset."""
    timestep, _final = env.step(state, actions, pool)
    return timestep.last_state, timestep.info
