"""Phase 3 training-loop units: GAE truncation, HL-Gauss, build cost vs
engine, curriculum env == mode preset, config loading.

All `joe`-marked (opt-in gate) — the default suite budget stays untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.joe

import jax
import jax.numpy as jnp

from generals.core.env import GeneralsEnv
from generals.core.game import get_observation
from generals.modifiers.build_castles import build_cost_grid

from training.joe.config import Config
from training.joe.env import competition_env_kwargs, make_competition_env
from training.joe.networks import build_cost_from_obs, compute_build_mask
from training.joe.train.ppo import (
    compute_gae,
    compute_mc_returns,
    hl_gauss_targets,
    make_value_loss_fn,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


# ---- GAE ----


def _col(xs, dtype=jnp.float32):
    return jnp.asarray(xs, dtype=dtype).reshape(-1, 1)


def test_gae_terminated_bootstraps_zero():
    # One env, 3 steps, terminal at t=2 with reward +1. gamma=1, lambda=0.9.
    rewards = _col([0.0, 0.0, 1.0])
    values = _col([0.1, 0.2, 0.3])
    next_values = _col([0.2, 0.3, 0.9])  # t=2 next_value must be ignored
    terminated = _col([0, 0, 1], jnp.bool_)
    truncated = _col([0, 0, 0], jnp.bool_)
    advs = compute_gae(rewards, values, next_values, terminated, truncated, 1.0, 0.9)
    d2 = 1.0 - 0.3                    # bootstrap forced to 0 at termination
    d1 = 0.0 + 0.3 - 0.2
    d0 = 0.0 + 0.2 - 0.1
    assert advs[2, 0] == pytest.approx(d2)
    assert advs[1, 0] == pytest.approx(d1 + 0.9 * d2)
    assert advs[0, 0] == pytest.approx(d0 + 0.9 * (d1 + 0.9 * d2))


def test_gae_truncation_zeroes_carry():
    # Truncation at t=1: its delta bootstraps from the reset state and must
    # not leak backwards — adv[0] is delta[0] alone.
    rewards = _col([0.0, 0.0, 0.0])
    values = _col([0.1, 0.2, 0.3])
    next_values = _col([0.2, 5.0, 0.3])  # poisoned reset-state bootstrap at t=1
    terminated = _col([0, 0, 0], jnp.bool_)
    truncated = _col([0, 1, 0], jnp.bool_)
    advs = compute_gae(rewards, values, next_values, terminated, truncated, 1.0, 0.9)
    assert advs[0, 0] == pytest.approx(0.2 - 0.1)          # no leak from t=1
    assert advs[1, 0] == pytest.approx(5.0 - 0.2)          # wrong, but masked
    assert advs[2, 0] == pytest.approx(0.3 - 0.3 + 0.9 * 0.0)


def test_mc_returns_validity_mask():
    rewards = _col([0.0, 1.0, 0.0])
    terminated = _col([0, 1, 0], jnp.bool_)
    truncated = _col([0, 0, 0], jnp.bool_)
    rets, valid = compute_mc_returns(rewards, terminated, truncated, 1.0)
    # Episode ends at t=1 -> t<=1 valid with return 1; t=2 still running.
    assert valid[:, 0].tolist() == [1.0, 1.0, 0.0]
    assert rets[0, 0] == pytest.approx(1.0)
    assert rets[1, 0] == pytest.approx(1.0)


# ---- HL-Gauss value targets ----


def _ce_cfg():
    return Config(value_loss="ce", num_bins=128, v_min=-1.0, v_max=1.0,
                  hl_sigma=0.04, curriculum=None)


def test_hl_gauss_targets_shape_and_mass():
    cfg = _ce_cfg()
    for ret in (-1.0, -0.37, 0.0, 0.5, 1.0):
        probs = hl_gauss_targets(cfg, jnp.float32(ret))
        assert probs.shape == (128,)
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)
        centers = jnp.linspace(-1.0, 1.0, 128)
        # Peak bin is the one nearest the return
        assert abs(float(centers[int(probs.argmax())]) - ret) <= 2.0 / 127
        # sigma=0.04 concentrates the mass tightly around the return
        near = jnp.abs(centers - ret) < 0.15
        assert float(probs[near].sum()) > 0.999


def test_ce_value_loss_minimized_at_target():
    cfg = _ce_cfg()
    loss_fn = make_value_loss_fn(cfg)
    ret = jnp.float32(0.5)
    target = hl_gauss_targets(cfg, ret)
    logits_match = jnp.log(target + 1e-12)
    logits_off = jnp.log(hl_gauss_targets(cfg, jnp.float32(-0.5)) + 1e-12)
    assert float(loss_fn(logits_match, ret)) < float(loss_fn(logits_off, ret))


def test_value_loss_rejects_unknown():
    with pytest.raises(ValueError):
        make_value_loss_fn(Config(value_loss="huber", curriculum=None))


# ---- Build cost and mask vs the engine, on a live state ----


def _small_build_env():
    return GeneralsEnv(grid_dims=(10, 10), build_castles=True,
                       min_generals_distance=3, pool_size=4, truncation=100)


def test_build_cost_from_obs_matches_engine_on_live_state():
    env = _small_build_env()
    state = env.init_state(jax.random.PRNGKey(0))
    # Give p0 an extra castle so the surcharge kernel has structure to see
    state = state._replace(
        castles=state.castles.at[2, 3].set(True),
        ownership=state.ownership.at[0, 2, 3].set(True),
        armies=state.armies.at[2, 3].set(5),
    )
    for player in (0, 1):
        engine_cost = build_cost_grid(state, player)
        obs = get_observation(state, player)
        obs_cost = build_cost_from_obs(obs)
        assert bool(jnp.all(engine_cost == obs_cost)), f"player {player}"


def test_build_mask_matches_engine_validity():
    env = _small_build_env()
    state = env.init_state(jax.random.PRNGKey(1))
    state = state._replace(armies=state.armies.at[:, :].add(40))
    for player in (0, 1):
        cost = build_cost_grid(state, player)
        obs = get_observation(state, player)
        mask = compute_build_mask(obs, cost.astype(jnp.float32))
        # Engine validity (_apply_one): own cell, plain, armies >= cost
        own = state.ownership[player]
        plain = ~state.generals & ~state.castles
        affords = state.armies >= cost
        assert bool(jnp.all(mask == (own & plain & affords))), f"player {player}"


# ---- Curriculum env kwargs == mode preset (plan section 4) ----

_ENV_ATTRS = [
    "min_grid_size", "max_grid_size", "pad_to", "truncation",
    "mountain_density_range", "num_castles_range", "min_generals_distance",
    "max_generals_distance", "castle_val_range", "perfect_info",
    "build_castles", "deathtouch_turn",
]


def test_final_stage_env_is_bit_identical_to_preset():
    preset_env = GeneralsEnv(mode="competition")
    final_env = make_competition_env(
        preset_env.min_generals_distance, None, preset_env.pool_size)
    for attr in _ENV_ATTRS:
        assert getattr(final_env, attr) == getattr(preset_env, attr), attr


def test_early_stage_env_differs_only_in_distance_window():
    kwargs_early = competition_env_kwargs(2, 6, 100)
    kwargs_final = competition_env_kwargs(17, None, 100)
    assert kwargs_early.pop("min_generals_distance") == 2
    assert kwargs_early.pop("max_generals_distance") == 6
    assert kwargs_final.pop("min_generals_distance") == 17
    assert kwargs_final.pop("max_generals_distance") is None
    assert kwargs_early == kwargs_final


# ---- Config loading ----


@pytest.mark.parametrize("tier", ["S", "M"])
def test_config_loads_frozen_yaml(tier, capsys):
    cfg = Config.from_yaml(str(CONFIG_DIR / f"{tier}.yaml"))
    assert "ignoring unknown config keys" not in capsys.readouterr().out
    stages = cfg.curriculum_stages
    assert stages is not None and len(stages) == 5
    assert stages[0].min_generals_distance == 2
    assert stages[-1].min_generals_distance == 17
    assert stages[-1].max_generals_distance is None
    assert cfg.num_actions == 10 * 21 * 21
    assert cfg.hl_sigma == 0.04
    assert cfg.build_castles is True
