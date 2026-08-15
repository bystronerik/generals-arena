"""Deployment fidelity vs the training path, on a live engine (marker joe).

Three layers, each pinning one seam of the Phase 5 port
(docs/research/strategies/averagejoe-competition-plan.md):

1. Wire fidelity: engine state -> ``encode_observation`` -> the bot's wire
   parsing reproduces ``obs_to_array`` bit-for-bit, and the bot-local build
   cost / move mask / build mask equal the engine-side ones, every step of a
   real build_castles game.
2. Code parity: the bot-local copies (``joe_obs``, ``joe_net``) match
   ``training/joe/networks`` — identical augmented obs on a shared rollout,
   and identical network outputs after serializing a training net into the
   bot's class.
3. End to end: the deployed ``Agent`` (artifact weights, jit step, wire
   parsing) picks the exact action the training eval path
   (``greedy_action_transformer`` on state-side inputs) picks. Skipped when
   the gitignored artifact has not been exported.

Every heavy import (jax, equinox, the engine, the training package) lives
inside the module-scoped ``deps`` fixture: collecting this file in the
default suite must not load jax into the shared pytest process — that
measurably slows every other test (GC over a much larger heap).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.joe

BOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = BOT_DIR / "artifact"


@pytest.fixture(scope="module")
def deps():
    _comp = str(REPO_ROOT / "competition-module")
    if _comp not in sys.path:
        sys.path.insert(0, _comp)

    import jax.numpy as jnp
    import jax.random as jrandom
    import numpy as np

    from _common.wire import _read_observation
    from competition.protocol import encode_observation
    from generals.core.action import compute_valid_move_mask, sample_valid_action
    from generals.core.env import GeneralsEnv
    from generals.core.game import get_observation
    from generals.modifiers.build_castles import build_cost_grid

    import joe_net
    import joe_obs
    from agent import Agent, frame_to_raw
    from training.joe import networks as train_nets
    from training.joe.networks import transformer as train_transformer

    d = SimpleNamespace(**{k: v for k, v in locals().items()
                           if not k.startswith("_comp")})

    def small_env():
        return GeneralsEnv(grid_dims=(10, 10), build_castles=True,
                           min_generals_distance=3, pool_size=4,
                           truncation=100)

    def wire_roundtrip(obs):
        """Engine Observation -> wire text -> the bot's parsed frame."""
        text = encode_observation(obs)
        stdin = io.StringIO(text)
        first = stdin.readline()
        H, W = obs.armies.shape
        return _read_observation(stdin, H, W, first)

    def play(env, num_steps, seed=0, on_state=None):
        """Random build_castles game; calls ``on_state(state, t)`` per step."""
        key = jrandom.PRNGKey(seed)
        key, reset_key = jrandom.split(key)
        pool, state = env.reset(reset_key)
        for t in range(num_steps):
            if on_state is not None:
                on_state(state, t)
            key, k0, k1 = jrandom.split(key, 3)
            actions = jnp.stack([
                sample_valid_action(k0, get_observation(state, 0)),
                sample_valid_action(k1, get_observation(state, 1)),
            ])
            _, state = env.step(state, actions, pool)

    d.small_env = small_env
    d.wire_roundtrip = wire_roundtrip
    d.play = play
    d.TINY = dict(grid_size=21, pad_to=21, history_size=7, patch_size=3,
                  depth=1, embed_dim=48, n_head=4, ff_factor=2,
                  use_bf16=False, value_loss="ce", num_bins=16)
    return d


# ---- 1. Wire fidelity ----


def test_wire_frame_matches_training_tensor_every_step(deps):
    jnp, np = deps.jnp, deps.np
    env = deps.small_env()
    checked = []

    def check(state, t):
        for player in (0, 1):
            obs = deps.get_observation(state, player)
            train_raw = np.asarray(deps.train_nets.obs_to_array(obs))
            bot_raw = deps.frame_to_raw(deps.wire_roundtrip(obs))
            np.testing.assert_array_equal(
                bot_raw, train_raw, err_msg=f"step {t} player {player}")

            engine_cost = np.asarray(deps.build_cost_grid(state, player))
            bot_cost = np.asarray(
                deps.joe_obs.build_cost_from_raw(jnp.asarray(bot_raw)))
            np.testing.assert_array_equal(
                bot_cost, engine_cost, err_msg=f"cost step {t} player {player}")

            engine_move = np.asarray(deps.compute_valid_move_mask(
                obs.armies, obs.owned_cells, obs.mountains))
            bot_move = np.asarray(deps.joe_obs.compute_valid_move_mask(
                jnp.asarray(bot_raw[0]), jnp.asarray(bot_raw[5] > 0),
                jnp.asarray(bot_raw[3] > 0)))
            np.testing.assert_array_equal(
                bot_move, engine_move, err_msg=f"move step {t} player {player}")

            train_build = np.asarray(deps.train_nets.compute_build_mask(
                obs, jnp.asarray(engine_cost, dtype=jnp.float32)))
            bot_build = np.asarray(deps.joe_obs.compute_build_mask_from_raw(
                jnp.asarray(bot_raw), jnp.asarray(bot_cost)))
            np.testing.assert_array_equal(
                bot_build, train_build, err_msg=f"build step {t} player {player}")
        checked.append(t)

    deps.play(env, num_steps=40, on_state=check)
    assert len(checked) == 40


# ---- 2. Code parity: bot-local copies vs training/joe/networks ----


def test_augment_obs_parity_over_rollout(deps):
    jnp, np = deps.jnp, deps.np
    env = deps.small_env()
    states = {"train": deps.train_nets.init_obs_state(21),
              "bot": deps.joe_obs.init_obs_state(21)}

    def check(state, t):
        obs = deps.get_observation(state, 0)
        raw = deps.train_nets.obs_to_array(obs)
        cost = deps.build_cost_grid(state, 0).astype(jnp.float32)
        train_aug, states["train"] = deps.train_nets.augment_obs(
            raw, cost, states["train"])
        bot_aug, states["bot"] = deps.joe_obs.augment_obs(
            raw, cost, states["bot"])
        np.testing.assert_array_equal(
            np.asarray(bot_aug), np.asarray(train_aug), err_msg=f"step {t}")
        for a, b in zip(states["train"], states["bot"]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    deps.play(env, num_steps=25, on_state=check)


def test_network_parity_after_serialization_roundtrip(deps, tmp_path):
    import equinox as eqx

    jnp, np, jrandom = deps.jnp, deps.np, deps.jrandom
    train_net = deps.train_transformer.HistoryTransformer(
        **deps.TINY, key=jrandom.PRNGKey(42))
    path = tmp_path / "tiny.eqx"
    eqx.tree_serialise_leaves(str(path), train_net)
    bot_net = eqx.tree_deserialise_leaves(
        str(path),
        deps.joe_net.HistoryTransformer(**deps.TINY, key=jrandom.PRNGKey(7)))

    rng = np.random.default_rng(3)
    obs = jnp.asarray(rng.normal(size=(39, 21, 21)).astype(np.float32))
    move = jnp.asarray(rng.random((21, 21, 4)) < 0.3)
    build = jnp.asarray(rng.random((21, 21)) < 0.2)
    temporal = jnp.asarray(rng.normal(size=(2, 512)).astype(np.float32))

    t_logits, t_value, _ = train_net._forward(obs, move, build, temporal)
    b_logits, b_value, _ = bot_net._forward(obs, move, build, temporal)
    np.testing.assert_array_equal(np.asarray(b_logits), np.asarray(t_logits))
    np.testing.assert_array_equal(np.asarray(b_value), np.asarray(t_value))

    t_act = deps.train_transformer.greedy_action_transformer(
        train_net, obs, move, build, temporal)
    b_act = deps.joe_net.greedy_action_transformer(
        bot_net, obs, move, build, temporal)
    np.testing.assert_array_equal(np.asarray(b_act), np.asarray(t_act))


# ---- 3. End to end: deployed Agent == training eval path ----


@pytest.mark.skipif(not (ARTIFACT_DIR / "ema.eqx").exists(),
                    reason="no exported artifact (scripts/joe_export_bot.py)")
def test_agent_matches_training_eval_path(deps):
    import equinox as eqx

    jnp, np, jrandom = deps.jnp, deps.np, deps.jrandom
    with open(ARTIFACT_DIR / "manifest.json") as f:
        arch = json.load(f)["network"]
    ref_net = eqx.tree_deserialise_leaves(
        str(ARTIFACT_DIR / "ema.eqx"),
        deps.train_transformer.HistoryTransformer(
            grid_size=int(arch["pad_to"]), pad_to=int(arch["pad_to"]),
            history_size=int(arch["history_size"]),
            patch_size=int(arch["patch_size"]), depth=int(arch["depth"]),
            embed_dim=int(arch["embed_dim"]), n_head=int(arch["n_head"]),
            ff_factor=int(arch["ff_factor"]), use_bf16=False,
            value_loss=arch["value_loss"], num_bins=int(arch["num_bins"]),
            v_min=float(arch["v_min"]), v_max=float(arch["v_max"]),
            key=jrandom.PRNGKey(0)))

    env = deps.small_env()
    agent = deps.Agent(player_id=0, H=10, W=10)
    ref = {"st": deps.train_nets.init_obs_state(int(arch["pad_to"])), "n": 0}

    def check(state, t):
        obs = deps.get_observation(state, 0)
        # Training eval path (train/evaluations.py), single sample
        raw = deps.train_nets.obs_to_array(obs)
        cost = deps.build_cost_grid(state, 0).astype(jnp.float32)
        aug, ref["st"] = deps.train_nets.augment_obs(raw, cost, ref["st"])
        move = deps.compute_valid_move_mask(
            obs.armies, obs.owned_cells, obs.mountains)
        build = deps.train_nets.compute_build_mask(obs, cost)
        temporal = jnp.stack([ref["st"].opponent_army_history,
                              ref["st"].opponent_land_history])
        want = tuple(int(x) for x in np.asarray(
            deps.train_transformer.greedy_action_transformer(
                ref_net, aug, move, build, temporal)))
        if want[0] == 1:
            want = (1, 0, 0, 0, 0)

        got = agent.act(deps.wire_roundtrip(obs))

        # The inference stack is pinned on the *unpenalised* logits: whatever
        # the repetition penalty does to the selection, the network the bot runs
        # must still be the network training ran, fed the same tensor.
        assert int(jnp.argmax(agent.logits)) == int(jnp.argmax(
            deps.train_transformer.HistoryTransformer._forward(
                ref_net, aug, move, build, temporal)[0])), (
            f"step {t}: the bot's logits disagree with the training path")

        # The emitted action may differ from the greedy one only when the
        # penalty says so, and never on the first move, where visits are zero.
        if not bool(agent.overrode):
            assert got == want, f"step {t}: agent {got} != training path {want}"
        else:
            ref["n"] += 1
        if t == 0:
            assert not bool(agent.overrode), "first move must be the raw argmax"

    deps.play(env, num_steps=15, on_state=check)


# ---- 4. The repetition penalty ----


@pytest.mark.skipif(not (ARTIFACT_DIR / "ema.eqx").exists(),
                    reason="no exported artifact (scripts/joe_export_bot.py)")
def test_repetition_penalty_breaks_a_frozen_board(deps):
    """A board that never changes must not produce an action that never changes.

    This is the limit cycle in its purest form: the network is a deterministic
    function of the observation, so replaying one frame forever pins the raw
    argmax to a single action. Only the visit counter can break it, which makes
    this a direct test of the fix rather than of the policy.
    """
    jnp, np = deps.jnp, deps.np
    from collections import Counter

    # Replay one mid-game frame, not the opening: on turn 0 joe holds a single
    # army and can only pass, and pass carries no cell, so by design it never
    # accumulates a penalty and this test would pass while measuring nothing.
    env = deps.small_env()
    frame = {}
    deps.play(env, num_steps=40, on_state=lambda state, t: frame.__setitem__(
        "f", deps.wire_roundtrip(deps.get_observation(state, 0))))

    agent = deps.Agent(player_id=0, H=10, W=10)
    assert agent.act(frame["f"]) is not None
    assert not bool(agent.overrode), "a fresh agent must take the raw argmax"

    raw, emitted, overrides = [], [], 0
    for _ in range(24):
        emitted.append(agent.act(frame["f"]))
        raw.append(int(jnp.argmax(agent.logits)))
        overrides += int(bool(agent.overrode))

    # The frame never changes, so the network settles on one preferred action
    # and keeps returning it. That fixed preference is the thing to break; if it
    # is absent this fixture has stopped reproducing the pathology.
    choice, repeats = Counter(raw).most_common(1)[0]
    assert repeats >= 10, (
        f"the network did not settle on a frozen board ({repeats}/24 repeats); "
        f"this fixture no longer reproduces the limit cycle")

    assert overrides > 0, (
        "the penalty never fired on a frozen board: joe would still cycle")
    stuck = sum(1 for i, a in zip(raw, emitted)
                if i == choice and a == emitted[raw.index(choice)])
    assert stuck < repeats, (
        f"joe emitted the frozen action {stuck} times while the network asked "
        f"for it {repeats} times: the penalty did not break the repeat")
    assert len(set(emitted)) > 1, (
        "joe emitted a single action on every turn of a frozen board")

    # The penalty must stay bounded, or a long game could eventually push a
    # legal action below the -1e9 the masks use for illegal ones. Repeated
    # +1-then-decay is geometric, so it can never pass 1 / (1 - decay).
    from agent import REPEAT_DECAY, REPEAT_PENALTY
    cap = 1.0 / (1.0 - REPEAT_DECAY)
    peak = float(np.asarray(agent.visits).max())
    assert peak <= cap, f"visit count {peak} passed the geometric bound {cap}"
    assert REPEAT_PENALTY * cap < 1e6, "penalty could reach the mask sentinel"
