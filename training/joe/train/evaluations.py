"""Evaluation: greedy network vs random, the curriculum gate (D4).

Ported from AverageJoe ``train/evaluations.py`` minus the reference-Elo
machinery: no frozen reference checkpoints exist under competition rules,
so in-training eval is win-rate vs random only. Exported bots are rated in
the arena (plan section 7, Phase 5).

The random opponent is the engine's ``sample_valid_action`` (uniform over
valid moves, 10% pass). It never builds — fine for a gate whose only job is
to order curriculum stages.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.action import compute_valid_move_mask, sample_valid_action
from generals.core.game import get_observation
from generals.modifiers.build_castles import build_cost_grid

from training.joe.networks import compute_build_mask, obs_to_array, reset_done_envs


def _net_inputs(states, player, augment_fn, obs_st):
    """Observation, masks, and augmented input for one seat."""
    obs = jax.vmap(lambda s: get_observation(s, player))(states)
    cost = jax.vmap(lambda s: build_cost_grid(s, player))(states).astype(jnp.float32)
    obs_arr = jax.vmap(obs_to_array)(obs)
    move = jax.vmap(
        lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs)
    build = jax.vmap(compute_build_mask)(obs, cost)
    obs_aug, obs_st = jax.vmap(augment_fn)(obs_arr, cost, obs_st)
    temporal = jnp.stack(
        [obs_st.opponent_army_history, obs_st.opponent_land_history], axis=1)
    return obs, obs_aug, move, build, temporal, obs_st


@jax.jit(static_argnames=["env", "truncation", "n_maps", "augment_fn", "greedy_fn",
                          "net_player"])
def _evaluate_side(env, network, key, truncation, n_maps, obs_state,
                   augment_fn, greedy_fn, pool, net_player):
    """Play n_maps games with the network on one seat vs random."""
    rand_player = 1 - net_player
    key, init_key, play_key = jrandom.split(key, 3)
    init_keys = jrandom.split(init_key, n_maps)
    states = jax.vmap(env.init_state)(init_keys)

    def scan_body(carry, _):
        states, rng, finished, net_won, net_lost, drew, obs_st = carry
        _, obs_aug, move, build, temporal, obs_st = _net_inputs(
            states, net_player, augment_fn, obs_st)
        net_a = jax.vmap(greedy_fn, in_axes=(None, 0, 0, 0, 0))(
            network, obs_aug, move, build, temporal)

        rand_obs = jax.vmap(lambda s: get_observation(s, rand_player))(states)
        rng, k = jrandom.split(rng)
        rand_a = jax.vmap(sample_valid_action)(jrandom.split(k, n_maps), rand_obs)

        seat_actions = [None, None]
        seat_actions[net_player] = net_a
        seat_actions[rand_player] = rand_a
        actions = jnp.stack(seat_actions, axis=1)
        timesteps, new_states = jax.vmap(
            lambda s, a: env.step(s, a, pool))(states, actions)

        dones = timesteps.terminated | timesteps.truncated
        new_done = dones & ~finished
        net_won = net_won | (new_done & (timesteps.info.winner == net_player))
        net_lost = net_lost | (new_done & (timesteps.info.winner == rand_player))
        drew = drew | (new_done & timesteps.truncated & ~timesteps.terminated)
        finished = finished | dones
        obs_st = reset_done_envs(obs_st, dones)
        return (new_states, rng, finished, net_won, net_lost, drew, obs_st), None

    z = jnp.zeros(n_maps, jnp.bool_)
    (_, _, fin, won, lost, drew, _), _ = jax.lax.scan(
        scan_body, (states, play_key, z, z, z, z, obs_state), None,
        length=truncation)
    return fin, won, lost, drew


@jax.jit(static_argnames=["env", "truncation", "n_maps", "augment_fn", "greedy_fn",
                          "net_player"])
def _evaluate_side_vs_ref(env, network, ref_network, key, truncation, n_maps,
                          obs_state_net, obs_state_ref, augment_fn, greedy_fn,
                          pool, net_player):
    """Play n_maps games with the network on one seat vs the frozen reference.

    Both seats play greedy (argmax) — the deployment policy — so outcomes
    are deterministic per map and variance comes from the map sample. Each
    seat carries its own AugmentedObsState.
    """
    ref_player = 1 - net_player
    key, init_key = jrandom.split(key)
    init_keys = jrandom.split(init_key, n_maps)
    states = jax.vmap(env.init_state)(init_keys)

    def scan_body(carry, _):
        states, finished, net_won, net_lost, drew, obs_net, obs_ref = carry
        _, obs_aug_n, move_n, build_n, temporal_n, obs_net = _net_inputs(
            states, net_player, augment_fn, obs_net)
        net_a = jax.vmap(greedy_fn, in_axes=(None, 0, 0, 0, 0))(
            network, obs_aug_n, move_n, build_n, temporal_n)

        _, obs_aug_r, move_r, build_r, temporal_r, obs_ref = _net_inputs(
            states, ref_player, augment_fn, obs_ref)
        ref_a = jax.vmap(greedy_fn, in_axes=(None, 0, 0, 0, 0))(
            ref_network, obs_aug_r, move_r, build_r, temporal_r)

        seat_actions = [None, None]
        seat_actions[net_player] = net_a
        seat_actions[ref_player] = ref_a
        actions = jnp.stack(seat_actions, axis=1)
        timesteps, new_states = jax.vmap(
            lambda s, a: env.step(s, a, pool))(states, actions)

        dones = timesteps.terminated | timesteps.truncated
        new_done = dones & ~finished
        net_won = net_won | (new_done & (timesteps.info.winner == net_player))
        net_lost = net_lost | (new_done & (timesteps.info.winner == ref_player))
        drew = drew | (new_done & timesteps.truncated & ~timesteps.terminated)
        finished = finished | dones
        obs_net = reset_done_envs(obs_net, dones)
        obs_ref = reset_done_envs(obs_ref, dones)
        return (new_states, finished, net_won, net_lost, drew,
                obs_net, obs_ref), None

    z = jnp.zeros(n_maps, jnp.bool_)
    (_, fin, won, lost, drew, _, _), _ = jax.lax.scan(
        scan_body, (states, z, z, z, z, obs_state_net, obs_state_ref), None,
        length=truncation)
    return fin, won, lost, drew


def evaluate_vs_ref(env, network, ref_network, key, truncation, n_maps,
                    obs_state, augment_fn, greedy_fn, pool):
    """Play n_maps games per seat (network as p0, then as p1) vs the
    frozen reference network. Returns (wins, losses, draws, finished)."""
    key, k0, k1 = jrandom.split(key, 3)
    fin0, won0, lost0, drew0 = _evaluate_side_vs_ref(
        env, network, ref_network, k0, truncation, n_maps, obs_state,
        obs_state, augment_fn, greedy_fn, pool, 0)
    fin1, won1, lost1, drew1 = _evaluate_side_vs_ref(
        env, network, ref_network, k1, truncation, n_maps, obs_state,
        obs_state, augment_fn, greedy_fn, pool, 1)
    wins = int(jnp.sum(won0) + jnp.sum(won1))
    losses = int(jnp.sum(lost0) + jnp.sum(lost1))
    draws = int(jnp.sum(drew0) + jnp.sum(drew1))
    finished = int(jnp.sum(fin0) + jnp.sum(fin1))
    return wins, losses, draws, finished


def should_eval(it, eval_freq):
    """The single eval-cadence predicate (it==0 gives a baseline)."""
    return it == 0 or (it + 1) % eval_freq == 0


def evaluate(env, network, key, truncation, n_maps, obs_state,
             augment_fn, greedy_fn, pool):
    """Play n_maps games per seat (network as p0, then as p1) vs random.

    Returns (wins, losses, draws, finished) as ints.
    """
    key, k0, k1 = jrandom.split(key, 3)
    fin0, won0, lost0, drew0 = _evaluate_side(
        env, network, k0, truncation, n_maps, obs_state,
        augment_fn, greedy_fn, pool, 0)
    fin1, won1, lost1, drew1 = _evaluate_side(
        env, network, k1, truncation, n_maps, obs_state,
        augment_fn, greedy_fn, pool, 1)
    wins = int(jnp.sum(won0) + jnp.sum(won1))
    losses = int(jnp.sum(lost0) + jnp.sum(lost1))
    draws = int(jnp.sum(drew0) + jnp.sum(drew1))
    finished = int(jnp.sum(fin0) + jnp.sum(fin1))
    return wins, losses, draws, finished


def periodic_eval(it, cfg, eval_freq, network, eval_env, eval_pool,
                  single_state, augment_fn, greedy_fn, logger, key,
                  last_eval_wr, ema_network=None, ref_network=None):
    """Run the vs-random eval on eval iters; log results.

    Returns (eval_ran, last_eval_wr, key). last_eval_wr updates only when
    the eval runs — it drives curriculum advancement. When ``ref_network``
    (and the EMA net) are given, the frozen-reference eval also runs on
    the same cadence; it is a pure metric and gates nothing.
    """
    eval_ran = False
    if should_eval(it, eval_freq):
        key, eval_key = jrandom.split(key)
        n_maps = cfg.eval_games // 2
        eval_obs_state = jax.tree.map(
            lambda x: jnp.tile(x, (n_maps, *([1] * x.ndim))), single_state)
        ew, el, ed, edone = evaluate(
            eval_env, network, eval_key, cfg.truncation, n_maps,
            eval_obs_state, augment_fn, greedy_fn, eval_pool)
        last_eval_wr = ew / max(edone, 1)
        eval_ran = True
        print(f"  EVAL: {ew}W/{el}L/{ed}D ({last_eval_wr * 100:.0f}%) "
              f"greedy vs random over {edone} games", flush=True)
        logger.log_eval(it, ew, el, ed, edone)

        if ref_network is not None and ema_network is not None:
            key, ref_key = jrandom.split(key)
            rw, rl, rd, rdone = evaluate_vs_ref(
                eval_env, ema_network, ref_network, ref_key, cfg.truncation,
                n_maps, eval_obs_state, augment_fn, greedy_fn, eval_pool)
            ref_wr = rw / max(rdone, 1)
            print(f"  REF EVAL: {rw}W/{rl}L/{rd}D ({ref_wr * 100:.0f}%) "
                  f"EMA greedy vs reference over {rdone} games", flush=True)
            logger.log(it, {
                "eval_ref/wins": rw, "eval_ref/losses": rl,
                "eval_ref/draws": rd, "eval_ref/finished": rdone,
                "eval_ref/win_rate": ref_wr,
            })
    return eval_ran, last_eval_wr, key
