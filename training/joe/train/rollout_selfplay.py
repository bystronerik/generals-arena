"""Rollout collection: self-play environment interaction logic.

Ported from AverageJoe ``train/rollout_selfplay.py`` for the competition
ruleset (plan section 4):

- The per-cell build cost comes from ``build_cost_grid`` on the live state
  and feeds both the 39th observation channel and the build validity mask.
- The network takes the build mask alongside the move mask and emits the
  10-action head; both masks travel in the rollout data for the PPO replay.
- Rewards are computed from ``TimeStep.info.winner`` directly — sparse
  terminal win/loss only (D5), so no next-observation plumbing.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.action import compute_valid_move_mask
from generals.core.game import get_observation
from generals.modifiers.build_castles import build_cost_grid

from training.joe.networks import compute_build_mask, obs_to_array, reset_done_envs
from training.joe.train.rewards import win_lose_reward


def _observe_both(states):
    """Both seats' observations, costs, and masks as single 2N batches."""
    obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
    obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)
    cat = lambda a, b: jax.tree.map(lambda x, y: jnp.concatenate([x, y]), a, b)
    obs_both = cat(obs_p0, obs_p1)
    cost_p0 = jax.vmap(lambda s: build_cost_grid(s, 0))(states)
    cost_p1 = jax.vmap(lambda s: build_cost_grid(s, 1))(states)
    cost = jnp.concatenate([cost_p0, cost_p1]).astype(jnp.float32)
    obs_arr = jax.vmap(obs_to_array)(obs_both)
    move_masks = jax.vmap(
        lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_both)
    build_masks = jax.vmap(compute_build_mask)(obs_both, cost)
    return obs_p0, obs_arr, cost, move_masks, build_masks


@jax.jit(static_argnames=["env", "num_steps", "augment_fn"])
def collect_rollout(states, env, network, key, num_steps,
                    obs_state_p0, obs_state_p1, augment_fn, pool):
    """Collect a self-play rollout: one network plays both seats.

    Concatenates both players into a single 2N batch for obs processing,
    augmentation, and the network forward pass; splits only for env.step.
    Returns training data with env dimension 2N.
    """
    n = states.armies.shape[0]
    cat = lambda a, b: jax.tree.map(lambda x, y: jnp.concatenate([x, y]), a, b)

    osp_init = cat(obs_state_p0, obs_state_p1)

    def scan_body(carry, _):
        states, key, osp, _, _ = carry

        obs_p0, obs_arr, cost, move_masks, build_masks = _observe_both(states)
        obs_aug, new_osp = jax.vmap(augment_fn)(obs_arr, cost, osp)
        obs_aug = obs_aug.astype(jnp.bfloat16)

        temporal = jnp.stack(
            [new_osp.opponent_army_history, new_osp.opponent_land_history], axis=1)

        keys = jrandom.split(key, 2 * n + 1)
        key = keys[0]
        actions, vals, lps, _, _, _ = jax.vmap(
            network, in_axes=(0, 0, 0, 0, 0, None))(
            obs_aug, move_masks, build_masks, temporal, keys[1:], None)

        a0, a1 = actions[:n], actions[n:]
        env_actions = jnp.stack([a0, a1], axis=1)
        timesteps, new_states = jax.vmap(
            lambda s, a: env.step(s, a, pool))(states, env_actions)

        terminated = timesteps.terminated
        truncated = timesteps.truncated
        dones = terminated | truncated
        winners = timesteps.info.winner
        winners_p1 = jnp.where(winners >= 0, 1 - winners, winners)

        rewards_p0 = win_lose_reward(winners)
        rewards_p1 = win_lose_reward(winners_p1)

        # Reset obs state for done envs (both players see the same dones)
        dones_both = jnp.concatenate([dones, dones])
        osp = reset_done_envs(new_osp, dones_both)

        # Mean owned castles (p0 only, scalar diagnostic)
        owned_castles_p0 = (obs_p0.castles * obs_p0.owned_cells).sum()

        data = (
            obs_aug,                                    # (2N, C, H, W) bf16
            move_masks,                                 # (2N, H, W, 4) bool
            build_masks,                                # (2N, H, W) bool
            temporal,                                   # (2N, 2, temporal_window)
            actions,                                    # (2N, 5)
            lps,                                        # (2N,)
            vals,                                       # (2N,)
            jnp.concatenate([rewards_p0, rewards_p1]),  # (2N,)
            jnp.concatenate([terminated, terminated]),  # (2N,)
            jnp.concatenate([truncated, truncated]),    # (2N,)
            jnp.concatenate([winners, winners_p1]),     # (2N,)
            owned_castles_p0,                           # scalar
        )

        return (new_states, key, osp, timesteps.last_state, new_osp), data

    (final_states, final_key, final_osp,
     last_pre_reset, last_osp), rollout_data = jax.lax.scan(
        scan_body,
        (states, key, osp_init, states, osp_init),
        None, length=num_steps,
    )

    (obs, move_masks, build_masks, temporal, actions, lps, vals,
     rews, terminated, truncated, winners, owned_castles) = rollout_data

    # Bootstrap values from the pre-reset final states — single 2N batch
    _, boot_arr, boot_cost, boot_move, boot_build = _observe_both(last_pre_reset)
    boot_aug, boot_osp = jax.vmap(augment_fn)(boot_arr, boot_cost, last_osp)
    boot_aug = boot_aug.astype(jnp.bfloat16)
    boot_temporal = jnp.stack(
        [boot_osp.opponent_army_history, boot_osp.opponent_land_history], axis=1)

    boot_keys = jrandom.split(final_key, 2 * n)
    _, final_val, _, _, _, _ = jax.vmap(
        network, in_axes=(0, 0, 0, 0, 0, None))(
        boot_aug, boot_move, boot_build, boot_temporal, boot_keys, None)

    next_vals = jnp.concatenate([vals[1:], final_val[None]], axis=0)

    rollout_data = (obs, move_masks, build_masks, temporal, actions, lps,
                    vals, next_vals, rews, terminated, truncated, winners,
                    owned_castles)
    final_osp0 = jax.tree.map(lambda x: x[:n], final_osp)
    final_osp1 = jax.tree.map(lambda x: x[n:], final_osp)
    return final_states, rollout_data, final_key, final_osp0, final_osp1
