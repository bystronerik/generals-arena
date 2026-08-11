"""Shared network utilities, ported from AverageJoe ``networks/common.py``.

Competition-mode deltas from the released code (plan section 4):

- The augmented observation has **25 base channels** instead of 24: channel
  24 is the own build cost per cell (decision 4), pre-scaled by 1/50 like the
  other army-denominated channels. With the default history of 7 the full
  observation is 25 + 2*7 = **39 channels**.
- Actions have **10 per-cell channels** instead of 9: 0-3 full move, 4-7
  half move, 8 pass, 9 build. The build channel maps to pass-field 2 in the
  engine's action array (``generals/modifiers/build_castles.py``).
- ``castles`` replaces the classic ``cities`` channel — same slot, same
  meaning under ``Observation.castles``.

Everything else (history stacks, seen masks, padding-as-mountains, temporal
windows, the 1/50 normalization) is the released recipe unchanged.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

# Base channels: the paper's 24 plus the build-cost channel.
N_BASE_CHANNELS = 25
DEFAULT_HISTORY = 7
N_CHANNELS = N_BASE_CHANNELS + 2 * DEFAULT_HISTORY  # 39
N_ACTION_CHANNELS = 10

# Build-cost constants mirror generals/modifiers/build_castles.py. Kept as a
# local copy so obs-side cost computation has no state dependency; the Phase 3
# suite asserts equality against build_cost_grid on live states.
_BUILD_BASE_COST = 35
_BUILD_PROXIMITY_PENALTY = 14
_BUILD_PROXIMITY_DECAY = 2
_BUILD_RADIUS = 6


# ---- Observation state (augmented networks) ----


class AugmentedObsState(NamedTuple):
    army_stack: Array                     # (history_size, pad_to, pad_to)
    enemy_stack: Array                    # (history_size, pad_to, pad_to)
    last_army: Array                      # (pad_to, pad_to)
    last_enemy_army: Array                # (pad_to, pad_to)
    castles: Array                        # (pad_to, pad_to) bool
    generals: Array                       # (pad_to, pad_to) bool
    mountains: Array                      # (pad_to, pad_to) bool
    seen: Array                           # (pad_to, pad_to) bool
    enemy_seen: Array                     # (pad_to, pad_to) bool
    last_enemy_army_seen_value: Array     # (pad_to, pad_to)
    last_enemy_army_seen_timestep: Array  # (pad_to, pad_to)
    opponent_army_history: Array          # (temporal_window,) — time series
    opponent_land_history: Array          # (temporal_window,) — time series
    temporal_step: Array                  # () int — current write index


def init_obs_state(grid_size, pad_to=None, history_size=DEFAULT_HISTORY,
                   temporal_window=512):
    """Create zero-initialized obs state for a single environment."""
    p = pad_to if pad_to is not None else grid_size
    return AugmentedObsState(
        army_stack=jnp.zeros((history_size, p, p)),
        enemy_stack=jnp.zeros((history_size, p, p)),
        last_army=jnp.zeros((p, p)),
        last_enemy_army=jnp.zeros((p, p)),
        castles=jnp.zeros((p, p), dtype=jnp.bool_),
        generals=jnp.zeros((p, p), dtype=jnp.bool_),
        mountains=jnp.zeros((p, p), dtype=jnp.bool_),
        seen=jnp.zeros((p, p), dtype=jnp.bool_),
        enemy_seen=jnp.zeros((p, p), dtype=jnp.bool_),
        last_enemy_army_seen_value=jnp.zeros((p, p)),
        last_enemy_army_seen_timestep=jnp.zeros((p, p)),
        opponent_army_history=jnp.zeros((temporal_window,)),
        opponent_land_history=jnp.zeros((temporal_window,)),
        temporal_step=jnp.int32(0),
    )


def reset_obs_state(obs_state):
    """Reset all state to zeros."""
    return jax.tree.map(jnp.zeros_like, obs_state)


def reset_done_envs(obs_state, dones):
    """Reset obs_state for envs where done=True. Works on batched (N, ...) state."""
    def reset_leaf(leaf):
        expand = dones.reshape(dones.shape[0], *([1] * (leaf.ndim - 1)))
        return jnp.where(expand, jnp.zeros_like(leaf), leaf)
    return jax.tree.map(reset_leaf, obs_state)


def _max_pool_2d(x, window_size=3):
    """Max pool a single (H, W) array with SAME padding."""
    x = x[None, :, :]
    pooled = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max,
        (1, window_size, window_size), (1, 1, 1), 'SAME'
    )
    return pooled[0, :, :]


# ---- Observation conversion ----


def obs_to_array(obs):
    """Convert Observation to (14, H, W) float array.

    Channels 0-8: spatial grids, 9-13: scalar stats broadcast to (H, W).
    """
    shape = obs.armies.shape
    return jnp.stack(
        [
            obs.armies,
            obs.generals,
            obs.castles,
            obs.mountains,
            obs.neutral_cells,
            obs.owned_cells,
            obs.opponent_cells,
            obs.fog_cells,
            obs.structures_in_fog,
            jnp.broadcast_to(obs.owned_land_count, shape),
            jnp.broadcast_to(obs.owned_army_count, shape),
            jnp.broadcast_to(obs.opponent_land_count, shape),
            jnp.broadcast_to(obs.opponent_army_count, shape),
            jnp.broadcast_to(obs.timestep, shape),
        ],
        axis=0,
    ).astype(jnp.float32)


def build_cost_from_obs(obs):
    """(H, W) own castle price per cell, computed from an Observation.

    Same kernel as ``build_cost_grid`` in the build_castles modifier: 35 base
    plus max(0, 14 - 2d) per own structure at manhattan distance d. Own
    structures are always visible, so the observation carries everything the
    state-side function reads.
    """
    H, W = obs.armies.shape
    structures = ((obs.castles | obs.generals) & obs.owned_cells).astype(jnp.int32)
    padded = jnp.pad(structures, _BUILD_RADIUS)
    cost = jnp.full((H, W), _BUILD_BASE_COST, dtype=jnp.int32)
    R = _BUILD_RADIUS
    for di in range(-R, R + 1):
        for dj in range(-R, R + 1):
            surcharge = _BUILD_PROXIMITY_PENALTY - _BUILD_PROXIMITY_DECAY * (abs(di) + abs(dj))
            if surcharge > 0:
                shifted = padded[R + di:R + di + H, R + dj:R + dj + W]
                cost = cost + surcharge * shifted
    return cost


def compute_build_mask(obs, build_cost):
    """(H, W) bool mask of cells where a build action is valid.

    Engine validity (``build_castles._apply_one``): own cell, plain (no
    general/castle), and armies >= cost.
    """
    plain = ~(obs.generals > 0) & ~(obs.castles > 0)
    return obs.owned_cells & plain & (obs.armies >= build_cost)


def augment_obs(obs_arr, build_cost, obs_state):
    """Augment raw observation with build cost, history, and memory.

    Args:
        obs_arr: (14, H, W) from obs_to_array — single sample, no batch dim
        build_cost: (H, W) own build cost from build_cost_from_obs (or the
            state-side build_cost_grid during training)
        obs_state: AugmentedObsState — single sample

    Returns:
        augmented_obs: (25 + 2*history_size, pad_to, pad_to) float array
        new_obs_state: AugmentedObsState
    """
    # Channel indices in obs_arr (from obs_to_array)
    armies, generals_ch, castles_ch, mountains_ch = 0, 1, 2, 3
    neutral_cells, owned_cells, opponent_cells = 4, 5, 6
    fog_cells, structures_in_fog = 7, 8
    owned_land_count, owned_army_count = 9, 10
    opponent_land_count, opponent_army_count = 11, 12
    timestep_ch = 13

    # Infer pad_to from obs_state spatial dimensions
    p = obs_state.last_army.shape[0]

    # Pad obs from grid_size to pad_to x pad_to (border = mountains). In-env
    # boards arrive already mountain-padded to pad_to, so this block is a
    # no-op there; the deployment bot feeds true 18-21 boards through it.
    h, w = obs_arr.shape[1], obs_arr.shape[2]
    pad_h, pad_w = p - h, p - w
    obs = jnp.pad(obs_arr, ((0, 0), (0, pad_h), (0, pad_w)))
    cost = jnp.pad(build_cost.astype(jnp.float32), ((0, pad_h), (0, pad_w)))
    pad_mask = (jnp.arange(p)[:, None] >= h) | (jnp.arange(p)[None, :] >= w)
    # Accumulate visible padding cells as confirmed mountains in obs_state
    visible = _max_pool_2d(obs[owned_cells]) > 0
    seen_pad_mountains = obs_state.mountains | (pad_mask & visible)
    # Visible (or previously seen) padding → mountains; rest → structures_in_fog
    obs = obs.at[mountains_ch].set(jnp.where(pad_mask & seen_pad_mountains, 1.0, obs[mountains_ch]))
    obs = obs.at[structures_in_fog].set(jnp.where(pad_mask & ~seen_pad_mountains, 1.0, obs[structures_in_fog]))
    # Broadcast scalar channels into padding (training has them everywhere)
    for ch in (owned_land_count, owned_army_count, opponent_land_count,
               opponent_army_count, timestep_ch):
        obs = obs.at[ch].set(jnp.where(pad_mask, obs[ch, 0, 0], obs[ch]))

    # Calculate current army states
    current_army = obs[armies] * obs[owned_cells]
    current_enemy_army = obs[armies] * obs[opponent_cells]

    # Update history stacks
    new_army_stack = jnp.concatenate([
        (current_army - obs_state.last_army)[None, :, :],
        obs_state.army_stack[:-1, :, :]
    ], axis=0)
    new_enemy_stack = jnp.concatenate([
        (current_enemy_army - obs_state.last_enemy_army)[None, :, :],
        obs_state.enemy_stack[:-1, :, :]
    ], axis=0)

    # Max pooling for visibility
    new_seen = obs_state.seen | (_max_pool_2d(obs[owned_cells]) > 0)
    new_enemy_seen = obs_state.enemy_seen | (_max_pool_2d(obs[opponent_cells]) > 0)

    # Accumulate static structures
    new_castles = obs_state.castles | (obs[castles_ch] > 0)
    new_generals = obs_state.generals | (obs[generals_ch] > 0)
    new_mountains = obs_state.mountains | (obs[mountains_ch] > 0) | seen_pad_mountains

    # Update last seen enemy army
    new_last_enemy_army_seen_value = jnp.where(
        current_enemy_army > 0, current_enemy_army,
        obs_state.last_enemy_army_seen_value
    )
    # Store raw step count; log decay applied only in observation channel
    new_last_enemy_army_seen_timestep = jnp.where(
        current_enemy_army > 0, 0.0, obs_state.last_enemy_army_seen_timestep + 1.0
    )

    # Update temporal opponent stat histories (sliding window: roll left, write newest at end)
    opp_army_val = obs[opponent_army_count, 0, 0]  # scalar (broadcast channel)
    opp_land_val = obs[opponent_land_count, 0, 0]
    new_opponent_army_history = jnp.roll(obs_state.opponent_army_history, -1).at[-1].set(opp_army_val)
    new_opponent_land_history = jnp.roll(obs_state.opponent_land_history, -1).at[-1].set(opp_land_val)
    new_temporal_step = obs_state.temporal_step + 1

    # Coordinate channels (normalized to [0, 1])
    coords_x = jnp.broadcast_to(jnp.arange(p, dtype=jnp.float32)[None, :] / (p - 1), (p, p))
    coords_y = jnp.broadcast_to(jnp.arange(p, dtype=jnp.float32)[:, None] / (p - 1), (p, p))

    # Build augmented observation (25 base channels)
    ones = jnp.ones((p, p))
    channels = jnp.stack([
        obs[armies],                                  # 0
        current_army,                                 # 1
        current_enemy_army,                           # 2
        obs[armies] * obs[neutral_cells],             # 3
        new_seen.astype(jnp.float32),                 # 4
        new_enemy_seen.astype(jnp.float32),           # 5
        new_generals.astype(jnp.float32),             # 6
        new_castles.astype(jnp.float32),              # 7
        new_mountains.astype(jnp.float32),            # 8
        obs[neutral_cells],                           # 9
        obs[owned_cells],                             # 10
        obs[opponent_cells],                          # 11
        obs[fog_cells],                               # 12
        obs[structures_in_fog],                       # 13
        obs[timestep_ch] * ones,                      # 14
        (obs[timestep_ch] % 50) * ones / 50,          # 15
        obs[owned_land_count] * ones,                 # 16
        obs[owned_army_count] * ones,                 # 17
        obs[opponent_land_count] * ones,              # 18
        obs[opponent_army_count] * ones,              # 19
        new_last_enemy_army_seen_value,               # 20
        jnp.log1p(new_last_enemy_army_seen_timestep) / 5.0,  # 21: log decay
        coords_x,                                     # 22
        coords_y,                                     # 23
        cost / 50.0,                                  # 24: own build cost (decision 4)
    ], axis=0)

    # Concatenate history stacks
    augmented_obs = jnp.concatenate([channels, new_army_stack, new_enemy_stack], axis=0)

    new_state = AugmentedObsState(
        army_stack=new_army_stack,
        enemy_stack=new_enemy_stack,
        last_army=current_army,
        last_enemy_army=current_enemy_army,
        castles=new_castles,
        generals=new_generals,
        mountains=new_mountains,
        seen=new_seen,
        enemy_seen=new_enemy_seen,
        last_enemy_army_seen_value=new_last_enemy_army_seen_value,
        last_enemy_army_seen_timestep=new_last_enemy_army_seen_timestep,
        opponent_army_history=new_opponent_army_history,
        opponent_land_history=new_opponent_land_history,
        temporal_step=new_temporal_step,
    )
    return augmented_obs, new_state


# ---- Action encoding/decoding (10 channels) ----


def decode_action(idx, pad_to):
    """Convert flat logit index to engine action array.

    Args:
        idx: scalar index into (10 * pad_to * pad_to,) logit vector.
            Channel 0-3 full move, 4-7 half move, 8 pass, 9 build.
        pad_to: grid dimension

    Returns:
        (5,) int32 array: [pass_field, row, col, direction, is_half]
        where pass_field is 0=move, 1=pass, 2=build (build_castles modifier).
    """
    gc = pad_to * pad_to
    d, pos = idx // gc, idx % gc
    r, c = pos // pad_to, pos % pad_to
    is_pass = d == 8
    is_build = d == 9
    is_half = (d >= 4) & (d < 8)
    pf = jnp.where(is_build, 2, jnp.where(is_pass, 1, 0))
    ad = jnp.where(is_half, d - 4, jnp.where(d < 4, d, 0))
    return jnp.array([pf, r, c, ad, is_half], dtype=jnp.int32)


def encode_action(action, pad_to):
    """Convert engine action array back to flat logit index.

    Args:
        action: (5,) array [pass_field, row, col, direction, is_half]
        pad_to: grid dimension

    Returns:
        scalar int32 index into (10 * pad_to * pad_to,) logit vector
    """
    gc = pad_to * pad_to
    pf = action[0].astype(jnp.int32)
    r = action[1].astype(jnp.int32)
    c = action[2].astype(jnp.int32)
    d = action[3].astype(jnp.int32)
    ih = action[4].astype(jnp.int32)
    ed = jnp.where(pf == 2, 9,
                   jnp.where(pf == 1, 8,
                             jnp.where(ih > 0, d + 4, d)))
    return ed * gc + r * pad_to + c


# ---- Mask and normalization ----


def prepare_action_mask(direction_mask, build_mask, pad_to, allow_pass=True):
    """Pad masks and build the 10-channel action penalty.

    Args:
        direction_mask: (H, W, 4) move validity mask (1=valid, 0=invalid)
        build_mask: (H, W) build validity mask from compute_build_mask
        pad_to: target spatial dimension
        allow_pass: if False, mask out the pass action

    Returns:
        (10, pad_to, pad_to) penalty array (-1e9 for invalid, 0 for valid)
    """
    mask_t = 1 - jnp.transpose(direction_mask, (2, 0, 1))
    pad_h = pad_to - mask_t.shape[1]
    pad_w = pad_to - mask_t.shape[2]
    mask = jnp.pad(mask_t, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=1)
    full_mask = mask[:4]
    half_mask = mask[:4]
    pass_val = 0.0 if allow_pass else 1.0
    pass_mask = jnp.full((1, pad_to, pad_to), pass_val)
    build_pen = jnp.pad(
        1 - build_mask.astype(jnp.float32),
        ((0, pad_h), (0, pad_w)), constant_values=1)[None]
    return jnp.concatenate(
        [full_mask, half_mask, pass_mask, build_pen], axis=0) * -1e9


def normalize_observations(obs):
    """Normalize augmented observation channels (25 base + history).

    All divisors standardized to 50, as in the released code. Channel 24
    (build cost) is already scaled in augment_obs.

    Args:
        obs: (C, pad_to, pad_to) augmented observation

    Returns:
        normalized observation, same shape
    """
    army_channels = [0, 1, 2, 3, 17, 19, 20] + list(range(N_BASE_CHANNELS, obs.shape[0]))
    obs = obs.at[jnp.array(army_channels)].divide(50.0)
    obs = obs.at[14].divide(50.0)
    obs = obs.at[jnp.array([16, 18])].divide(50.0)
    return obs
