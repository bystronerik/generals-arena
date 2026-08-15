//! Port of the joe bot's observation pipeline (`joe_obs.py` in the Python
//! sibling — the deployment copy of `training/joe/networks/common.py`).
//!
//! Everything here is integer logic plus f32 adds, subtracts, multiplies and
//! divides-by-50 mirrored site by site in the Python's op order, so the
//! parity target is bit-exact (port-plan §4). The one transcendental is
//! `log1p` on channel 21; its agreement is measured, not assumed.
//!
//! Layout: 25 base observation channels plus 2 × 7 history channels = 39
//! total; 10 per-cell action channels — 0–3 full move, 4–7 half move,
//! 8 pass, 9 build.

use crate::io::wire::{
    Observation, OWNER_ME, OWNER_OPP, TYPE_CASTLE, TYPE_FOG, TYPE_GENERAL, TYPE_MOUNTAIN,
    TYPE_PLAIN, TYPE_STRUCTURE_IN_FOG,
};
use crate::xla_math::{log1p, RECIP_5, RECIP_50};

pub const PAD: usize = 21;
pub const CELLS: usize = PAD * PAD; // 441
pub const HISTORY: usize = 7;
pub const N_BASE_CHANNELS: usize = 25;
pub const N_CHANNELS: usize = N_BASE_CHANNELS + 2 * HISTORY; // 39
pub const N_ACTION_CHANNELS: usize = 10;
pub const TEMPORAL_WINDOW: usize = 512;
pub const N_RAW_CHANNELS: usize = 14;

// Build-cost constants (generals/modifiers/build_castles.py).
// `pub` on the base cost is the fork's one mechanical edit to this file
// (tactics-plan.md §1): `tactics::filters` reads the crowding surcharge off
// this grid as `cost - BUILD_BASE_COST`, and a second 35 in the crate would be
// one rule with two spellings. No code path changed.
pub const BUILD_BASE_COST: i32 = 35;
const BUILD_PROXIMITY_PENALTY: i32 = 14;
const BUILD_PROXIMITY_DECAY: i32 = 2;
const BUILD_RADIUS: i32 = 6;

/// Direction offsets `(dr, dc)`: UP, DOWN, LEFT, RIGHT — the engine's order,
/// which is also the wire protocol's dir codes.
pub const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

// Channel indices in the raw (14, H, W) array — the order of
// `Observation.as_tensor` / training's `obs_to_array`.
pub const CH_ARMIES: usize = 0;
pub const CH_GENERALS: usize = 1;
pub const CH_CASTLES: usize = 2;
pub const CH_MOUNTAINS: usize = 3;
pub const CH_NEUTRAL: usize = 4;
pub const CH_OWNED: usize = 5;
pub const CH_OPPONENT: usize = 6;
pub const CH_FOG: usize = 7;
pub const CH_STRUCTURES_IN_FOG: usize = 8;
pub const CH_OWNED_LAND: usize = 9;
pub const CH_OWNED_ARMY: usize = 10;
pub const CH_OPPONENT_LAND: usize = 11;
pub const CH_OPPONENT_ARMY: usize = 12;
pub const CH_TIMESTEP: usize = 13;

/// Wire frame -> the engine's `(14, H, W)` observation tensor, flat
/// channel-major. Mirrors `agent.py::frame_to_raw`.
pub fn frame_to_raw(obs: &Observation, raw: &mut Vec<f32>) {
    let n = obs.h * obs.w;
    raw.clear();
    raw.resize(N_RAW_CHANNELS * n, 0.0);
    for i in 0..n {
        let t = obs.type_grid[i];
        let o = obs.owner_grid[i];
        raw[CH_ARMIES * n + i] = obs.army_grid[i] as f32;
        raw[CH_GENERALS * n + i] = (t == TYPE_GENERAL) as i32 as f32;
        raw[CH_CASTLES * n + i] = (t == TYPE_CASTLE) as i32 as f32;
        raw[CH_MOUNTAINS * n + i] = (t == TYPE_MOUNTAIN) as i32 as f32;
        raw[CH_NEUTRAL * n + i] =
            ((o == 0) && (t == TYPE_PLAIN || t == TYPE_CASTLE)) as i32 as f32;
        raw[CH_OWNED * n + i] = (o == OWNER_ME) as i32 as f32;
        raw[CH_OPPONENT * n + i] = (o == OWNER_OPP) as i32 as f32;
        raw[CH_FOG * n + i] = (t == TYPE_FOG) as i32 as f32;
        raw[CH_STRUCTURES_IN_FOG * n + i] = (t == TYPE_STRUCTURE_IN_FOG) as i32 as f32;
        raw[CH_OWNED_LAND * n + i] = obs.my_land as f32;
        raw[CH_OWNED_ARMY * n + i] = obs.my_army as f32;
        raw[CH_OPPONENT_LAND * n + i] = obs.opp_land as f32;
        raw[CH_OPPONENT_ARMY * n + i] = obs.opp_army as f32;
        raw[CH_TIMESTEP * n + i] = obs.turn as f32;
    }
}

/// `(H, W)` own castle price per cell, from the raw array. Pure integer
/// kernel: 35 base plus `max(0, 14 - 2d)` per own structure at manhattan
/// distance `d` within radius 6. Mirrors `build_cost_from_raw`.
pub fn build_cost_from_raw(raw: &[f32], h: usize, w: usize, cost: &mut Vec<i32>) {
    let n = h * w;
    cost.clear();
    cost.resize(n, BUILD_BASE_COST);
    for si in 0..h {
        for sj in 0..w {
            let i = si * w + sj;
            let is_structure = (raw[CH_CASTLES * n + i] > 0.0
                || raw[CH_GENERALS * n + i] > 0.0)
                && raw[CH_OWNED * n + i] > 0.0;
            if !is_structure {
                continue;
            }
            for di in -BUILD_RADIUS..=BUILD_RADIUS {
                for dj in -BUILD_RADIUS..=BUILD_RADIUS {
                    let surcharge = BUILD_PROXIMITY_PENALTY
                        - BUILD_PROXIMITY_DECAY * (di.abs() + dj.abs());
                    if surcharge <= 0 {
                        continue;
                    }
                    let (ti, tj) = (si as i32 + di, sj as i32 + dj);
                    if ti >= 0 && ti < h as i32 && tj >= 0 && tj < w as i32 {
                        cost[ti as usize * w + tj as usize] += surcharge;
                    }
                }
            }
        }
    }
}

/// `(H, W, 4)` move validity, flat as `[cell * 4 + dir]`. Mirrors the
/// engine's `compute_valid_move_mask`: own cell with more than one army,
/// destination in bounds and not a mountain.
pub fn compute_valid_move_mask(raw: &[f32], h: usize, w: usize, mask: &mut Vec<bool>) {
    let n = h * w;
    mask.clear();
    mask.resize(n * 4, false);
    for i in 0..h {
        for j in 0..w {
            let idx = i * w + j;
            let can_move_from =
                raw[CH_OWNED * n + idx] > 0.0 && raw[CH_ARMIES * n + idx] > 1.0;
            if !can_move_from {
                continue;
            }
            for (d, (dr, dc)) in DIRECTIONS.iter().enumerate() {
                let (ti, tj) = (i as i32 + dr, j as i32 + dc);
                if ti < 0 || ti >= h as i32 || tj < 0 || tj >= w as i32 {
                    continue;
                }
                let t_idx = ti as usize * w + tj as usize;
                if raw[CH_MOUNTAINS * n + t_idx] == 0.0 {
                    mask[idx * 4 + d] = true;
                }
            }
        }
    }
}

/// `(H, W)` build validity: own plain cell (no general/castle) with
/// `armies >= cost`. Mirrors `compute_build_mask_from_raw`.
pub fn compute_build_mask_from_raw(
    raw: &[f32],
    h: usize,
    w: usize,
    cost: &[i32],
    mask: &mut Vec<bool>,
) {
    let n = h * w;
    mask.clear();
    mask.resize(n, false);
    for i in 0..n {
        let plain = raw[CH_GENERALS * n + i] <= 0.0 && raw[CH_CASTLES * n + i] <= 0.0;
        mask[i] = raw[CH_OWNED * n + i] > 0.0 && plain && raw[CH_ARMIES * n + i] >= cost[i] as f32;
    }
}

/// The persistent augmentation state — `AugmentedObsState` as plain arrays,
/// carried across turns for up to 1,200 of them.
#[derive(Debug, Clone)]
pub struct AugState {
    pub army_stack: Vec<f32>,                    // (HISTORY, PAD, PAD)
    pub enemy_stack: Vec<f32>,                   // (HISTORY, PAD, PAD)
    pub last_army: Vec<f32>,                     // (PAD, PAD)
    pub last_enemy_army: Vec<f32>,               // (PAD, PAD)
    pub castles: Vec<bool>,                      // (PAD, PAD)
    pub generals: Vec<bool>,                     // (PAD, PAD)
    pub mountains: Vec<bool>,                    // (PAD, PAD)
    pub seen: Vec<bool>,                         // (PAD, PAD)
    pub enemy_seen: Vec<bool>,                   // (PAD, PAD)
    pub last_enemy_army_seen_value: Vec<f32>,    // (PAD, PAD)
    pub last_enemy_army_seen_timestep: Vec<f32>, // (PAD, PAD)
    pub opponent_army_history: Vec<f32>,         // (TEMPORAL_WINDOW,)
    pub opponent_land_history: Vec<f32>,         // (TEMPORAL_WINDOW,)
    pub temporal_step: i32,
}

impl AugState {
    pub fn zeros() -> Self {
        Self {
            army_stack: vec![0.0; HISTORY * CELLS],
            enemy_stack: vec![0.0; HISTORY * CELLS],
            last_army: vec![0.0; CELLS],
            last_enemy_army: vec![0.0; CELLS],
            castles: vec![false; CELLS],
            generals: vec![false; CELLS],
            mountains: vec![false; CELLS],
            seen: vec![false; CELLS],
            enemy_seen: vec![false; CELLS],
            last_enemy_army_seen_value: vec![0.0; CELLS],
            last_enemy_army_seen_timestep: vec![0.0; CELLS],
            opponent_army_history: vec![0.0; TEMPORAL_WINDOW],
            opponent_land_history: vec![0.0; TEMPORAL_WINDOW],
            temporal_step: 0,
        }
    }
}

/// 3×3 max pool with SAME padding over one `(PAD, PAD)` plane — the
/// visibility kernel (`_max_pool_2d`). Inputs are 0/1, so `> 0.0` afterwards
/// is exact.
fn max_pool_3x3_positive(plane: &[f32], out: &mut [bool]) {
    for i in 0..PAD {
        for j in 0..PAD {
            let mut any = false;
            'window: for di in -1i32..=1 {
                for dj in -1i32..=1 {
                    let (ti, tj) = (i as i32 + di, j as i32 + dj);
                    if ti < 0 || ti >= PAD as i32 || tj < 0 || tj >= PAD as i32 {
                        continue;
                    }
                    if plane[ti as usize * PAD + tj as usize] > 0.0 {
                        any = true;
                        break 'window;
                    }
                }
            }
            out[i * PAD + j] = any;
        }
    }
}

/// Scratch buffers for `augment_obs`, allocated once at startup so the
/// per-turn path never grows the heap.
pub struct AugScratch {
    padded: Vec<f32>,   // (14, PAD, PAD)
    cost_f32: Vec<f32>, // (PAD, PAD)
    visible: Vec<bool>, // (PAD, PAD) — own visibility (3×3 pool of owned)
    enemy_visible: Vec<bool>,
    seen_pad_mountains: Vec<bool>,
    current_army: Vec<f32>,
    current_enemy_army: Vec<f32>,
}

impl AugScratch {
    pub fn new() -> Self {
        Self {
            padded: vec![0.0; N_RAW_CHANNELS * CELLS],
            cost_f32: vec![0.0; CELLS],
            visible: vec![false; CELLS],
            enemy_visible: vec![false; CELLS],
            seen_pad_mountains: vec![false; CELLS],
            current_army: vec![0.0; CELLS],
            current_enemy_army: vec![0.0; CELLS],
        }
    }
}

/// Mirror of `augment_obs`: pad to 21×21, apply the seen-pad-mountain rule,
/// update the persistent state, and emit the `(39, 21, 21)` augmented tensor
/// (unnormalized — `normalize_observations` is a separate step, as in
/// Python).
///
/// Reads `state` (the previous turn's state) and fully overwrites `next`.
#[allow(clippy::too_many_arguments)]
pub fn augment_obs(
    raw: &[f32],
    h: usize,
    w: usize,
    cost: &[i32],
    state: &AugState,
    next: &mut AugState,
    scratch: &mut AugScratch,
    aug: &mut [f32], // (N_CHANNELS, PAD, PAD)
) {
    assert!(h <= PAD && w <= PAD);
    let n = h * w;

    // Pad obs from (h, w) to PAD×PAD with zeros (channel content for the
    // border is decided below), and the cost grid alongside.
    let padded = &mut scratch.padded;
    padded.iter_mut().for_each(|v| *v = 0.0);
    for c in 0..N_RAW_CHANNELS {
        for i in 0..h {
            for j in 0..w {
                padded[c * CELLS + i * PAD + j] = raw[c * n + i * w + j];
            }
        }
    }
    let cost_f32 = &mut scratch.cost_f32;
    cost_f32.iter_mut().for_each(|v| *v = 0.0);
    for i in 0..h {
        for j in 0..w {
            cost_f32[i * PAD + j] = cost[i * w + j] as f32;
        }
    }

    let pad_mask = |i: usize, j: usize| i >= h || j >= w;

    // Own visibility (3×3 pool of the owned channel), then the pad rule:
    // visible padding accumulates as confirmed mountains; the rest of the
    // padding reads as structures-in-fog.
    max_pool_3x3_positive(&padded[CH_OWNED * CELLS..(CH_OWNED + 1) * CELLS], &mut scratch.visible);
    max_pool_3x3_positive(
        &padded[CH_OPPONENT * CELLS..(CH_OPPONENT + 1) * CELLS],
        &mut scratch.enemy_visible,
    );
    for i in 0..PAD {
        for j in 0..PAD {
            let idx = i * PAD + j;
            scratch.seen_pad_mountains[idx] =
                state.mountains[idx] || (pad_mask(i, j) && scratch.visible[idx]);
            if pad_mask(i, j) {
                if scratch.seen_pad_mountains[idx] {
                    padded[CH_MOUNTAINS * CELLS + idx] = 1.0;
                } else {
                    padded[CH_STRUCTURES_IN_FOG * CELLS + idx] = 1.0;
                }
            }
        }
    }

    // Broadcast scalar channels into padding (training has them everywhere).
    for ch in [CH_OWNED_LAND, CH_OWNED_ARMY, CH_OPPONENT_LAND, CH_OPPONENT_ARMY, CH_TIMESTEP] {
        let v = padded[ch * CELLS]; // (0, 0) is always in-board
        for i in 0..PAD {
            for j in 0..PAD {
                if pad_mask(i, j) {
                    padded[ch * CELLS + i * PAD + j] = v;
                }
            }
        }
    }

    // Current army planes and the delta history stacks.
    for idx in 0..CELLS {
        scratch.current_army[idx] = padded[CH_ARMIES * CELLS + idx] * padded[CH_OWNED * CELLS + idx];
        scratch.current_enemy_army[idx] =
            padded[CH_ARMIES * CELLS + idx] * padded[CH_OPPONENT * CELLS + idx];
    }
    for idx in 0..CELLS {
        next.army_stack[idx] = scratch.current_army[idx] - state.last_army[idx];
        next.enemy_stack[idx] = scratch.current_enemy_army[idx] - state.last_enemy_army[idx];
    }
    next.army_stack[CELLS..HISTORY * CELLS]
        .copy_from_slice(&state.army_stack[..(HISTORY - 1) * CELLS]);
    next.enemy_stack[CELLS..HISTORY * CELLS]
        .copy_from_slice(&state.enemy_stack[..(HISTORY - 1) * CELLS]);

    // Visibility memory, static structures, and last-seen enemy army.
    for idx in 0..CELLS {
        next.seen[idx] = state.seen[idx] || scratch.visible[idx];
        next.enemy_seen[idx] = state.enemy_seen[idx] || scratch.enemy_visible[idx];
        next.castles[idx] = state.castles[idx] || padded[CH_CASTLES * CELLS + idx] > 0.0;
        next.generals[idx] = state.generals[idx] || padded[CH_GENERALS * CELLS + idx] > 0.0;
        next.mountains[idx] = state.mountains[idx]
            || padded[CH_MOUNTAINS * CELLS + idx] > 0.0
            || scratch.seen_pad_mountains[idx];
        if scratch.current_enemy_army[idx] > 0.0 {
            next.last_enemy_army_seen_value[idx] = scratch.current_enemy_army[idx];
            next.last_enemy_army_seen_timestep[idx] = 0.0;
        } else {
            next.last_enemy_army_seen_value[idx] = state.last_enemy_army_seen_value[idx];
            next.last_enemy_army_seen_timestep[idx] =
                state.last_enemy_army_seen_timestep[idx] + 1.0;
        }
        next.last_army[idx] = scratch.current_army[idx];
        next.last_enemy_army[idx] = scratch.current_enemy_army[idx];
    }

    // Temporal opponent-stat windows: roll left, newest at the end.
    let opp_army_val = padded[CH_OPPONENT_ARMY * CELLS];
    let opp_land_val = padded[CH_OPPONENT_LAND * CELLS];
    next.opponent_army_history[..TEMPORAL_WINDOW - 1]
        .copy_from_slice(&state.opponent_army_history[1..]);
    next.opponent_army_history[TEMPORAL_WINDOW - 1] = opp_army_val;
    next.opponent_land_history[..TEMPORAL_WINDOW - 1]
        .copy_from_slice(&state.opponent_land_history[1..]);
    next.opponent_land_history[TEMPORAL_WINDOW - 1] = opp_land_val;
    next.temporal_step = state.temporal_step + 1;

    // The 25 base channels, in the Python's stack order.
    for idx in 0..CELLS {
        aug[idx] = padded[CH_ARMIES * CELLS + idx];
        aug[CELLS + idx] = scratch.current_army[idx];
        aug[2 * CELLS + idx] = scratch.current_enemy_army[idx];
        aug[3 * CELLS + idx] =
            padded[CH_ARMIES * CELLS + idx] * padded[CH_NEUTRAL * CELLS + idx];
        aug[4 * CELLS + idx] = next.seen[idx] as i32 as f32;
        aug[5 * CELLS + idx] = next.enemy_seen[idx] as i32 as f32;
        aug[6 * CELLS + idx] = next.generals[idx] as i32 as f32;
        aug[7 * CELLS + idx] = next.castles[idx] as i32 as f32;
        aug[8 * CELLS + idx] = next.mountains[idx] as i32 as f32;
        aug[9 * CELLS + idx] = padded[CH_NEUTRAL * CELLS + idx];
        aug[10 * CELLS + idx] = padded[CH_OWNED * CELLS + idx];
        aug[11 * CELLS + idx] = padded[CH_OPPONENT * CELLS + idx];
        aug[12 * CELLS + idx] = padded[CH_FOG * CELLS + idx];
        aug[13 * CELLS + idx] = padded[CH_STRUCTURES_IN_FOG * CELLS + idx];
        let t = padded[CH_TIMESTEP * CELLS + idx];
        aug[14 * CELLS + idx] = t;
        aug[15 * CELLS + idx] = (t % 50.0) * RECIP_50;
        aug[16 * CELLS + idx] = padded[CH_OWNED_LAND * CELLS + idx];
        aug[17 * CELLS + idx] = padded[CH_OWNED_ARMY * CELLS + idx];
        aug[18 * CELLS + idx] = padded[CH_OPPONENT_LAND * CELLS + idx];
        aug[19 * CELLS + idx] = padded[CH_OPPONENT_ARMY * CELLS + idx];
        aug[20 * CELLS + idx] = next.last_enemy_army_seen_value[idx];
        aug[21 * CELLS + idx] = log1p(next.last_enemy_army_seen_timestep[idx]) * RECIP_5;
        aug[24 * CELLS + idx] = cost_f32[idx] * RECIP_50;
    }
    // Coordinate channels (normalized to [0, 1]).
    const RECIP_PAD_MINUS_1: f32 = 1.0 / (PAD - 1) as f32;
    for i in 0..PAD {
        for j in 0..PAD {
            aug[22 * CELLS + i * PAD + j] = j as f32 * RECIP_PAD_MINUS_1;
            aug[23 * CELLS + i * PAD + j] = i as f32 * RECIP_PAD_MINUS_1;
        }
    }
    // History stacks: channels 25..32 army deltas, 32..39 enemy deltas.
    aug[N_BASE_CHANNELS * CELLS..(N_BASE_CHANNELS + HISTORY) * CELLS]
        .copy_from_slice(&next.army_stack);
    aug[(N_BASE_CHANNELS + HISTORY) * CELLS..N_CHANNELS * CELLS]
        .copy_from_slice(&next.enemy_stack);
}

/// Mirror of `normalize_observations`: divide the army-valued channels, the
/// timestep channel, and the land-count channels by 50. Channel 24 (build
/// cost) is already scaled in `augment_obs`.
pub fn normalize_observations(aug: &mut [f32]) {
    let mut divide = |c: usize| {
        for v in &mut aug[c * CELLS..(c + 1) * CELLS] {
            *v *= RECIP_50;
        }
    };
    for c in [0, 1, 2, 3, 17, 19, 20] {
        divide(c);
    }
    for c in N_BASE_CHANNELS..N_CHANNELS {
        divide(c);
    }
    divide(14);
    divide(16);
    divide(18);
}

/// Mirror of `prepare_action_mask`: the `(10, PAD, PAD)` penalty plane —
/// `-1e9` for invalid, `0` for valid. Channels 0–3 full move, 4–7 half move
/// (same validity), 8 pass (always allowed), 9 build. Cells outside the true
/// board are invalid everywhere except the pass channel.
pub fn prepare_action_mask(
    move_mask: &[bool], // (h, w, 4) flat [cell * 4 + dir]
    build_mask: &[bool],
    h: usize,
    w: usize,
    penalties: &mut [f32], // (N_ACTION_CHANNELS, PAD, PAD)
) {
    penalties.iter_mut().for_each(|v| *v = -1e9);
    for i in 0..h {
        for j in 0..w {
            let cell = i * w + j;
            let out = i * PAD + j;
            for d in 0..4 {
                if move_mask[cell * 4 + d] {
                    penalties[d * CELLS + out] = 0.0;
                    penalties[(d + 4) * CELLS + out] = 0.0;
                }
            }
            if build_mask[cell] {
                penalties[9 * CELLS + out] = 0.0;
            }
        }
    }
    for v in &mut penalties[8 * CELLS..9 * CELLS] {
        *v = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_cost_single_structure() {
        // One own general at (2, 2) on a 5×5 board: cost at distance d is
        // 35 + max(0, 14 - 2d).
        let (h, w) = (5, 5);
        let n = h * w;
        let mut raw = vec![0.0f32; N_RAW_CHANNELS * n];
        raw[CH_GENERALS * n + 2 * w + 2] = 1.0;
        raw[CH_OWNED * n + 2 * w + 2] = 1.0;
        let mut cost = Vec::new();
        build_cost_from_raw(&raw, h, w, &mut cost);
        assert_eq!(cost[2 * w + 2], 35 + 14);
        assert_eq!(cost[2 * w + 3], 35 + 12);
        assert_eq!(cost[0], 35 + 14 - 2 * 4); // distance 4
    }

    #[test]
    fn move_mask_edges_and_mountains() {
        let (h, w) = (2, 2);
        let n = h * w;
        let mut raw = vec![0.0f32; N_RAW_CHANNELS * n];
        raw[CH_OWNED * n] = 1.0; // (0,0) mine
        raw[CH_ARMIES * n] = 5.0;
        raw[CH_MOUNTAINS * n + 1] = 1.0; // (0,1) mountain
        let mut mask = Vec::new();
        compute_valid_move_mask(&raw, h, w, &mut mask);
        // From (0,0): up out of bounds, down ok, left out of bounds,
        // right blocked by mountain.
        assert_eq!(&mask[0..4], &[false, true, false, false]);
        // (1,0) not owned: nothing valid.
        assert_eq!(&mask[2 * 4..3 * 4], &[false; 4]);
    }
}
