//! The 3970-logit action codec and the legal mask.
//!
//! Port of `bots/morpheus/action.py`. Layout is channel-major over the padded
//! 21×21 board:
//!
//! ```text
//! index = channel * 441 + row * 21 + col     channels 0..8
//! pass  = 3969
//! ```
//!
//! Channels 0–3 are full moves (up, down, left, right), 4–7 the half-army
//! split of the same, 8 a castle build. Direction order matches the wire
//! protocol.
//!
//! Note the index arithmetic uses the **padded** stride 21, not the board's
//! live width. A 18×18 board still encodes at stride 21, so the codec is
//! independent of the board it is describing — which is what lets one network
//! output cover every board size the preset generates.

use crate::memory::{VisibleMemory, TYPE_MOUNTAIN, TYPE_STRUCTURE_FOG};
use crate::state::{GameState, MAX_CELLS};
use crate::transition::{build_cost_grid, DIRECTIONS};
use crate::wire::Observation;

pub const PAD: usize = 21;
pub const N_CELLS: usize = PAD * PAD;
pub const N_CHANNELS: usize = 9;
pub const N_ACTIONS: usize = N_CHANNELS * N_CELLS + 1;
pub const PASS_INDEX: usize = N_ACTIONS - 1;
pub const CH_BUILD: usize = 8;

/// Wire `(pass, row, col, direction, split)` to a logit index.
pub fn encode_action(action: [i32; 5]) -> usize {
    let [pass_f, row, col, direction, split] = action;
    if pass_f == 1 {
        return PASS_INDEX;
    }
    let cell = row as usize * PAD + col as usize;
    if pass_f == 2 {
        return CH_BUILD * N_CELLS + cell;
    }
    let channel = direction as usize + if split == 1 { 4 } else { 0 };
    channel * N_CELLS + cell
}

/// Logit index back to a wire action. `None` outside the layout.
pub fn decode_action(index: usize) -> Option<[i32; 5]> {
    if index == PASS_INDEX {
        return Some([1, 0, 0, 0, 0]);
    }
    if index >= PASS_INDEX {
        return None;
    }
    let channel = index / N_CELLS;
    let rem = index % N_CELLS;
    let (row, col) = ((rem / PAD) as i32, (rem % PAD) as i32);
    match channel {
        CH_BUILD => Some([2, row, col, 0, 0]),
        0..=3 => Some([0, row, col, channel as i32, 0]),
        4..=7 => Some([0, row, col, channel as i32 - 4, 1]),
        _ => None,
    }
}

/// Own general plus own castles: visible owner-1 structures and latched ones.
fn own_structures(obs: &Observation, memory: &VisibleMemory) -> [bool; MAX_CELLS] {
    let mut out = memory.own_general;
    for i in 0..obs.h * obs.w {
        let own = obs.owner_grid[i] as i32 == 1;
        let t = obs.type_grid[i] as i32;
        if own && (t == 3 || t == 4 || memory.known_castle[i]) {
            out[i] = true;
        }
    }
    out
}

/// The exact live build-cost grid from the perspective player's own view.
///
/// Built by folding every own structure into a synthetic state's `castles`
/// plane and pricing that, because `build_cost_grid` charges on
/// `(castles | generals) & own` and the observer cannot see the enemy's.
pub fn live_build_cost(obs: &Observation, memory: &VisibleMemory) -> [i32; MAX_CELLS] {
    let mut synthetic = GameState::empty(obs.h, obs.w);
    let structures = own_structures(obs, memory);
    for i in 0..obs.h * obs.w {
        synthetic.armies[i] = obs.army_grid[i];
        synthetic.ownership[0][i] = obs.owner_grid[i] as i32 == 1;
        synthetic.castles[i] = structures[i];
        synthetic.mountains[i] = memory.known_mountain[i];
        synthetic.passable[i] = !memory.known_mountain[i];
    }
    synthetic.general_positions = [[0, 0], [-1, -1]];
    synthetic.time = obs.turn;
    build_cost_grid(&synthetic, 0)
}

/// Boolean mask of length 3970. Pass is always legal.
pub fn legal_mask(
    obs: &Observation,
    memory: &VisibleMemory,
    cost_grid: Option<&[i32; MAX_CELLS]>,
) -> [bool; N_ACTIONS] {
    let owned_cost;
    let cost = match cost_grid {
        Some(grid) => grid,
        None => {
            owned_cost = live_build_cost(obs, memory);
            &owned_cost
        }
    };

    let mut mask = [false; N_ACTIONS];
    mask[PASS_INDEX] = true;
    let (h, w) = (obs.h as i32, obs.w as i32);

    // Moves: an owned cell with at least two units can push all-but-one in any
    // direction that stays on the board and is not blocked. Splitting needs a
    // third unit, or the half would be the same as the whole.
    for r in 0..h {
        for c in 0..w {
            let at = (r * w + c) as usize;
            if obs.owner_grid[at] as i32 != 1 {
                continue;
            }
            let src_army = obs.army_grid[at];
            if src_army < 2 {
                continue;
            }
            let allow_half = src_army > 2;
            for (d, (dr, dc)) in DIRECTIONS.iter().enumerate() {
                let (nr, nc) = (r + dr, c + dc);
                if nr < 0 || nc < 0 || nr >= h || nc >= w {
                    continue;
                }
                let dest_t = obs.type_grid[(nr * w + nc) as usize] as i32;
                if dest_t == TYPE_MOUNTAIN || dest_t == TYPE_STRUCTURE_FOG {
                    continue;
                }
                mask[encode_action([0, r, c, d as i32, 0])] = true;
                if allow_half {
                    mask[encode_action([0, r, c, d as i32, 1])] = true;
                }
            }
        }
    }

    // Builds: an owned, affordable, provably-plain cell. "Provably" is the
    // point of `known_passable_base` — a cell that has never been seen might
    // be a castle, and a build there would be silently refused by the engine.
    for r in 0..h {
        for c in 0..w {
            let at = (r * w + c) as usize;
            if obs.owner_grid[at] as i32 != 1 {
                continue;
            }
            if memory.own_general[at] || memory.known_enemy_general[at] {
                continue;
            }
            if memory.known_castle[at] || memory.known_mountain[at] {
                continue;
            }
            if !memory.known_passable_base[at] {
                continue;
            }
            if obs.army_grid[at] < cost[at] {
                continue;
            }
            mask[encode_action([2, r, c, 0, 0])] = true;
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_codec_round_trips_every_index() {
        for index in 0..N_ACTIONS {
            let action = decode_action(index).expect("every index decodes");
            assert_eq!(encode_action(action), index, "index {index}");
        }
    }

    #[test]
    fn pass_is_the_last_index_and_the_layout_is_channel_major() {
        assert_eq!(PASS_INDEX, 3969);
        assert_eq!(encode_action([1, 0, 0, 0, 0]), 3969);
        // channel 0 (up, full) at (1,2)
        assert_eq!(encode_action([0, 1, 2, 0, 0]), 1 * PAD + 2);
        // the half-split of the same move is four channels along
        assert_eq!(encode_action([0, 1, 2, 0, 1]), 4 * N_CELLS + PAD + 2);
        assert_eq!(encode_action([2, 1, 2, 0, 0]), CH_BUILD * N_CELLS + PAD + 2);
    }

    #[test]
    fn the_codec_uses_the_padded_stride_not_the_board_width() {
        // Same cell on an 18-wide board encodes identically: the layout is a
        // property of the network output, not of the board in play.
        assert_eq!(encode_action([0, 3, 0, 1, 0]), N_CELLS + 3 * PAD);
    }

    #[test]
    fn out_of_layout_indices_do_not_decode() {
        assert!(decode_action(N_ACTIONS).is_none());
        assert!(decode_action(usize::MAX).is_none());
    }
}
