//! The eight dihedral symmetries of the padded 21×21 board.
//!
//! Port of `bots/morpheus/symmetry.py`. This is a **training-side** concept —
//! nothing in a match applies a symmetry — and it is ported anyway because the
//! tensor path is checked through it: a transform is a cheap, total test that
//! coordinates, direction channels, generals and policy logits all remap
//! *together*. A tensor whose row-coordinate plane disagreed with its board
//! mask would look fine cell by cell and break the moment it was rotated.
//!
//! Coordinates transform on the padded square, not the live board. The pass
//! logit never moves: it has no position to move to.

use crate::board::action::{decode_action, encode_action, N_ACTIONS, PAD, PASS_INDEX};
use crate::board::memory::VisibleMemory;

/// 0 up, 1 down, 2 left, 3 right.
const DIR_UP: usize = 0;
const DIR_DOWN: usize = 1;
const DIR_LEFT: usize = 2;
const DIR_RIGHT: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Symmetry {
    Id,
    Rot90,
    Rot180,
    Rot270,
    FlipH,
    FlipV,
    Rot90FlipH,
    Rot270FlipH,
}

pub const ALL: [Symmetry; 8] = [
    Symmetry::Id,
    Symmetry::Rot90,
    Symmetry::Rot180,
    Symmetry::Rot270,
    Symmetry::FlipH,
    Symmetry::FlipV,
    Symmetry::Rot90FlipH,
    Symmetry::Rot270FlipH,
];

fn rot90_rc(r: usize, c: usize) -> (usize, usize) {
    (c, PAD - 1 - r)
}
fn rot180_rc(r: usize, c: usize) -> (usize, usize) {
    (PAD - 1 - r, PAD - 1 - c)
}
fn rot270_rc(r: usize, c: usize) -> (usize, usize) {
    (PAD - 1 - c, r)
}
fn flip_h_rc(r: usize, c: usize) -> (usize, usize) {
    (r, PAD - 1 - c)
}
fn flip_v_rc(r: usize, c: usize) -> (usize, usize) {
    (PAD - 1 - r, c)
}

impl Symmetry {
    pub fn name(self) -> &'static str {
        match self {
            Symmetry::Id => "id",
            Symmetry::Rot90 => "rot90",
            Symmetry::Rot180 => "rot180",
            Symmetry::Rot270 => "rot270",
            Symmetry::FlipH => "flip_h",
            Symmetry::FlipV => "flip_v",
            Symmetry::Rot90FlipH => "rot90_flip_h",
            Symmetry::Rot270FlipH => "rot270_flip_h",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        ALL.iter().copied().find(|s| s.name() == name)
    }

    /// The symmetry that undoes this one. Every reflection is its own inverse.
    pub fn inverse(self) -> Self {
        match self {
            Symmetry::Rot90 => Symmetry::Rot270,
            Symmetry::Rot270 => Symmetry::Rot90,
            other => other,
        }
    }

    pub fn transform_rc(self, r: usize, c: usize) -> (usize, usize) {
        match self {
            Symmetry::Id => (r, c),
            Symmetry::Rot90 => rot90_rc(r, c),
            Symmetry::Rot180 => rot180_rc(r, c),
            Symmetry::Rot270 => rot270_rc(r, c),
            Symmetry::FlipH => flip_h_rc(r, c),
            Symmetry::FlipV => flip_v_rc(r, c),
            // Composed as `flip_h(rot(rc))`, matching the Python's argument
            // order — the other order is a different group element.
            Symmetry::Rot90FlipH => {
                let (r2, c2) = rot90_rc(r, c);
                flip_h_rc(r2, c2)
            }
            Symmetry::Rot270FlipH => {
                let (r2, c2) = rot270_rc(r, c);
                flip_h_rc(r2, c2)
            }
        }
    }

    pub fn transform_dir(self, d: usize) -> usize {
        let rot90 = [DIR_RIGHT, DIR_LEFT, DIR_UP, DIR_DOWN];
        let rot180 = [DIR_DOWN, DIR_UP, DIR_RIGHT, DIR_LEFT];
        let rot270 = [DIR_LEFT, DIR_RIGHT, DIR_DOWN, DIR_UP];
        let flip_h = [DIR_UP, DIR_DOWN, DIR_RIGHT, DIR_LEFT];
        let flip_v = [DIR_DOWN, DIR_UP, DIR_LEFT, DIR_RIGHT];
        match self {
            Symmetry::Id => d,
            Symmetry::Rot90 => rot90[d],
            Symmetry::Rot180 => rot180[d],
            Symmetry::Rot270 => rot270[d],
            Symmetry::FlipH => flip_h[d],
            Symmetry::FlipV => flip_v[d],
            Symmetry::Rot90FlipH => flip_h[rot90[d]],
            Symmetry::Rot270FlipH => flip_h[rot270[d]],
        }
    }
}

/// Apply a spatial transform to one padded plane.
pub fn transform_plane(plane: &[f32; PAD * PAD], sym: Symmetry) -> [f32; PAD * PAD] {
    let mut out = [0.0f32; PAD * PAD];
    for r in 0..PAD {
        for c in 0..PAD {
            let (nr, nc) = sym.transform_rc(r, c);
            out[nr * PAD + nc] = plane[r * PAD + c];
        }
    }
    out
}

/// Remap a wire action. Pass has no position and does not move.
pub fn transform_action(action: [i32; 5], sym: Symmetry) -> [i32; 5] {
    let [pass_f, row, col, direction, split] = action;
    if pass_f == 1 {
        return [1, 0, 0, 0, 0];
    }
    let (nr, nc) = sym.transform_rc(row as usize, col as usize);
    if pass_f == 2 {
        return [2, nr as i32, nc as i32, 0, 0];
    }
    [
        0,
        nr as i32,
        nc as i32,
        sym.transform_dir(direction as usize) as i32,
        split,
    ]
}

/// Remap a length-3970 policy or logit vector.
pub fn transform_policy(logits: &[f32; N_ACTIONS], sym: Symmetry) -> [f32; N_ACTIONS] {
    let mut out = [0.0f32; N_ACTIONS];
    out[PASS_INDEX] = logits[PASS_INDEX];
    for index in 0..PASS_INDEX {
        if let Some(action) = decode_action(index) {
            out[encode_action(transform_action(action, sym))] = logits[index];
        }
    }
    out
}

/// Remap every spatial memory plane on the padded board, then crop to H×W.
///
/// Only meaningful when the content stays inside the live board after the
/// transform — on a non-square board most elements move it outside. The
/// Python carries the same caveat; this exists for square boards and tests.
pub fn transform_memory(memory: &VisibleMemory, sym: Symmetry) -> VisibleMemory {
    let (h, w) = (memory.h, memory.w);
    let mut out = VisibleMemory::empty(h, w);

    for r in 0..h {
        for c in 0..w {
            let (nr, nc) = sym.transform_rc(r, c);
            if nr >= h || nc >= w {
                continue; // moved outside the live board; cropped away
            }
            let (from, to) = (r * w + c, nr * w + nc);
            out.known_mountain[to] = memory.known_mountain[from];
            out.known_passable_base[to] = memory.known_passable_base[from];
            out.known_castle[to] = memory.known_castle[from];
            out.own_general[to] = memory.own_general[from];
            out.known_enemy_general[to] = memory.known_enemy_general[from];
            out.ever_visible[to] = memory.ever_visible[from];
            out.last_seen_turn[to] = memory.last_seen_turn[from];
            out.remembered_owner[to] = memory.remembered_owner[from];
            out.remembered_army[to] = memory.remembered_army[from];
            out.remembered_was_castle[to] = memory.remembered_was_castle[from];
            out.remembered_castle_owner[to] = memory.remembered_castle_owner[from];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_group_is_closed_under_inversion() {
        for sym in ALL {
            for r in 0..PAD {
                for c in 0..PAD {
                    let (nr, nc) = sym.transform_rc(r, c);
                    assert_eq!(
                        sym.inverse().transform_rc(nr, nc),
                        (r, c),
                        "{} did not undo",
                        sym.name()
                    );
                }
            }
        }
    }

    #[test]
    fn directions_invert_too() {
        for sym in ALL {
            for d in 0..4 {
                assert_eq!(sym.inverse().transform_dir(sym.transform_dir(d)), d);
            }
        }
    }

    #[test]
    fn every_element_is_distinct() {
        // Eight names for eight group elements: if two coincided, half the
        // training augmentation would be duplicate data.
        let fingerprints: Vec<Vec<(usize, usize)>> = ALL
            .iter()
            .map(|s| {
                (0..PAD)
                    .flat_map(|r| (0..PAD).map(move |c| (r, c)))
                    .map(|(r, c)| s.transform_rc(r, c))
                    .collect()
            })
            .collect();
        for i in 0..8 {
            for j in i + 1..8 {
                assert_ne!(
                    fingerprints[i], fingerprints[j],
                    "{} and {} are the same map",
                    ALL[i].name(),
                    ALL[j].name()
                );
            }
        }
    }

    #[test]
    fn a_moves_direction_follows_its_cell() {
        // The point of transforming directions with coordinates: a move's
        // destination must land where the rotated source's neighbour is.
        let dirs = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)];
        for sym in ALL {
            for (d, (dr, dc)) in dirs.iter().enumerate() {
                let (r, c) = (10usize, 7usize);
                let (dest_r, dest_c) = ((r as i32 + dr) as usize, (c as i32 + dc) as usize);
                let (nr, nc) = sym.transform_rc(r, c);
                let (ndr, ndc) = sym.transform_rc(dest_r, dest_c);
                let nd = sym.transform_dir(d);
                let (mr, mc) = dirs[nd];
                assert_eq!(
                    ((nr as i32 + mr) as usize, (nc as i32 + mc) as usize),
                    (ndr, ndc),
                    "{} broke the move geometry",
                    sym.name()
                );
            }
        }
    }

    #[test]
    fn a_plane_round_trips_through_a_symmetry_and_its_inverse() {
        let mut plane = [0.0f32; PAD * PAD];
        for (i, slot) in plane.iter_mut().enumerate() {
            *slot = i as f32;
        }
        for sym in ALL {
            let there = transform_plane(&plane, sym);
            let back = transform_plane(&there, sym.inverse());
            assert_eq!(back, plane, "{} lost data", sym.name());
        }
    }

    #[test]
    fn the_pass_logit_never_moves() {
        let mut logits = [0.0f32; N_ACTIONS];
        logits[PASS_INDEX] = 42.0;
        for sym in ALL {
            let out = transform_policy(&logits, sym);
            assert_eq!(out[PASS_INDEX], 42.0, "{}", sym.name());
        }
    }

    #[test]
    fn policy_remapping_is_a_permutation() {
        let mut logits = [0.0f32; N_ACTIONS];
        for (i, slot) in logits.iter_mut().enumerate() {
            *slot = i as f32 + 1.0;
        }
        for sym in ALL {
            let out = transform_policy(&logits, sym);
            let mut sorted = out.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut want = logits.to_vec();
            want.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(sorted, want, "{} dropped or duplicated a logit", sym.name());
        }
    }
}
