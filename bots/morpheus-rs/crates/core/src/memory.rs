//! Persistent visible memory — the planes, at M1.
//!
//! Port of the `VisibleMemory` shape in `bots/morpheus/memory.py`. The
//! *update* rule (`update_memory`) is M2 work; M1 needs only the container,
//! because the legal mask reads five of its planes and the parity harness
//! feeds them straight from the recorded corpus.
//!
//! Keeping the struct here rather than inlining the five planes into
//! `action.rs` is deliberate: M2 fills in the update against this exact shape,
//! and a mask that had quietly grown its own notion of "remembered" would be
//! the kind of divergence tier-1 parity is meant to catch early.

use crate::state::MAX_CELLS;

// Competition type codes, perspective-relative wire.
pub const TYPE_FOG: i32 = 0;
pub const TYPE_PLAIN: i32 = 1;
pub const TYPE_MOUNTAIN: i32 = 2;
pub const TYPE_CASTLE: i32 = 3;
pub const TYPE_GENERAL: i32 = 4;
pub const TYPE_STRUCTURE_FOG: i32 = 5;

pub const OWNER_NEUTRAL: i32 = 0;
pub const OWNER_ME: i32 = 1;
pub const OWNER_ENEMY: i32 = 2;

/// Persistent facts and last-seen dynamic values on the true H×W board.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VisibleMemory {
    pub h: usize,
    pub w: usize,
    pub known_mountain: [bool; MAX_CELLS],
    /// Non-mountain, non-castle terrain that has been seen.
    pub known_passable_base: [bool; MAX_CELLS],
    pub known_castle: [bool; MAX_CELLS],
    /// At most one cell.
    pub own_general: [bool; MAX_CELLS],
    /// Latched once found, or all false.
    pub known_enemy_general: [bool; MAX_CELLS],
    pub ever_visible: [bool; MAX_CELLS],
    /// `-1` before first sight.
    pub last_seen_turn: [i32; MAX_CELLS],
    /// 0/1/2 at last sight.
    pub remembered_owner: [i32; MAX_CELLS],
    pub remembered_army: [i32; MAX_CELLS],
    pub remembered_was_castle: [bool; MAX_CELLS],
    pub remembered_castle_owner: [i32; MAX_CELLS],
}

impl VisibleMemory {
    pub fn empty(h: usize, w: usize) -> Self {
        Self {
            h,
            w,
            known_mountain: [false; MAX_CELLS],
            known_passable_base: [false; MAX_CELLS],
            known_castle: [false; MAX_CELLS],
            own_general: [false; MAX_CELLS],
            known_enemy_general: [false; MAX_CELLS],
            ever_visible: [false; MAX_CELLS],
            last_seen_turn: [-1; MAX_CELLS],
            remembered_owner: [0; MAX_CELLS],
            remembered_army: [0; MAX_CELLS],
            remembered_was_castle: [false; MAX_CELLS],
            remembered_castle_owner: [0; MAX_CELLS],
        }
    }

    #[inline]
    pub fn idx(&self, row: usize, col: usize) -> usize {
        row * self.w + col
    }

    #[inline]
    pub fn cells(&self) -> usize {
        self.h * self.w
    }
}
