//! Persistent visible memory: what a seat knows after every frame it has seen.
//!
//! Port of `bots/morpheus/memory.py`. Static terrain and latched generals
//! never expire; dynamic owner and army memory keep the last seen value and
//! its age.
//!
//! The interesting part is terrain inference *through fog*. The wire gives
//! type 5 ("structure in fog") for both mountains and castles, so the update
//! has to decide which — and it decides from history: a cell already known to
//! be passable that turns into a structure must be a **new castle**, because
//! competition maps start with no castles and mountains never appear. A cell
//! that has never been passable is a **mountain**. Get that backwards and the
//! bot builds its map of the board wrong in a way no single frame reveals.

use crate::state::MAX_CELLS;
use crate::io::wire::Observation;

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

/// Fold one observation into persistent memory.
///
/// Visible cells replace dynamic memory outright. Invisible cells still teach
/// terrain: type 0 proves there is no structure, type 5 proves there is one.
///
/// Order matters throughout — the Python applies these as successive masked
/// writes, and later rules overwrite earlier ones on the same cell. Fog
/// inference runs first, visible terrain overwrites it, and generals overwrite
/// that. Reordering any pair changes the answer on cells the rules disagree
/// about.
pub fn update_memory(memory: &VisibleMemory, obs: &Observation) -> VisibleMemory {
    debug_assert_eq!(memory.h, obs.h);
    debug_assert_eq!(memory.w, obs.w);

    let mut next = memory.clone();
    let n = memory.cells();
    let turn = obs.turn;

    for i in 0..n {
        let t = obs.type_grid[i] as i32;
        let owner = obs.owner_grid[i] as i32;
        let visible = t != TYPE_FOG && t != TYPE_STRUCTURE_FOG;

        match t {
            // Type 0 proves the cell carries no mountain and no castle.
            // `known_castle` is deliberately *not* cleared: the encoder emits
            // type 5 for a castle in fog, so a castle can never appear as
            // type 0, and clearing here would only erase a true fact.
            TYPE_FOG => {
                next.known_passable_base[i] = true;
                next.known_mountain[i] = false;
            }
            TYPE_STRUCTURE_FOG => {
                // Was this ground ever known to be walkable? If so the
                // structure is new and must be a castle; if not, a mountain
                // that has simply never been in sight.
                let was_passable = memory.known_passable_base[i]
                    || memory.known_castle[i]
                    || memory.ever_visible[i];
                next.known_castle[i] = was_passable;
                next.known_mountain[i] = !was_passable;
                next.known_passable_base[i] = false;
            }
            TYPE_MOUNTAIN => {
                next.known_mountain[i] = true;
                next.known_passable_base[i] = false;
                next.known_castle[i] = false;
            }
            TYPE_PLAIN => {
                next.known_passable_base[i] = true;
                next.known_mountain[i] = false;
                next.known_castle[i] = false;
            }
            TYPE_CASTLE => {
                next.known_castle[i] = true;
                next.known_mountain[i] = false;
                next.known_passable_base[i] = false;
            }
            _ => {}
        }

        // Generals latch for both seats and clear every terrain flag on that
        // cell — a general is none of plain, castle or mountain.
        if t == TYPE_GENERAL && (owner == OWNER_ME || owner == OWNER_ENEMY) {
            if owner == OWNER_ME {
                next.own_general[i] = true;
            } else {
                next.known_enemy_general[i] = true;
            }
            next.known_mountain[i] = false;
            next.known_passable_base[i] = false;
            next.known_castle[i] = false;
        }

        if visible {
            next.ever_visible[i] = true;
            next.last_seen_turn[i] = turn;
            next.remembered_owner[i] = owner;
            next.remembered_army[i] = obs.army_grid[i];
            next.remembered_was_castle[i] = t == TYPE_CASTLE;
            // Castle *ownership* memory only refreshes while the cell is
            // visibly a castle, so a remembered castle keeps its last known
            // owner rather than inheriting whoever walked past later.
            if t == TYPE_CASTLE {
                next.remembered_castle_owner[i] = owner;
            }
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(types: &[i32], owners: &[i32], armies: &[i32], turn: i32) -> Observation {
        let mut obs = Observation::with_dims(1, types.len());
        obs.turn = turn;
        for i in 0..types.len() {
            obs.type_grid[i] = types[i] as u8;
            obs.owner_grid[i] = owners[i] as u8;
            obs.army_grid[i] = armies[i];
        }
        obs
    }

    #[test]
    fn a_structure_on_never_seen_ground_is_a_mountain() {
        let memory = VisibleMemory::empty(1, 1);
        let next = update_memory(&memory, &observation(&[TYPE_STRUCTURE_FOG], &[0], &[0], 1));
        assert!(next.known_mountain[0]);
        assert!(!next.known_castle[0]);
    }

    #[test]
    fn a_structure_on_ground_known_passable_is_a_new_castle() {
        // The competition map starts with no castles, so a structure appearing
        // where the bot has already walked can only be one a player built.
        let memory = VisibleMemory::empty(1, 1);
        let seen = update_memory(&memory, &observation(&[TYPE_PLAIN], &[1], &[3], 1));
        assert!(seen.known_passable_base[0]);
        let built = update_memory(&seen, &observation(&[TYPE_STRUCTURE_FOG], &[0], &[0], 9));
        assert!(built.known_castle[0]);
        assert!(!built.known_mountain[0]);
        assert!(!built.known_passable_base[0]);
    }

    #[test]
    fn plain_fog_proves_the_ground_is_walkable() {
        let memory = VisibleMemory::empty(1, 1);
        let next = update_memory(&memory, &observation(&[TYPE_FOG], &[0], &[0], 1));
        assert!(next.known_passable_base[0]);
        assert!(!next.known_mountain[0]);
        assert!(!next.ever_visible[0], "fog is not sight");
    }

    #[test]
    fn a_known_castle_survives_falling_into_plain_fog() {
        // A castle in fog is encoded as type 5, never type 0, so a type 0
        // frame carries no evidence against a remembered castle.
        let memory = VisibleMemory::empty(1, 1);
        let seen = update_memory(&memory, &observation(&[TYPE_CASTLE], &[2], &[40], 5));
        assert!(seen.known_castle[0]);
        let later = update_memory(&seen, &observation(&[TYPE_FOG], &[0], &[0], 20));
        assert!(later.known_castle[0], "a true fact was erased");
    }

    #[test]
    fn generals_latch_and_clear_terrain() {
        let memory = VisibleMemory::empty(1, 2);
        let next = update_memory(
            &memory,
            &observation(&[TYPE_GENERAL, TYPE_GENERAL], &[1, 2], &[1, 1], 3),
        );
        assert!(next.own_general[0] && !next.known_enemy_general[0]);
        assert!(next.known_enemy_general[1] && !next.own_general[1]);
        for i in 0..2 {
            assert!(!next.known_castle[i] && !next.known_mountain[i]);
            assert!(!next.known_passable_base[i]);
        }
        // Latched: the general stays remembered once it drops back into fog.
        let later = update_memory(&next, &observation(&[TYPE_FOG, TYPE_FOG], &[0, 0], &[0, 0], 9));
        assert!(later.own_general[0] && later.known_enemy_general[1]);
    }

    #[test]
    fn dynamic_memory_only_refreshes_on_visible_cells() {
        let memory = VisibleMemory::empty(1, 1);
        let seen = update_memory(&memory, &observation(&[TYPE_PLAIN], &[2], &[17], 40));
        assert_eq!(seen.last_seen_turn[0], 40);
        assert_eq!(seen.remembered_owner[0], 2);
        assert_eq!(seen.remembered_army[0], 17);

        let fogged = update_memory(&seen, &observation(&[TYPE_FOG], &[0], &[0], 88));
        assert_eq!(fogged.last_seen_turn[0], 40, "age must keep counting from 40");
        assert_eq!(fogged.remembered_owner[0], 2);
        assert_eq!(fogged.remembered_army[0], 17);
    }

    #[test]
    fn castle_ownership_is_remembered_from_the_last_castle_sighting() {
        let memory = VisibleMemory::empty(1, 1);
        let castle = update_memory(&memory, &observation(&[TYPE_CASTLE], &[2], &[40], 10));
        assert!(castle.remembered_was_castle[0]);
        assert_eq!(castle.remembered_castle_owner[0], 2);

        // Seeing the same cell as a plain later must not rewrite the castle
        // owner — only a castle sighting does that.
        let plain = update_memory(&castle, &observation(&[TYPE_PLAIN], &[1], &[2], 11));
        assert!(!plain.remembered_was_castle[0]);
        assert_eq!(plain.remembered_castle_owner[0], 2);
    }

    #[test]
    fn an_unseen_cell_keeps_the_never_seen_sentinel() {
        let memory = VisibleMemory::empty(1, 1);
        let next = update_memory(&memory, &observation(&[TYPE_STRUCTURE_FOG], &[0], &[0], 5));
        assert_eq!(next.last_seen_turn[0], -1);
        assert!(!next.ever_visible[0]);
    }
}
