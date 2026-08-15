//! Hand-built positions for the search tests.
//!
//! The tactics tests build *frames* and let memory derive the rest, because a
//! trigger's job is to read a frame. A search test wants the opposite: the
//! board the rules act on, with no fog story attached, so these build a
//! [`Sim`] directly. The one path that turns a frame into a `Sim` —
//! [`Sim::from_frame`] — is tested from frames, in `sim.rs`.
//!
//! `search` has its own fixtures rather than borrowing the tactics layer's,
//! for the reason the layering rule exists: this module sits below `tactics`
//! and must not name it.

use super::sim::{Move, Sim, ME, NEUTRAL, OPP};

/// Flat index of `(row, col)` on a board `w` wide. The width comes first so a
/// call reads as "on a board this wide, the cell here".
pub fn at(w: usize, row: usize, col: usize) -> usize {
    row * w + col
}

/// A board under construction: everything plain, neutral, and passable until
/// a test says otherwise.
pub struct Setup {
    sim: Sim,
    own_general_set: bool,
}

impl Setup {
    fn put(&mut self, row: usize, col: usize, owner: u8, army: i32) -> usize {
        let cell = at(self.sim.w, row, col);
        self.sim.owner[cell] = owner;
        self.sim.army[cell] = army;
        cell
    }

    pub fn mine(&mut self, row: usize, col: usize, army: i32) -> usize {
        self.put(row, col, ME, army)
    }

    pub fn opp(&mut self, row: usize, col: usize, army: i32) -> usize {
        self.put(row, col, OPP, army)
    }

    pub fn own_general(&mut self, row: usize, col: usize, army: i32) -> usize {
        let cell = self.put(row, col, ME, army);
        self.sim.structures.push(cell);
        self.sim.own_general = cell;
        self.own_general_set = true;
        cell
    }

    pub fn enemy_general(&mut self, row: usize, col: usize, army: i32) -> usize {
        let cell = self.put(row, col, OPP, army);
        self.sim.structures.push(cell);
        self.sim.enemy_general = Some(cell);
        cell
    }

    pub fn castle(&mut self, row: usize, col: usize, owner: u8, army: i32) -> usize {
        let cell = self.put(row, col, owner, army);
        self.sim.structures.push(cell);
        cell
    }

    /// Impassable to both sides — a mountain we have seen.
    pub fn mountain(&mut self, row: usize, col: usize) -> usize {
        let cell = self.put(row, col, NEUTRAL, 0);
        self.sim.passable_me[cell] = false;
        self.sim.passable_opp[cell] = false;
        cell
    }

    /// A structure in fog: they may walk it, we may not, because it is a
    /// mountain or a castle and the frame cannot say which.
    pub fn closed_to_us(&mut self, row: usize, col: usize) -> usize {
        let cell = at(self.sim.w, row, col);
        self.sim.passable_me[cell] = false;
        cell
    }
}

/// Build a board. The closure places what the test is about; a test that does
/// not place our general is a test that has not said what it is testing.
pub fn board(h: usize, w: usize, place: impl FnOnce(&mut Setup)) -> Sim {
    let mut sim = Sim::blank(h, w);
    // RULES.md §07's threshold, spelled out rather than imported: the
    // constants block lives in `tactics`, above this module. A test that cares
    // sets `turn` on either side of it.
    sim.deathtouch_turn = 800;
    sim.passable_me.iter_mut().for_each(|p| *p = true);
    sim.passable_opp.iter_mut().for_each(|p| *p = true);

    let mut setup = Setup { sim, own_general_set: false };
    place(&mut setup);
    assert!(setup.own_general_set, "a fixture board needs our general");
    setup.sim
}

/// The move from `(row, col)` in `dir`, full-army.
pub fn step(w: usize, row: usize, col: usize, dir: u8) -> Move {
    Move { from: at(w, row, col), dir, half: false }
}
