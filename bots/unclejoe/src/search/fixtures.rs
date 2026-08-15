//! Hand-built positions for the search tests.
//!
//! Two kinds, because this layer is entered from both ends. A prover wants the
//! board the rules act on, with no fog story attached, so [`board`] builds a
//! [`Sim`] directly. The two entry points that read a *frame* —
//! [`Sim::from_frame`] and [`Afterstate`](super::afterstate::Afterstate) —
//! want the other kind, so [`wire_frame`] builds an `Observation` and
//! [`remembering`] gives it a past.
//!
//! `search` has its own fixtures rather than borrowing the tactics layer's,
//! for the reason the layering rule exists: this module sits below `tactics`
//! and must not name it.

use crate::board::memory::Memory;
use crate::io::wire::{
    Observation, OWNER_ME, TYPE_CASTLE, TYPE_FOG, TYPE_MOUNTAIN, TYPE_PLAIN,
    TYPE_STRUCTURE_IN_FOG,
};

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

/// An all-plain, all-visible, all-neutral frame. A test that wants fog, an
/// owner or an army writes it in with [`set`].
pub fn wire_frame(h: usize, w: usize, turn: i32) -> Observation {
    let mut obs = Observation::with_dims(h, w);
    obs.turn = turn;
    obs.type_grid.iter_mut().for_each(|t| *t = TYPE_PLAIN);
    obs
}

pub fn set(obs: &mut Observation, row: usize, col: usize, cell_type: i32, owner: i32, army: i32) {
    let cell = row * obs.w + col;
    obs.type_grid[cell] = cell_type;
    obs.owner_grid[cell] = owner;
    obs.army_grid[cell] = army;
}

/// Fog the frame the way the engine would.
///
/// [`wire_frame`] starts out showing everything, which no frame off the wire
/// ever does: RULES.md §06 gives a player the 3×3 neighbourhood of every cell
/// they own and nothing else. A test *about* visibility needs a frame that
/// obeys the rule it is testing, so this applies it — the engine's
/// `get_visibility` pool, then its split of the dark cells into plain fog and
/// unresolved structures.
pub fn apply_fog(obs: &mut Observation) {
    let (h, w) = (obs.h, obs.w);
    let mut lit = vec![false; h * w];
    for row in 0..h {
        for col in 0..w {
            'pool: for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    let (r, c) = (row as i32 + dr, col as i32 + dc);
                    if r < 0 || c < 0 || r >= h as i32 || c >= w as i32 {
                        continue;
                    }
                    if obs.owner_grid[r as usize * w + c as usize] == OWNER_ME {
                        lit[row * w + col] = true;
                        break 'pool;
                    }
                }
            }
        }
    }
    for (cell, lit) in lit.iter().enumerate() {
        if *lit {
            continue;
        }
        obs.type_grid[cell] = match obs.type_grid[cell] {
            TYPE_MOUNTAIN | TYPE_CASTLE => TYPE_STRUCTURE_IN_FOG,
            _ => TYPE_FOG,
        };
        obs.owner_grid[cell] = 0;
        obs.army_grid[cell] = 0;
    }
}

/// Memory that has seen these frames in order — a past without a game.
pub fn remembering(frames: &[&Observation]) -> Memory {
    let first = frames.first().expect("at least one frame");
    let mut mem = Memory::new(first.h, first.w);
    for frame in frames {
        mem.update(frame);
    }
    mem
}
