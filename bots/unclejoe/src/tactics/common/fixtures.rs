//! Hand-built frames for the tactics tests, shared by the tactic modules so
//! each one's tests are only its own cases.
//!
//! A frame here is a plain `Observation` filled cell by cell. It is not always
//! a position the engine could produce — several tests want a fogged cell
//! beside a lit one to isolate an arm of a predicate — and that is the point:
//! a trigger reads the frame it is handed and must answer for any of them.

use crate::board::memory::Memory;
use crate::io::wire::{Observation, OWNER_ME, OWNER_OPP, TYPE_GENERAL, TYPE_PLAIN};
use crate::tactics::common::reach::Reach;

pub const H: usize = 9;
pub const W: usize = 9;

pub fn at(row: usize, col: usize) -> usize {
    row * W + col
}

/// An all-plain, all-neutral, all-visible board with our general at (4, 4) on
/// 10 army, and no enemy anywhere.
pub fn frame(turn: i32) -> Observation {
    let mut obs = Observation::with_dims(H, W);
    obs.turn = turn;
    obs.type_grid.iter_mut().for_each(|t| *t = TYPE_PLAIN);
    put(&mut obs, 4, 4, TYPE_GENERAL, OWNER_ME, 10);
    obs
}

pub fn put(obs: &mut Observation, row: usize, col: usize, cell_type: i32, owner: i32, army: i32) {
    let i = at(row, col);
    obs.type_grid[i] = cell_type;
    obs.owner_grid[i] = owner;
    obs.army_grid[i] = army;
}

/// Fill the scalar row the way the engine does: our total, and the opponent's
/// total with nothing hidden. A test that wants hidden army overwrites
/// `opp_army` afterwards.
pub fn total_armies(obs: &mut Observation) {
    obs.my_army = (0..H * W)
        .filter(|&i| obs.owner_grid[i] == OWNER_ME)
        .map(|i| obs.army_grid[i])
        .sum();
    obs.opp_army = (0..H * W)
        .filter(|&i| obs.owner_grid[i] == OWNER_OPP)
        .map(|i| obs.army_grid[i])
        .sum();
}

/// Memory that has seen these frames in order — the way to give a predicate a
/// past without playing a game.
pub fn remembering(frames: &[&Observation]) -> Memory {
    let mut mem = Memory::new(H, W);
    for frame in frames {
        mem.update(frame);
    }
    mem
}

/// Evaluate one tactic against a frame with no history but that frame.
pub fn fire_with<T>(
    obs: &Observation,
    evaluate: fn(&Observation, &Memory, &mut Reach) -> Option<T>,
) -> Option<T> {
    evaluate(obs, &remembering(&[obs]), &mut Reach::default())
}
