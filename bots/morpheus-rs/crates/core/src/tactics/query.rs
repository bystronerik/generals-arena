//! Read-only questions about the board. Nothing here decides anything.
//!
//! Visibility, generals, own structures, army concentration, the
//! reveal-count grid, and the belief's posterior on the enemy general. Every
//! function is a pure read of an observation, a memory, or a belief; the
//! planners and the scorer are the callers that turn these answers into
//! moves.

use crate::belief::BeliefState;
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY, TYPE_CASTLE, TYPE_GENERAL,
};
use crate::board::observe::visibility_from_owned;
use crate::board::transition::DEATHTOUCH_TURN;
use crate::io::wire::Observation;

use super::*;

/// How many currently invisible cells become visible if we own `dest`.
pub fn newly_revealed_cells(obs: &Observation, dest_r: i32, dest_c: i32) -> i64 {
    let g = Grids::new(obs);
    let n = obs.h * obs.w;
    let mut owned = vec![false; n];
    for i in 0..n {
        owned[i] = obs.owner_grid[i] as i32 == 1;
    }
    if owned[g.at(dest_r, dest_c)] {
        return 0;
    }
    let before = visibility_from_owned(&owned, obs.h, obs.w);
    owned[g.at(dest_r, dest_c)] = true;
    let after = visibility_from_owned(&owned, obs.h, obs.w);
    (0..n).filter(|&i| after[i] && !before[i]).count() as i64
}

/// Per-cell `newly_revealed_cells` for the whole board at once.
///
/// Visibility is a 3×3 dilation of ownership, so owning one new cell reveals
/// exactly the currently-invisible cells inside that cell's 3×3 box. Cells we
/// already own reveal nothing.
pub fn reveal_count_grid(obs: &Observation) -> Vec<i64> {
    let (h, w) = (obs.h, obs.w);
    let n = h * w;
    let mut owned = vec![false; n];
    for i in 0..n {
        owned[i] = obs.owner_grid[i] as i32 == 1;
    }
    let before = visibility_from_owned(&owned, h, w);
    let mut out = vec![0i64; n];
    for r in 0..h as i32 {
        for c in 0..w as i32 {
            let at = (r * w as i32 + c) as usize;
            if owned[at] {
                continue;
            }
            let mut box_count = 0i64;
            for dr in -1..=1i32 {
                for dc in -1..=1i32 {
                    let (nr, nc) = (r + dr, c + dc);
                    if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 {
                        continue;
                    }
                    if !before[(nr * w as i32 + nc) as usize] {
                        box_count += 1;
                    }
                }
            }
            out[at] = box_count;
        }
    }
    out
}

// ------------------------------------------------------------- board queries

pub fn enemy_is_visible(obs: &Observation, memory: &VisibleMemory) -> bool {
    let n = obs.h * obs.w;
    if (0..n).any(|i| obs.owner_grid[i] as i32 == OWNER_ENEMY) {
        return true;
    }
    (0..n).any(|i| memory.known_enemy_general[i])
}

pub fn enemy_general_visible(obs: &Observation, memory: &VisibleMemory) -> bool {
    let n = obs.h * obs.w;
    if (0..n)
        .any(|i| obs.type_grid[i] as i32 == TYPE_GENERAL && obs.owner_grid[i] as i32 == OWNER_ENEMY)
    {
        return true;
    }
    (0..n).any(|i| memory.known_enemy_general[i])
}

/// Own general and own castles, latched or currently visible.
pub fn own_structure_mask(obs: &Observation, memory: &VisibleMemory) -> Vec<bool> {
    let n = obs.h * obs.w;
    let mut out = vec![false; n];
    for i in 0..n {
        let own = obs.owner_grid[i] as i32 == 1;
        let t = obs.type_grid[i] as i32;
        let gen = memory.own_general[i] || (t == TYPE_GENERAL && own);
        let castle = memory.known_castle[i] || t == TYPE_CASTLE;
        out[i] = gen || (castle && own);
    }
    out
}

/// Largest army sitting on an own general or castle.
pub fn structure_idle_army(obs: &Observation, memory: &VisibleMemory) -> i64 {
    let structures = own_structure_mask(obs, memory);
    let n = obs.h * obs.w;
    let mut best: Option<i64> = None;
    for i in 0..n {
        if structures[i] {
            let a = obs.army_grid[i] as i64;
            best = Some(match best {
                Some(current) => current.max(a),
                None => a,
            });
        }
    }
    best.unwrap_or(0)
}

/// Latched, else visible, own general cell.
pub fn own_general_cell(obs: &Observation, memory: &VisibleMemory) -> Option<Cell> {
    let n = obs.h * obs.w;
    for i in 0..n {
        if memory.own_general[i] {
            return Some((i / obs.w, i % obs.w));
        }
    }
    for i in 0..n {
        if obs.type_grid[i] as i32 == TYPE_GENERAL && obs.owner_grid[i] as i32 == 1 {
            return Some((i / obs.w, i % obs.w));
        }
    }
    None
}

/// Latched, else currently visible, enemy general cell.
pub fn known_enemy_general_cell(obs: &Observation, memory: &VisibleMemory) -> Option<Cell> {
    let n = obs.h * obs.w;
    for i in 0..n {
        if memory.known_enemy_general[i] {
            return Some((i / obs.w, i % obs.w));
        }
    }
    for i in 0..n {
        if obs.type_grid[i] as i32 == TYPE_GENERAL && obs.owner_grid[i] as i32 == OWNER_ENEMY {
            return Some((i / obs.w, i % obs.w));
        }
    }
    None
}

/// Cell holding the largest own army; ties by row-major order.
///
/// `exclude` drops one cell (the floored general) so gather targets a stack
/// that can actually march, and falls back to the whole board when there is no
/// other own cell.
pub fn king_cell(obs: &Observation, exclude: Option<Cell>) -> Option<Cell> {
    let n = obs.h * obs.w;
    let mut own: Vec<bool> = (0..n).map(|i| obs.owner_grid[i] as i32 == 1).collect();
    if let Some(cell) = exclude {
        let at = cell.0 * obs.w + cell.1;
        if own[at] && own.iter().filter(|&&v| v).count() > 1 {
            own[at] = false;
        }
    }
    let max_a = (0..n)
        .filter(|&i| own[i])
        .map(|i| obs.army_grid[i] as i64)
        .max()?;
    (0..n)
        .find(|&i| own[i] && obs.army_grid[i] as i64 == max_a)
        .map(|i| (i / obs.w, i % obs.w))
}

/// `(max_share, max_army, total_army)` on own land.
///
/// `max_army` ignores `exclude` so the commitment and thrash logic measures
/// against the largest *movable* stack; `total_army` stays the full total.
pub fn army_concentration(obs: &Observation, exclude: Option<Cell>) -> (f64, i64, i64) {
    let n = obs.h * obs.w;
    let own: Vec<bool> = (0..n).map(|i| obs.owner_grid[i] as i32 == 1).collect();
    if !own.iter().any(|&v| v) {
        return (0.0, 0, 0);
    }
    let total: i64 = (0..n)
        .filter(|&i| own[i])
        .map(|i| obs.army_grid[i] as i64)
        .sum();
    let mut movable = own.clone();
    if let Some(cell) = exclude {
        let at = cell.0 * obs.w + cell.1;
        if movable[at] && movable.iter().filter(|&&v| v).count() > 1 {
            movable[at] = false;
        }
    }
    let max_a = (0..n)
        .filter(|&i| movable[i])
        .map(|i| obs.army_grid[i] as i64)
        .max()
        .unwrap_or(0);
    if total <= 0 {
        return (0.0, max_a, 0);
    }
    (max_a as f64 / total as f64, max_a, total)
}

/// The general cell while the garrison floor pins it, else `None`.
pub(super) fn movable_exclude_cell(obs: &Observation, memory: &VisibleMemory) -> Option<Cell> {
    let turn = turn_of(obs);
    if !(GARRISON_FLOOR_FROM..DEATHTOUCH_TURN).contains(&turn) {
        return None;
    }
    own_general_cell(obs, memory)
}

// ------------------------------------------------------------- seek targeting

/// Mode of the particle posterior over the enemy general's cell.
///
/// Weighted, with the particle count breaking ties when every weight is zero,
/// and first-seen order breaking the rest — Python's `max` keeps the first
/// maximum and its `dict` preserves insertion order, so the tie rule is the
/// order the particles introduced each cell.
pub fn believed_enemy_general(belief: Option<&BeliefState>) -> Option<Cell> {
    let belief = belief?;
    if belief.n() == 0 {
        return None;
    }
    let enemy = belief.enemy_seat();
    let mut mass: Vec<(Cell, f64, usize)> = Vec::new();
    for particle in &belief.particles {
        let g = particle.state.general_positions[enemy];
        if g[0] < 0 || g[1] < 0 {
            continue;
        }
        let cell = (g[0] as usize, g[1] as usize);
        match mass.iter_mut().find(|entry| entry.0 == cell) {
            Some(entry) => {
                entry.1 += particle.weight.max(0.0);
                entry.2 += 1;
            }
            None => mass.push((cell, particle.weight.max(0.0), 1)),
        }
    }
    let mut best: Option<&(Cell, f64, usize)> = None;
    for entry in &mass {
        let better = match best {
            None => true,
            Some(current) => (entry.1, entry.2) > (current.1, current.2),
        };
        if better {
            best = Some(entry);
        }
    }
    best.map(|entry| entry.0)
}
