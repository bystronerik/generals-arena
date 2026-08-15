//! The castle savings pipeline: where to build, and what to pay in.
//!
//! [`castle_build_site`] picks a base-price own plain near the general and
//! never on the front, [`castle_tithe_move`] feeds it on a period, and
//! [`opponent_mobile`] is the check that stands the whole thing down when the
//! opponent has an army worth answering.

use crate::board::action::live_build_cost;
use crate::belief::Action5;
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY, TYPE_CASTLE, TYPE_GENERAL, TYPE_MOUNTAIN,
};
use crate::board::transition::{BASE_COST, DIRECTIONS};
use crate::io::wire::Observation;

use super::*;

/// Savings/build site: the base-price own plain cell nearest the general.
///
/// Active only inside the build window with fewer than `CASTLE_TARGET` own
/// castles. The site must cost exactly `BASE_COST` — never pay a surcharge —
/// and sit at least `CASTLE_SAFE_ENEMY_DIST` Manhattan from any visible enemy
/// cell, because a castle on the front is a gift.
pub fn castle_build_site(obs: &Observation, memory: &VisibleMemory) -> Option<Cell> {
    let turn = turn_of(obs);
    if !(GARRISON_FLOOR_FROM..=CASTLE_WINDOW_UNTIL).contains(&turn) {
        return None;
    }
    let n = obs.h * obs.w;
    let visible_castles = (0..n)
        .filter(|&i| obs.type_grid[i] as i32 == TYPE_CASTLE && obs.owner_grid[i] as i32 == 1)
        .count() as i64;
    let latched = (0..n)
        .filter(|&i| memory.known_castle[i] && obs.owner_grid[i] as i32 == 1)
        .count() as i64;
    if visible_castles.max(latched) >= CASTLE_TARGET {
        return None;
    }

    let cost = live_build_cost(obs, memory);
    let candidate: Vec<bool> = (0..n)
        .map(|i| {
            let t = obs.type_grid[i] as i32;
            obs.owner_grid[i] as i32 == 1
                && cost[i] == BASE_COST
                && t != TYPE_GENERAL
                && t != TYPE_CASTLE
                && t != TYPE_MOUNTAIN
        })
        .collect();
    if !candidate.iter().any(|&v| v) {
        return None;
    }

    let enemy_cells: Vec<Cell> = (0..n)
        .filter(|&i| obs.owner_grid[i] as i32 == OWNER_ENEMY)
        .map(|i| (i / obs.w, i % obs.w))
        .collect();
    let gcell = own_general_cell(obs, memory);

    let mut best: Option<Cell> = None;
    let mut best_key: Option<(i64, i64, usize, usize)> = None;
    for i in 0..n {
        if !candidate[i] {
            continue;
        }
        let (r, c) = (i / obs.w, i % obs.w);
        if !enemy_cells.is_empty() {
            let d_enemy = enemy_cells
                .iter()
                .map(|&(er, ec)| {
                    (er as i64 - r as i64).abs() + (ec as i64 - c as i64).abs()
                })
                .min()
                .unwrap();
            if d_enemy < CASTLE_SAFE_ENEMY_DIST as i64 {
                continue;
            }
        }
        let d_gen = match gcell {
            Some((gr, gc)) => (gr as i64 - r as i64).abs() + (gc as i64 - c as i64).abs(),
            None => 0,
        };
        // Sticky: a cell already holding a pile outranks a marginally closer
        // empty one, so the site does not churn and strand its savings.
        let pile = (obs.army_grid[i] as i64).min(BASE_COST as i64);
        let key = (-pile, d_gen, r, c);
        if best_key.is_none() || key < best_key.unwrap() {
            best_key = Some(key);
            best = Some((r, c));
        }
    }
    best
}

/// Opponent army free to move: scoreboard total minus one pinned per cell.
pub fn opponent_mobile(obs: &Observation) -> i64 {
    (obs.opp_army as i64 - obs.opp_land as i64).max(0)
}

/// One gather-step toward the savings site: the biggest catchment tip moves.
///
/// A full move along the BFS gradient through own land only — the tithe is
/// logistics, never combat.
pub fn castle_tithe_move(
    obs: &Observation,
    memory: &VisibleMemory,
    site: Cell,
) -> Option<Action5> {
    let g = Grids::new(obs);
    let field = path_distance_field(obs, &[site]);
    let gcell = own_general_cell(obs, memory);
    let n = obs.h * obs.w;

    let mut best: Option<Action5> = None;
    let mut best_army = 1i64;
    for i in 0..n {
        if obs.owner_grid[i] as i32 != 1 {
            continue;
        }
        let cell = (i / obs.w, i % obs.w);
        if cell == site || Some(cell) == gcell {
            continue;
        }
        let d0 = field.dist[i];
        if !(1..=CASTLE_CATCHMENT).contains(&d0) {
            continue;
        }
        let a = obs.army_grid[i] as i64;
        if a <= best_army {
            continue;
        }
        let (r, c) = (cell.0 as i32, cell.1 as i32);
        for (d, (dr, dc)) in DIRECTIONS.iter().enumerate() {
            let (nr, nc) = (r + dr, c + dc);
            if !g.inside(nr, nc) {
                continue;
            }
            if field.get(nr, nc) != d0 - 1 || g.owner(nr, nc) != 1 {
                continue;
            }
            best = Some([0, r, c, d as i32, 0]);
            best_army = a;
            break;
        }
    }
    best
}
