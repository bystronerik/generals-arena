//! Emergency defence: is the general about to fall, and what answers it.
//!
//! [`general_threat`] estimates arrival for every visible enemy stack —
//! erring toward defence, because the attacker may collect en route — and
//! [`defend_general_move`] is the forced reinforcement, which fires only when
//! arrival is imminent so that a wave loitering at the detection edge does not
//! divert the army every turn.

use crate::belief::Action5;
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY,
};
use crate::board::transition::{DEATHTOUCH_TURN, DIRECTIONS};
use crate::io::wire::Observation;

use super::*;

/// A visible enemy stack that can take our general: `(cell, arrival_steps)`.
///
/// A stack at path distance `d` arrives with `army - d` while the garrison
/// grows `d / 2`. Ties count as threats — the attacker may collect en route, so
/// the estimate errs toward defense. From deathtouch any stack that can reach
/// with one army is lethal.
pub fn general_threat(obs: &Observation, memory: &VisibleMemory) -> Option<(Cell, i32)> {
    let gcell = own_general_cell(obs, memory)?;
    let garrison = obs.army_grid[gcell.0 * obs.w + gcell.1] as i64;
    let turn = turn_of(obs);
    let field = path_distance_field(obs, &[gcell]);
    let n = obs.h * obs.w;

    let mut best: Option<(Cell, i32)> = None;
    let mut best_army = 0i64;
    for i in 0..n {
        if obs.owner_grid[i] as i32 != OWNER_ENEMY {
            continue;
        }
        let d = field.dist[i];
        if !(1..=DEFENSE_RADIUS).contains(&d) {
            continue;
        }
        let army = obs.army_grid[i] as i64;
        let arrival = army - d as i64;
        let lethal = if turn >= DEATHTOUCH_TURN {
            arrival >= 1
        } else {
            arrival >= garrison + (d / 2) as i64
        };
        if !lethal {
            continue;
        }
        let cell = (i / obs.w, i % obs.w);
        let better = match best {
            None => true,
            Some((_, best_d)) => (d, -army) < (best_d, -best_army),
        };
        if better {
            best = Some((cell, d));
            best_army = army;
        }
    }
    best
}

/// Best emergency response: capture the threat stack, else reinforce.
///
/// Reinforcement only counts if it lands on the general before the threat does
/// (`r <= d - 1`); the biggest such stack moves one gradient step home.
pub fn defend_general_move(
    obs: &Observation,
    memory: &VisibleMemory,
    threat: (Cell, i32),
) -> Option<Action5> {
    let (tcell, d) = threat;
    let gcell = own_general_cell(obs, memory)?;
    let g = Grids::new(obs);
    let t_army = obs.army_grid[tcell.0 * obs.w + tcell.1] as i64;

    // (a) Kill the threat outright from an adjacent own cell.
    let mut best_cap: Option<(i64, Action5)> = None;
    for (dd, (dr, dc)) in DIRECTIONS.iter().enumerate() {
        let sr = tcell.0 as i32 - dr;
        let sc = tcell.1 as i32 - dc;
        if !g.inside(sr, sc) {
            continue;
        }
        if g.owner(sr, sc) != 1 || (sr as usize, sc as usize) == gcell {
            continue;
        }
        let moved = g.army(sr, sc) - 1;
        if moved > t_army && best_cap.map_or(true, |(best, _)| moved > best) {
            best_cap = Some((moved, [0, sr, sc, dd as i32, 0]));
        }
    }
    if let Some((_, action)) = best_cap {
        return Some(action);
    }

    // (b) Reinforce the general in time.
    let field = path_distance_field(obs, &[gcell]);
    let n = obs.h * obs.w;
    let mut best: Option<(i64, Action5)> = None;
    for i in 0..n {
        if obs.owner_grid[i] as i32 != 1 || (obs.army_grid[i] as i64) < 2 {
            continue;
        }
        let cell = (i / obs.w, i % obs.w);
        if cell == gcell {
            continue;
        }
        let rr = field.dist[i];
        if !(1..=d - 1).contains(&rr) {
            continue;
        }
        let a = obs.army_grid[i] as i64;
        if best.map_or(false, |(best_a, _)| a <= best_a) {
            continue;
        }
        let (r, c) = (cell.0 as i32, cell.1 as i32);
        for (dd, (dr, dc)) in DIRECTIONS.iter().enumerate() {
            let (nr, nc) = (r + dr, c + dc);
            if !g.inside(nr, nc) {
                continue;
            }
            if field.get(nr, nc) != rr - 1 || g.owner(nr, nc) != 1 {
                continue;
            }
            best = Some((a, [0, r, c, dd as i32, 0]));
            break;
        }
    }
    best.map(|(_, action)| action)
}
