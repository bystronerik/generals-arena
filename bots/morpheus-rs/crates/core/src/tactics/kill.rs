//! The kill window: a forced finishing march on a visible enemy general.
//!
//! [`kill_plan`] is a greedy descent of the BFS gradient that must *arrive*
//! with more than the garrison plus its growth over the march, so a plan that
//! exists is a win rather than an attempt. It replans every turn and
//! disappears silently when the window closes.

use crate::belief::Action5;
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY, TYPE_FOG,
    TYPE_STRUCTURE_FOG,
};
use crate::board::transition::{DEATHTOUCH_TURN, DIRECTIONS};
use crate::io::wire::Observation;

use super::*;

/// Best winning march on the visible enemy general: `(steps, margin, first)`.
///
/// A greedy descent of the BFS gradient from each nearby own stack: full moves
/// that collect own armies en route, pay for neutral and enemy cells, and must
/// arrive with strictly more than the garrison plus its growth over the march.
/// From deathtouch any arrival wins. Replans every turn; if the window closes,
/// the plan silently disappears.
pub fn kill_plan(obs: &Observation, memory: &VisibleMemory) -> Option<(i32, i64, Action5)> {
    let g = Grids::new(obs);
    let gcell = known_enemy_general_cell(obs, memory)?;
    if g.owner(gcell.0 as i32, gcell.1 as i32) != OWNER_ENEMY {
        return None;
    }
    let garrison = g.army(gcell.0 as i32, gcell.1 as i32);
    let turn = turn_of(obs);
    let field = path_distance_field(obs, &[gcell]);

    let walk = |sr: i32, sc: i32| -> Option<(i32, i64, Action5)> {
        let mut army = g.army(sr, sc);
        let mut cur = (sr, sc);
        let mut first: Option<Action5> = None;
        let mut steps = 0i32;
        while field.get(cur.0, cur.1) > 1 {
            // Prefer collecting own armies; else the cheapest cell to cross.
            let mut best_n: Option<(i32, i32, usize)> = None;
            let mut best_key: Option<(i32, i64)> = None;
            for (d, (dr, dc)) in DIRECTIONS.iter().enumerate() {
                let (nr, nc) = (cur.0 + dr, cur.1 + dc);
                if !g.inside(nr, nc) {
                    continue;
                }
                if field.get(nr, nc) != field.get(cur.0, cur.1) - 1 {
                    continue;
                }
                // Fogged cells hide their army — a march priced on unknown
                // costs is a doomed march. Visible cells only.
                let t = g.kind(nr, nc);
                if t == TYPE_FOG || t == TYPE_STRUCTURE_FOG {
                    continue;
                }
                let o = g.owner(nr, nc);
                let a_n = g.army(nr, nc);
                let key = if o == 1 { (0, -a_n) } else { (1, a_n) };
                if best_key.is_none() || key < best_key.unwrap() {
                    best_key = Some(key);
                    best_n = Some((nr, nc, d));
                }
            }
            let (nr, nc, d) = best_n?;
            let moved = army - 1;
            if moved <= 0 {
                return None;
            }
            let o = g.owner(nr, nc);
            let a_n = g.army(nr, nc);
            if o == 1 {
                army = moved + a_n;
            } else {
                if moved <= a_n {
                    return None;
                }
                army = moved - a_n;
            }
            if first.is_none() {
                first = Some([0, cur.0, cur.1, d as i32, 0]);
            }
            cur = (nr, nc);
            steps += 1;
        }
        // The touch itself.
        let moved = army - 1;
        steps += 1;
        let need = if turn >= DEATHTOUCH_TURN {
            0
        } else {
            garrison + ((steps + 1) / 2) as i64
        };
        if moved <= need {
            return None;
        }
        if first.is_none() {
            // Already adjacent: the touch is the first move.
            for (d, (dr, dc)) in DIRECTIONS.iter().enumerate() {
                let (nr, nc) = (cur.0 + dr, cur.1 + dc);
                if nr >= 0 && nc >= 0 && nr as usize == gcell.0 && nc as usize == gcell.1 {
                    first = Some([0, cur.0, cur.1, d as i32, 0]);
                }
            }
        }
        Some((steps, moved - need, first?))
    };

    let n = obs.h * obs.w;
    let mut best: Option<(i32, i64, Action5)> = None;
    for i in 0..n {
        if obs.owner_grid[i] as i32 != 1 || (obs.army_grid[i] as i64) < 2 {
            continue;
        }
        let d0 = field.dist[i];
        if !(1..=KILL_HORIZON).contains(&d0) {
            continue;
        }
        let (r, c) = ((i / obs.w) as i32, (i % obs.w) as i32);
        let plan = match walk(r, c) {
            Some(plan) => plan,
            None => continue,
        };
        let better = match best {
            None => true,
            Some((bs, bm, _)) => (plan.0, -plan.1) < (bs, -bm),
        };
        if better {
            best = Some(plan);
        }
    }
    best
}

/// First step of the best winning kill march.
pub fn winning_kill_move(obs: &Observation, memory: &VisibleMemory) -> Option<Action5> {
    kill_plan(obs, memory).map(|plan| plan.2)
}
