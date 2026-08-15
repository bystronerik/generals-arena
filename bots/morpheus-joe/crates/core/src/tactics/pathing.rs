//! Distance fields, and progress along them.
//!
//! [`path_distance_field`] is the BFS every planner shares — one flood from
//! the goal set, walking around mountains — and [`move_progress`] /
//! [`path_progress`] are how a candidate move is scored against it.

use crate::board::memory::{
    TYPE_MOUNTAIN,
    TYPE_STRUCTURE_FOG,
};
use crate::board::transition::DIRECTIONS;
use crate::io::wire::Observation;

use super::*;

/// Manhattan progress toward `target` (+1 closer, −1 farther, 0 lateral).
pub fn move_progress(sr: i32, sc: i32, tr: i32, tc: i32, target: Option<Cell>) -> f64 {
    let target = match target {
        Some(target) => target,
        None => return 0.0,
    };
    let before = (sr - target.0 as i32).abs() + (sc - target.1 as i32).abs();
    let after = (tr - target.0 as i32).abs() + (tc - target.1 as i32).abs();
    (before - after) as f64
}

/// Path-aware progress: the drop in BFS distance to the seek goals.
///
/// Falls back to Manhattan when the field is missing or both ends are
/// unreachable, so fog carving that opens a route still gets a signal.
pub fn path_progress(
    sr: i32,
    sc: i32,
    tr: i32,
    tc: i32,
    dist_field: Option<&DistanceField>,
    fallback_target: Option<Cell>,
) -> f64 {
    let field = match dist_field {
        Some(field) => field,
        None => return move_progress(sr, sc, tr, tc, fallback_target),
    };
    if !field.inside(sr, sc) || !field.inside(tr, tc) {
        return move_progress(sr, sc, tr, tc, fallback_target);
    }
    let before = field.get(sr, sc);
    let after = field.get(tr, tc);
    if before < 0 && after < 0 {
        return move_progress(sr, sc, tr, tc, fallback_target);
    }
    if before < 0 {
        return 2.0;
    }
    if after < 0 {
        return -2.0;
    }
    (before - after) as f64
}

// ------------------------------------------------------------- path distances

/// BFS distance to the nearest goal through passable cells; `-1` unreachable.
pub struct DistanceField {
    pub h: i32,
    pub w: i32,
    pub dist: Vec<i32>,
}

impl DistanceField {
    #[inline]
    pub fn inside(&self, r: i32, c: i32) -> bool {
        r >= 0 && c >= 0 && r < self.h && c < self.w
    }

    #[inline]
    pub fn get(&self, r: i32, c: i32) -> i32 {
        self.dist[(r * self.w + c) as usize]
    }
}

pub fn path_distance_field(obs: &Observation, goals: &[Cell]) -> DistanceField {
    let (h, w) = (obs.h as i32, obs.w as i32);
    let n = obs.h * obs.w;
    let mut passable = vec![false; n];
    for i in 0..n {
        let t = obs.type_grid[i] as i32;
        passable[i] = t != TYPE_MOUNTAIN && t != TYPE_STRUCTURE_FOG;
    }
    let mut dist = vec![-1i32; n];
    let mut queue: Vec<(i32, i32)> = Vec::with_capacity(n);
    for &(gr, gc) in goals {
        let (gr, gc) = (gr as i32, gc as i32);
        if gr < 0 || gc < 0 || gr >= h || gc >= w {
            continue;
        }
        let at = (gr * w + gc) as usize;
        if !passable[at] || dist[at] == 0 {
            continue;
        }
        dist[at] = 0;
        queue.push((gr, gc));
    }
    let mut head = 0usize;
    while head < queue.len() {
        let (r, c) = queue[head];
        head += 1;
        let base = dist[(r * w + c) as usize];
        for (dr, dc) in DIRECTIONS {
            let (nr, nc) = (r + dr, c + dc);
            if nr < 0 || nc < 0 || nr >= h || nc >= w {
                continue;
            }
            let at = (nr * w + nc) as usize;
            if !passable[at] || dist[at] >= 0 {
                continue;
            }
            dist[at] = base + 1;
            queue.push((nr, nc));
        }
    }
    DistanceField { h, w, dist }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(h: usize, w: usize) -> Observation {
        let mut obs = Observation::with_dims(h, w);
        for i in 0..h * w {
            obs.type_grid[i] = 1;
        }
        obs
    }

    #[test]
    fn the_distance_field_walks_around_a_mountain() {
        let mut obs = board(3, 3);
        let centre = obs.idx(1, 1);
        obs.type_grid[centre] = TYPE_MOUNTAIN as u8;
        let field = path_distance_field(&obs, &[(0, 0)]);
        assert_eq!(field.get(0, 0), 0);
        assert_eq!(field.get(1, 1), -1);
        assert_eq!(field.get(2, 2), 4);
    }
}
