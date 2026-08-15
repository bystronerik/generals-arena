//! Where to drive: goal cells, the hunt target, and the assembly point.
//!
//! [`seek_goals`] is the goal set every distance field floods from, and its
//! ordering is the rule that matters — enemy land stays ahead of the belief's
//! posterior on purpose, because that is what keeps incursions near home
//! scored as progress and the border under broad pressure.

use crate::belief::BeliefState;
use crate::board::memory::{
    VisibleMemory, OWNER_ENEMY, TYPE_GENERAL,
};
use crate::io::wire::Observation;

use super::*;

/// Cells to drive toward: enemy general, else enemy land, else the posterior.
///
/// Enemy land stays ahead of the belief on purpose — it is what keeps
/// incursions near home scored as progress, and the border under broad
/// pressure. The directed hunt is a separate bonus field in
/// [`heuristic_action_scores`], not a goal replacement.
pub fn seek_goals(
    obs: &Observation,
    memory: &VisibleMemory,
    belief: Option<&BeliefState>,
) -> Vec<Cell> {
    let n = obs.h * obs.w;
    let rc = |i: usize| (i / obs.w, i % obs.w);

    if let Some(i) = (0..n).find(|&i| memory.known_enemy_general[i]) {
        return vec![rc(i)];
    }
    if let Some(i) = (0..n)
        .find(|&i| obs.type_grid[i] as i32 == TYPE_GENERAL && obs.owner_grid[i] as i32 == OWNER_ENEMY)
    {
        return vec![rc(i)];
    }
    let enemy: Vec<Cell> = (0..n)
        .filter(|&i| obs.owner_grid[i] as i32 == OWNER_ENEMY)
        .map(rc)
        .collect();
    if !enemy.is_empty() {
        return enemy;
    }
    if let Some(cell) = believed_enemy_general(belief) {
        return vec![cell];
    }
    let own = (0..n).find(|&i| memory.own_general[i]).or_else(|| {
        (0..n).find(|&i| obs.type_grid[i] as i32 == TYPE_GENERAL && obs.owner_grid[i] as i32 == 1)
    });
    match own {
        Some(i) => {
            let (gr, gc) = rc(i);
            vec![(obs.h - 1 - gr, obs.w - 1 - gc)]
        }
        None => Vec::new(),
    }
}

/// Drive toward the known/believed enemy general, else the nearest enemy land.
///
/// "Nearest" is measured from the largest own stack so the king does not aim
/// at a centroid behind mountains.
pub fn enemy_seek_target(
    obs: &Observation,
    memory: &VisibleMemory,
    belief: Option<&BeliefState>,
) -> Option<Cell> {
    let goals = seek_goals(obs, memory, belief);
    if goals.is_empty() {
        return None;
    }
    if goals.len() == 1 {
        return Some(goals[0]);
    }
    let king = king_cell(obs, None)?;
    let mut best = goals[0];
    let mut best_d = i64::MAX;
    for goal in &goals {
        let d = (goal.0 as i64 - king.0 as i64).abs() + (goal.1 as i64 - king.1 as i64).abs();
        if d < best_d {
            best_d = d;
            best = *goal;
        }
    }
    Some(best)
}

/// Where the attack wave forms: the frontmost own cell toward the hunt target.
///
/// Gathering used to target the largest movable stack, a target that moves
/// whenever any cell's army changes, so flows chased it and the army stayed
/// dribbled. This point is deterministic and rolls forward with each take.
pub fn wave_assembly_cell(
    obs: &Observation,
    memory: &VisibleMemory,
    belief: Option<&BeliefState>,
    exclude: Option<Cell>,
) -> Option<Cell> {
    let hunt = known_enemy_general_cell(obs, memory).or_else(|| believed_enemy_general(belief));
    let hunt = match hunt {
        Some(cell) => cell,
        None => return king_cell(obs, exclude),
    };
    let n = obs.h * obs.w;
    let mut own: Vec<bool> = (0..n).map(|i| obs.owner_grid[i] as i32 == 1).collect();
    if let Some(cell) = exclude {
        let at = cell.0 * obs.w + cell.1;
        if own[at] && own.iter().filter(|&&v| v).count() > 1 {
            own[at] = false;
        }
    }
    if !own.iter().any(|&v| v) {
        return None;
    }
    let field = path_distance_field(obs, &[hunt]);
    let mut dmin = i32::MAX;
    for i in 0..n {
        if own[i] && field.dist[i] >= 0 {
            dmin = dmin.min(field.dist[i]);
        }
    }
    if dmin == i32::MAX {
        return king_cell(obs, exclude);
    }
    let front: Vec<usize> = (0..n)
        .filter(|&i| own[i] && field.dist[i] == dmin)
        .collect();
    let max_a = front
        .iter()
        .map(|&i| obs.army_grid[i] as i64)
        .max()
        .unwrap_or(0);
    front
        .into_iter()
        .find(|&i| obs.army_grid[i] as i64 == max_a)
        .map(|i| (i / obs.w, i % obs.w))
}
