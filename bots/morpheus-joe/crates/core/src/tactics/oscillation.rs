//! The two-cell shuffle, and the two reasons it is allowed anyway.
//!
//! An army bouncing between the same pair of cells is the failure mode a
//! prior-only policy falls into most readily. [`blocks_oscillation`] refuses
//! the reverse of a recent move — unless the move reveals fog or takes an
//! enemy cell, which are the cases where repeating a segment is progress
//! rather than a loop.

use crate::belief::Action5;
use crate::board::memory::OWNER_ENEMY;
use crate::io::wire::Observation;

use super::*;

fn oscillation_history(
    prev_action: Option<Action5>,
    recent_actions: &[Action5],
) -> Vec<MoveSegment> {
    let source: Vec<Action5> = if !recent_actions.is_empty() {
        recent_actions.to_vec()
    } else if let Some(prev) = prev_action {
        vec![prev]
    } else {
        Vec::new()
    };
    let out: Vec<MoveSegment> = source.iter().filter_map(|&a| move_segment(a)).collect();
    if out.len() > OSCILLATION_HISTORY {
        out[out.len() - OSCILLATION_HISTORY..].to_vec()
    } else {
        out
    }
}

/// True when the move reverses a recent own-corridor edge.
///
/// The whole window, not only the previous step, so `A→B→C` then `C→B→A`
/// ping-pong is banned. A reverse onto enemy land, or onto a cell that unlocks
/// new vision, stays legal.
///
/// `reveal` is an optional precomputed [`reveal_count_grid`]. The Python calls
/// `newly_revealed_cells` here, which runs two whole-board dilations *per
/// candidate action*; the grid is the same quantity computed once, which is
/// what makes the redirect loop affordable at all. The two agree by
/// construction — see [`reveal_count_grid`] — and the `oscillation` parity
/// surface runs both paths on every case so the equivalence is checked rather
/// than asserted.
pub fn blocks_oscillation(
    action: Action5,
    prev_action: Option<Action5>,
    obs: &Observation,
    recent_actions: &[Action5],
    force_allow: bool,
    reveal: Option<&[i64]>,
) -> bool {
    if force_allow {
        return false;
    }
    let seg = match move_segment(action) {
        Some(seg) => seg,
        None => return false,
    };
    let g = Grids::new(obs);
    let (tr, tc) = seg.1;
    if !g.inside(tr, tc) {
        return true;
    }
    if g.owner(tr, tc) == OWNER_ENEMY {
        return false;
    }
    let revealed = match reveal {
        Some(grid) => grid[g.at(tr, tc)],
        None => newly_revealed_cells(obs, tr, tc),
    };
    if revealed > 0 {
        return false;
    }
    oscillation_history(prev_action, recent_actions)
        .into_iter()
        .any(|old| is_reverse_segment(seg, old))
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
    fn an_oscillating_reverse_is_blocked_but_an_enemy_take_is_not() {
        let mut obs = board(1, 3);
        obs.owner_grid[0] = 1;
        obs.army_grid[0] = 5;
        obs.owner_grid[1] = 1;
        obs.army_grid[1] = 5;
        // Previous move went (0,1) -> (0,0); the reverse is (0,0) -> (0,1).
        let prev = [0, 0, 1, 2, 0];
        assert!(blocks_oscillation(
            [0, 0, 0, 3, 0],
            Some(prev),
            &obs,
            &[],
            false,
            None
        ));
        obs.owner_grid[1] = 2;
        assert!(!blocks_oscillation(
            [0, 0, 0, 3, 0],
            Some(prev),
            &obs,
            &[],
            false,
            None
        ));
    }
}
