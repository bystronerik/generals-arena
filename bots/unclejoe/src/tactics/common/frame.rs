//! What one frame states about the army it does not show.
//!
//! Only quantities a turn can change belong here. Anything a rule keeps true
//! for the whole game is settled once in `board::memory` instead — that
//! module's docs give the split. This is here rather than in `board` because
//! it is a *bound a tactic reasons with*, not a fact the network consumes,
//! and both tactics must read it the same way.

use crate::io::wire::{Observation, OWNER_OPP};

/// Enemy army we cannot account for: the frame states the opponent's total
/// exactly, so everything not sitting on a visible enemy cell is somewhere in
/// the fog. A sound bound on any one hidden cell, since it bounds all of them
/// together.
pub fn hidden_army(obs: &Observation) -> i32 {
    let visible: i32 = (0..obs.h * obs.w)
        .filter(|&i| obs.owner_grid[i] == OWNER_OPP)
        .map(|i| obs.army_grid[i])
        .sum();
    (obs.opp_army - visible).max(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::wire::TYPE_PLAIN;
    use crate::tactics::common::fixtures::{frame, put, total_armies};

    #[test]
    fn hidden_army_is_the_unaccounted_remainder() {
        let mut obs = frame(0);
        put(&mut obs, 0, 8, TYPE_PLAIN, OWNER_OPP, 7);
        total_armies(&mut obs);
        assert_eq!(hidden_army(&obs), 0); // every enemy unit is in sight

        obs.opp_army = 40;
        assert_eq!(hidden_army(&obs), 33);

        // A frame that reports less than we can see cannot mean "negative
        // hidden army"; the floor is zero.
        obs.opp_army = 3;
        assert_eq!(hidden_army(&obs), 0);
    }
}
