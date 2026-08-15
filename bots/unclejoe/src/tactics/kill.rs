//! The kill tactic: can we take their general?
//!
//! At this milestone the file is only the trigger — the predicate that says a
//! kill proof is worth attempting. The proof search it will call arrives in
//! U3, and lands here, beside the predicate that gates it.
//!
//! The predicate is a **superset**: every position where a kill is provable
//! within the depth budget fires it. The converse is not claimed. Both
//! quantities it compares are deliberately generous — `reach_army` sums whole
//! cells rather than what they could actually send (a mover leaves one
//! behind, RULES.md §02) and ignores that only one of them moves per turn;
//! `last_seen_army` is a lower bound on what defends the general today. Each
//! error is in the fire-more direction, because a fire costs one search that
//! declines and a miss costs the point of the bot.

use crate::board::memory::Memory;
use crate::io::wire::{Observation, OWNER_ME};
use crate::tactics::common::Reach;
use crate::tactics::{DEATHTOUCH_TURN, SEARCH_DEPTH};

/// Why the trigger fired. `ArmyBound` when the army we can bring beats the
/// general's last-seen garrison; `Deathtouch` when the turn alone makes any
/// unit lethal (RULES.md §07).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cause {
    ArmyBound,
    Deathtouch,
}

impl Cause {
    pub fn name(self) -> &'static str {
        match self {
            Cause::ArmyBound => "army",
            Cause::Deathtouch => "deathtouch",
        }
    }
}

/// One kill trigger fire, with the numbers it fired on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fire {
    /// The enemy general's remembered cell.
    pub target: usize,
    pub cause: Cause,
    /// Total army on our movable cells within reach of it.
    pub reach_army: i32,
    /// The stale lower bound that `reach_army` was compared against.
    pub last_seen_army: i32,
    /// Turns since that sighting; `0` while the general is in view.
    pub stale_turns: i32,
}

/// Fire when the enemy general has been seen, some cell of ours that can move
/// at all is within `SEARCH_DEPTH` moves of it, and either deathtouch is live
/// or the army we could bring beats its last-seen garrison.
pub fn evaluate(obs: &Observation, mem: &Memory, reach: &mut Reach) -> Option<Fire> {
    let target = mem.enemy_general?;
    reach.compute(obs, mem, target, SEARCH_DEPTH);

    let mut reach_army = 0;
    for cell in 0..obs.h * obs.w {
        // A cell with one army cannot move (RULES.md §02), so it is not a
        // source of anything and does not count.
        if reach.reached(cell) && obs.owner_grid[cell] == OWNER_ME && obs.army_grid[cell] > 1 {
            reach_army += obs.army_grid[cell];
        }
    }
    // Armies are non-negative, so a zero total means no movable cell of ours
    // is in range at all.
    if reach_army == 0 {
        return None;
    }

    let cause = if reach_army > mem.last_seen_general_army {
        // Strictly more army takes the cell (RULES.md §05).
        Cause::ArmyBound
    } else if obs.turn >= DEATHTOUCH_TURN {
        // From turn 800 the garrison stops mattering: one unit that executes
        // onto the tile wins (RULES.md §07).
        Cause::Deathtouch
    } else {
        return None;
    };
    Some(Fire {
        target,
        cause,
        reach_army,
        last_seen_army: mem.last_seen_general_army,
        // `last_seen_turn` is set with `enemy_general`, so it is never the
        // pre-sighting sentinel here.
        stale_turns: obs.turn - mem.last_seen_turn,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::wire::{OWNER_OPP, TYPE_FOG, TYPE_GENERAL, TYPE_MOUNTAIN, TYPE_PLAIN};
    use crate::tactics::common::fixtures::{at, fire_with, frame, put, remembering, total_armies};

    fn fire(obs: &Observation) -> Option<Fire> {
        fire_with(obs, evaluate)
    }

    #[test]
    fn fires_at_every_depth_in_budget() {
        for depth in 1..=SEARCH_DEPTH as usize {
            let mut obs = frame(100);
            put(&mut obs, 0, 0, TYPE_GENERAL, OWNER_OPP, 5);
            put(&mut obs, 0, depth, TYPE_PLAIN, OWNER_ME, 12);
            total_armies(&mut obs);
            let kill = fire(&obs).expect("a kill inside the budget");
            assert_eq!(kill.cause, Cause::ArmyBound);
            assert_eq!(kill.target, at(0, 0));
            assert_eq!((kill.reach_army, kill.last_seen_army), (12, 5));
            assert_eq!(kill.stale_turns, 0);
        }
    }

    #[test]
    fn is_silent_past_the_depth_budget() {
        let mut obs = frame(100);
        put(&mut obs, 0, 0, TYPE_GENERAL, OWNER_OPP, 5);
        put(&mut obs, 0, SEARCH_DEPTH as usize + 1, TYPE_PLAIN, OWNER_ME, 99);
        total_armies(&mut obs);
        assert!(fire(&obs).is_none());
    }

    #[test]
    fn is_silent_before_the_general_is_seen() {
        // The general is there, in fog. Memory never learned it, so there is
        // nothing to prove a kill against.
        let mut obs = frame(100);
        put(&mut obs, 0, 0, TYPE_FOG, 0, 0);
        put(&mut obs, 0, 1, TYPE_PLAIN, OWNER_ME, 99);
        total_armies(&mut obs);
        assert!(fire(&obs).is_none());
    }

    #[test]
    fn is_silent_behind_remembered_mountains() {
        // The general's only two neighbours are mountains, and on this frame
        // both are back in fog: only memory can tell they are walls.
        let mut walled = frame(100);
        put(&mut walled, 0, 0, TYPE_GENERAL, OWNER_OPP, 5);
        put(&mut walled, 0, 1, TYPE_FOG, 0, 0);
        put(&mut walled, 1, 0, TYPE_FOG, 0, 0);
        put(&mut walled, 1, 1, TYPE_PLAIN, OWNER_ME, 40);
        total_armies(&mut walled);

        // Without the sighting the wall is invisible and the trigger fires.
        assert!(fire(&walled).is_some());

        // With it, the general is unreachable inside the budget.
        let mut seen = frame(99);
        put(&mut seen, 0, 1, TYPE_MOUNTAIN, 0, 0);
        put(&mut seen, 1, 0, TYPE_MOUNTAIN, 0, 0);
        let mem = remembering(&[&seen, &walled]);
        assert!(evaluate(&walled, &mem, &mut Reach::default()).is_none());
    }

    #[test]
    fn is_silent_pre_800_on_insufficient_army() {
        let mut obs = frame(799);
        put(&mut obs, 0, 0, TYPE_GENERAL, OWNER_OPP, 20);
        put(&mut obs, 0, 1, TYPE_PLAIN, OWNER_ME, 15);
        total_armies(&mut obs);
        assert!(fire(&obs).is_none());

        // The same position once deathtouch is live: the garrison stops
        // mattering.
        obs.turn = 800;
        assert_eq!(fire(&obs).expect("deathtouch kill").cause, Cause::Deathtouch);
    }

    #[test]
    fn needs_a_cell_that_can_move() {
        // Adjacent to a general on no army at all, but one army cannot move.
        let mut obs = frame(900);
        put(&mut obs, 0, 0, TYPE_GENERAL, OWNER_OPP, 0);
        put(&mut obs, 0, 1, TYPE_PLAIN, OWNER_ME, 1);
        total_armies(&mut obs);
        assert!(fire(&obs).is_none());
    }

    #[test]
    fn ages_its_bound_while_the_general_hides() {
        let mut seen = frame(100);
        put(&mut seen, 0, 0, TYPE_GENERAL, OWNER_OPP, 5);
        total_armies(&mut seen);

        let mut later = frame(112);
        put(&mut later, 0, 0, TYPE_FOG, 0, 0);
        put(&mut later, 0, 1, TYPE_PLAIN, OWNER_ME, 12);
        total_armies(&mut later);

        let mem = remembering(&[&seen, &later]);
        let kill = evaluate(&later, &mem, &mut Reach::default()).expect("kill");
        assert_eq!((kill.last_seen_army, kill.stale_turns), (5, 12));
    }
}
