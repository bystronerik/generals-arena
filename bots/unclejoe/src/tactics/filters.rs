//! Candidate filters: exact arithmetic that **removes** candidates.
//!
//! A different kind of object from everything else in this layer. A trigger
//! says a search is worth running; a search proves an action; a filter does
//! neither — it reads a rule off RULES.md and says that one candidate is not
//! worth valuing. Filters never rank, never score, and never choose: what
//! survives them is still the policy's list in the policy's order.
//!
//! Three of them, in two families.
//!
//! **The castle masks** ([`castle`]) are arithmetic on §03's build price. They
//! are the layer's only *uncalibrated* numbers — [`CASTLE_SURCHARGE_CAP`] and
//! [`CASTLE_LATE_TURN`] are guesses about what a castle is worth, not facts
//! about the rules, and the plan names them as this milestone's risk. What is
//! exact is the price they read: it comes off the pipeline's own cost grid, so
//! the filter and the engine cannot disagree about what a build costs.
//!
//! **The refutation veto** ([`refutation_possible`] plus
//! [`minimax::refutes`](crate::search::minimax::refutes)) closes the gap the
//! U3 override left open: the defense proof declines, and the bot then plays a
//! network move that the search has already watched lose the general. The veto
//! is exact in the direction a veto needs — `refutes` runs on the
//! visible-only board, so every reply it counts is one the opponent really
//! has, and it can never remove a candidate that is actually safe.
//!
//! [`CASTLE_SURCHARGE_CAP`]: super::CASTLE_SURCHARGE_CAP
//! [`CASTLE_LATE_TURN`]: super::CASTLE_LATE_TURN

use crate::board::memory::{is_visible, Memory};
use crate::board::obs::{BUILD_BASE_COST, DIRECTIONS};
use crate::io::wire::{Observation, OWNER_OPP};

use super::{CASTLE_LATE_TURN, CASTLE_SURCHARGE_CAP};

/// Why a candidate was removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mask {
    /// A build too close to our own structures to be worth §03's surcharge.
    Crowding,
    /// A build too late in the game to pay itself back.
    Lateness,
    /// A candidate a visible reply answers by taking our general.
    Refuted,
}

impl Mask {
    pub fn name(self) -> &'static str {
        match self {
            Mask::Crowding => "crowding",
            Mask::Lateness => "late",
            Mask::Refuted => "refuted",
        }
    }
}

/// Both castle masks, crowding first.
///
/// `cost` is the live pipeline's build-cost grid, so this reads the price the
/// engine would charge rather than a second implementation of §03.
pub fn castle(obs: &Observation, cost: &[i32], cell: usize) -> Option<Mask> {
    crowding(cost, cell).or_else(|| lateness(obs.turn))
}

/// §03's crowding surcharge: 35 base plus `max(0, 14 − 2·d)` for every
/// structure of ours at Manhattan distance `d`. Subtracting the base leaves
/// the surcharge, and anything over the cap is a build paying for the company
/// it keeps.
///
/// At [`CASTLE_SURCHARGE_CAP`](super::CASTLE_SURCHARGE_CAP) `= 8` exactly one
/// structure at `d ≥ 3` passes (surcharge 8). Anything closer, or two
/// structures crowding at all, is masked.
pub fn crowding(cost: &[i32], cell: usize) -> Option<Mask> {
    let surcharge = cost[cell] - BUILD_BASE_COST;
    (surcharge > CASTLE_SURCHARGE_CAP).then_some(Mask::Crowding)
}

/// §03 with §04: a castle produces one army every second turn, so a build at
/// price `p` needs `2p` turns to return what it cost. Past
/// [`CASTLE_LATE_TURN`](super::CASTLE_LATE_TURN) the game will not last that
/// long, and the army is worth more where it stands.
pub fn lateness(turn: i32) -> Option<Mask> {
    (turn > CASTLE_LATE_TURN).then_some(Mask::Lateness)
}

/// Can a refutation exist at all on this frame?
///
/// Exact, and free. [`refutes`](crate::search::minimax::refutes) is depth one,
/// and the only reply that takes a general is a move that lands on it — so a
/// refutation needs an enemy cell orthogonally adjacent to ours, holding an
/// army that can move at all (§02: one unit stays home). Our own candidate
/// cannot create such a cell; at most it captures one. So a `false` here
/// means every candidate on the slate is safe from a one-ply answer, and the
/// veto can skip the whole board.
///
/// It is deliberately not the defense trigger. That one compares the threat
/// against the garrison *as it stands*; the veto's whole point is the
/// candidate that empties the garrison first.
pub fn refutation_possible(obs: &Observation, mem: &Memory) -> bool {
    let Some(general) = mem.own_general else {
        return false;
    };
    let (row, col) = ((general / obs.w) as i32, (general % obs.w) as i32);
    DIRECTIONS.iter().any(|(dr, dc)| {
        let (r, c) = (row + dr, col + dc);
        if r < 0 || c < 0 || r >= obs.h as i32 || c >= obs.w as i32 {
            return false;
        }
        let cell = r as usize * obs.w + c as usize;
        is_visible(obs.type_grid[cell])
            && obs.owner_grid[cell] == OWNER_OPP
            && obs.army_grid[cell] > 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::obs::{build_cost_from_raw, frame_to_raw};
    use crate::io::wire::{OWNER_ME, TYPE_CASTLE, TYPE_FOG, TYPE_GENERAL, TYPE_PLAIN};
    use crate::tactics::common::fixtures::{at, frame, put, remembering};

    /// The pipeline's own cost grid for a frame — the same call `Seat::act`
    /// makes, so the filter is tested against the prices the engine charges.
    fn prices(obs: &Observation) -> Vec<i32> {
        let mut raw = Vec::new();
        let mut cost = Vec::new();
        frame_to_raw(obs, &mut raw);
        build_cost_from_raw(&raw, obs.h, obs.w, &mut cost);
        cost
    }

    /// RULES.md §03's worked example: our general and a castle of ours both at
    /// distance 2 make the price 35 + 10 + 10 = 55, and a surcharge of 20 is
    /// well past the cap.
    #[test]
    fn the_rules_worked_example_is_masked() {
        // The fixture frame already carries our general at (4, 4).
        let mut obs = frame(100);
        put(&mut obs, 4, 0, TYPE_CASTLE, OWNER_ME, 8);
        let cost = prices(&obs);
        // (4, 2) sits two steps from the general and two from the castle.
        assert_eq!(cost[at(4, 2)], 55);
        assert_eq!(castle(&obs, &cost, at(4, 2)), Some(Mask::Crowding));
    }

    /// A build clear of our own structures pays the bare 35 and passes.
    #[test]
    fn a_build_in_open_ground_passes() {
        let obs = frame(100);
        let cost = prices(&obs);
        // Eight steps from the general: §03 says nothing is added past six.
        assert_eq!(cost[at(0, 0)], 35);
        assert_eq!(castle(&obs, &cost, at(0, 0)), None);
    }

    /// The cap sits exactly on one structure at distance three.
    #[test]
    fn the_cap_admits_one_structure_at_three_steps_and_no_closer() {
        let obs = frame(100);
        let cost = prices(&obs);
        assert_eq!(cost[at(4, 7)], 35 + 8, "three steps out");
        assert_eq!(crowding(&cost, at(4, 7)), None);
        assert_eq!(cost[at(4, 6)], 35 + 10, "two steps out");
        assert_eq!(crowding(&cost, at(4, 6)), Some(Mask::Crowding));
    }

    /// Enemy structures never raise our price, so they never mask our build
    /// either (§03: prices depend only on your own structures).
    #[test]
    fn an_enemy_castle_next_door_masks_nothing() {
        let mut obs = frame(100);
        put(&mut obs, 0, 1, TYPE_CASTLE, OWNER_OPP, 8);
        let cost = prices(&obs);
        // (0, 0) is eight steps from our own general and next door to theirs.
        assert_eq!(cost[at(0, 0)], 35);
        assert_eq!(castle(&obs, &cost, at(0, 0)), None);
    }

    /// The lateness threshold is the last turn a build passes.
    #[test]
    fn lateness_masks_from_the_turn_after_the_threshold() {
        assert_eq!(lateness(CASTLE_LATE_TURN), None);
        assert_eq!(lateness(CASTLE_LATE_TURN + 1), Some(Mask::Lateness));

        // And it applies to a build that would otherwise be fine.
        let mut late = frame(CASTLE_LATE_TURN + 1);
        let cost = prices(&late);
        assert_eq!(castle(&late, &cost, at(0, 0)), Some(Mask::Lateness));
        late.turn = CASTLE_LATE_TURN;
        assert_eq!(castle(&late, &cost, at(0, 0)), None);
    }

    /// The veto's gate: an enemy stack beside the general arms it, and nothing
    /// else does.
    #[test]
    fn a_refutation_needs_a_stack_beside_the_general() {
        let quiet = frame(100);
        assert!(!refutation_possible(&quiet, &remembering(&[&quiet])));

        // Two steps away is not adjacent, and `refutes` is one ply.
        let mut near = frame(100);
        put(&mut near, 4, 6, TYPE_PLAIN, OWNER_OPP, 40);
        assert!(!refutation_possible(&near, &remembering(&[&near])));

        let mut beside = frame(100);
        put(&mut beside, 4, 5, TYPE_PLAIN, OWNER_OPP, 2);
        assert!(refutation_possible(&beside, &remembering(&[&beside])));
    }

    /// One army cannot move (§02), so a neighbour holding one is not a threat.
    /// Neither is a neighbour we cannot see, which is why the gate reads the
    /// frame and not the pessimistic model.
    #[test]
    fn a_neighbour_that_cannot_move_or_cannot_be_seen_is_not_a_threat() {
        let mut still = frame(100);
        put(&mut still, 4, 5, TYPE_PLAIN, OWNER_OPP, 1);
        assert!(!refutation_possible(&still, &remembering(&[&still])));

        let mut dark = frame(100);
        put(&mut dark, 4, 5, TYPE_FOG, 0, 0);
        assert!(!refutation_possible(&dark, &remembering(&[&dark])));
    }

    /// Without a general there is nothing to refute, and the gate says so
    /// rather than indexing into a board it has no origin on.
    #[test]
    fn no_general_no_refutation() {
        let mut obs = frame(100);
        put(&mut obs, 4, 4, TYPE_PLAIN, 0, 0);
        put(&mut obs, 4, 5, TYPE_PLAIN, OWNER_OPP, 40);
        let mem = remembering(&[&obs]);
        assert_eq!(mem.own_general, None);
        assert!(!refutation_possible(&obs, &mem));

        // With the general back, the same stack arms it.
        let mut with = frame(100);
        put(&mut with, 4, 5, TYPE_PLAIN, OWNER_OPP, 40);
        assert_eq!(with.type_grid[at(4, 4)], TYPE_GENERAL);
        assert!(refutation_possible(&with, &remembering(&[&with])));
    }
}
