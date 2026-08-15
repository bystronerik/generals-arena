//! The defense tactic: can they take ours?
//!
//! Three parts, in the order they run: the trigger, [`argmax_loses`], and
//! [`prove`]. The middle one is what keeps this tactic honest. A defense
//! search asked "is there a move that survives?" answers *yes* on almost every
//! quiet turn, because almost every move survives a turn where nothing is
//! happening — and an override on that answer would replace the network's move
//! with an arbitrary safe one. So the override needs two facts, not one: the
//! move the network chose **provably** loses the general, and another move
//! **provably** does not. Neither alone moves anything.
//!
//! Three trigger arms, and the third is what makes this more than a look at the
//! frame. Vision is a 3×3 pool around owned cells (RULES.md §06), so the ring
//! adjacent to our general is always lit and the fog arm only ever fires at
//! distance 2 or more — exactly where a stack can be sitting one step outside
//! our sight.
//!
//! **Nothing fires before first contact.** With no enemy cell ever seen,
//! `hidden_army` is the opponent's whole army and the fog arm reduces to "is
//! there an unseen cell within `D`", which is true on almost every early
//! turn: it fired on turn 0 of every game measured, and on 68% of all defense
//! fires in the corpus replay, before the opponent was ever in sight. Through
//! turn 13 that is *provably* empty — generals spawn ≥17 BFS steps apart and
//! army moves one step per turn, so no enemy cell can be within `D` of ours
//! yet. After that the gate is an **assumption** rather than a bound, and the
//! only one in the tactics layer: a stack that reached our neighbourhood
//! without ever crossing our vision is possible under the pessimistic fog
//! model, and we take it as not worth searching for.

use crate::board::memory::{is_visible, Memory};
use crate::io::wire::{Observation, OWNER_OPP};
use crate::search::minimax::{self, Goal, Limits, Report};
use crate::search::sim::{Choice, Sim};
use crate::tactics::common::{hidden_army, Reach};
use crate::tactics::{DEATHTOUCH_TURN, DEFENSE_DEPTH, SEARCH_DEPTH};

/// Why the trigger fired, worst first — the order they are reported in when
/// several apply on one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cause {
    /// A visible enemy stack in range that is at least as big as the garrison.
    VisibleStack,
    /// Past turn 800, a visible enemy stack in range that can move at all.
    Deathtouch,
    /// A cell in range we cannot see, and enough unaccounted enemy army to
    /// fill it.
    Fog,
}

impl Cause {
    pub fn name(self) -> &'static str {
        match self {
            Cause::VisibleStack => "stack",
            Cause::Deathtouch => "deathtouch",
            Cause::Fog => "fog",
        }
    }

    /// Report order: a stack we can see beats one the clock made lethal,
    /// which beats a guess about fog.
    fn severity(self) -> u8 {
        match self {
            Cause::VisibleStack => 2,
            Cause::Deathtouch => 1,
            Cause::Fog => 0,
        }
    }
}

/// One defense trigger fire, with the numbers it fired on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fire {
    /// Our general's cell.
    pub general: usize,
    pub cause: Cause,
    /// The cell that raised it.
    pub source: usize,
    /// The army credited to that cell: its own if visible, the whole hidden
    /// budget if not.
    pub threat_army: i32,
    pub general_army: i32,
}

/// Fire on a cell within `SEARCH_DEPTH` moves of our general that could
/// plausibly take it — but never before we have met the opponent.
pub fn evaluate(obs: &Observation, mem: &Memory, reach: &mut Reach) -> Option<Fire> {
    mem.first_contact_turn?;
    let general = mem.own_general?;
    let general_army = obs.army_grid[general];
    let hidden = hidden_army(obs);
    reach.compute(obs, mem, general, SEARCH_DEPTH);

    let mut worst: Option<Fire> = None;
    for cell in 0..obs.h * obs.w {
        if cell == general || !reach.reached(cell) {
            continue;
        }
        let visible = is_visible(obs.type_grid[cell]);
        let threat = if visible && obs.owner_grid[cell] == OWNER_OPP {
            let army = obs.army_grid[cell];
            if army >= general_army {
                Some((Cause::VisibleStack, army))
            } else if obs.turn >= DEATHTOUCH_TURN && army > 1 {
                Some((Cause::Deathtouch, army))
            } else {
                None
            }
        } else if !visible && hidden >= general_army {
            Some((Cause::Fog, hidden))
        } else {
            None
        };
        if let Some((cause, threat_army)) = threat {
            if worst.is_none_or(|w| cause.severity() > w.cause.severity()) {
                worst = Some(Fire {
                    general,
                    cause,
                    source: cell,
                    threat_army,
                    general_army,
                });
            }
        }
    }
    worst
}

/// Does the move the network chose hand over the general?
///
/// Over the **visible-only** board, so every reply counted here is one the
/// opponent demonstrably has: a `true` is a fact about the game, not about the
/// pessimistic model. Under fog the two boards agree anyway at this horizon —
/// vision is the 3×3 pool around owned cells (RULES.md §06), so every cell
/// that could reach our general in one move is lit.
pub fn argmax_loses(argmax: Choice, visible: &mut Sim, window: &[bool]) -> bool {
    minimax::refutes(visible, argmax, window)
}

/// Try to prove that some action keeps the general.
///
/// Depth one, where a defense proof means something. The horizon is not a
/// budget decision: at one ply the pessimistic board costs us nothing, because
/// every cell within a move of our general is visible and the fog bound has
/// nowhere to sit. At two it costs us everything — a fogged cell two steps out
/// holds the opponent's whole unaccounted army, so almost nothing is provable
/// and the search would spend the clock proving it. The trigger still scans
/// `SEARCH_DEPTH` moves out, which is a superset of what this can answer; the
/// extra fires cost a [`argmax_loses`] call that says no.
pub fn prove(fire: &Fire, sim: &mut Sim, window: &[bool], limits: &Limits) -> Report {
    let limits = Limits { depth: DEFENSE_DEPTH, ..*limits };
    minimax::prove(sim, Goal::Survive, fire.general, window, &limits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::wire::{TYPE_FOG, TYPE_MOUNTAIN, TYPE_PLAIN, TYPE_STRUCTURE_IN_FOG};
    use crate::tactics::common::fixtures::{at, fire_with, frame, put, remembering, total_armies};

    fn fire(obs: &Observation) -> Option<Fire> {
        fire_with(obs, evaluate)
    }

    #[test]
    fn fires_on_a_visible_stack_at_every_depth_in_budget() {
        for depth in 1..=SEARCH_DEPTH as usize {
            let mut obs = frame(100);
            put(&mut obs, 4, 4 + depth, TYPE_PLAIN, OWNER_OPP, 10);
            total_armies(&mut obs);
            let threat = fire(&obs).expect("a threat inside the budget");
            assert_eq!(threat.cause, Cause::VisibleStack);
            assert_eq!(threat.source, at(4, 4 + depth));
            assert_eq!((threat.threat_army, threat.general_army), (10, 10));
        }
    }

    #[test]
    fn is_silent_on_a_stack_too_small_or_too_far() {
        let mut small = frame(100);
        put(&mut small, 4, 5, TYPE_PLAIN, OWNER_OPP, 9);
        total_armies(&mut small);
        assert!(fire(&small).is_none());

        let mut far = frame(100);
        put(&mut far, 4, 4 + SEARCH_DEPTH as usize + 1, TYPE_PLAIN, OWNER_OPP, 50);
        total_armies(&mut far);
        assert!(fire(&far).is_none());
    }

    #[test]
    fn is_silent_behind_remembered_mountains() {
        // A big stack three steps out in a straight line, with a wall across
        // the line. On this frame the wall reads as structure-in-fog, which
        // could as well be a castle: only an earlier sighting says otherwise.
        let mut walled = frame(100);
        put(&mut walled, 4, 7, TYPE_PLAIN, OWNER_OPP, 50);
        for (row, col) in [(3, 5), (4, 5), (5, 5)] {
            put(&mut walled, row, col, TYPE_STRUCTURE_IN_FOG, 0, 0);
        }
        total_armies(&mut walled);

        // Without the sighting the wall might be passable, and the stack is
        // three moves away.
        assert_eq!(
            fire(&walled).expect("a threat through the gap").cause,
            Cause::VisibleStack
        );

        // With it, the way around is seven moves — past the budget.
        let mut seen = frame(99);
        for (row, col) in [(3, 5), (4, 5), (5, 5)] {
            put(&mut seen, row, col, TYPE_MOUNTAIN, 0, 0);
        }
        let mem = remembering(&[&seen, &walled]);
        assert!(evaluate(&walled, &mem, &mut Reach::default()).is_none());
    }

    #[test]
    fn deathtouch_arm_needs_the_clock() {
        let mut obs = frame(799);
        put(&mut obs, 4, 6, TYPE_PLAIN, OWNER_OPP, 2);
        total_armies(&mut obs);
        assert!(fire(&obs).is_none());

        obs.turn = 800;
        let threat = fire(&obs).expect("deathtouch threat");
        assert_eq!(threat.cause, Cause::Deathtouch);
        assert_eq!(threat.threat_army, 2);
    }

    #[test]
    fn fog_arm_fires_exactly_at_the_threshold() {
        // One fogged cell two steps out, one visible enemy cell far away, and
        // an opponent total that leaves a chosen amount unaccounted for.
        let mut obs = frame(100);
        put(&mut obs, 4, 6, TYPE_FOG, 0, 0);
        put(&mut obs, 0, 8, TYPE_PLAIN, OWNER_OPP, 5);
        total_armies(&mut obs);

        obs.opp_army = 5 + 9; // hidden 9 against a garrison of 10
        assert!(fire(&obs).is_none());

        obs.opp_army = 5 + 10; // hidden 10: a stack that size could be there
        let threat = fire(&obs).expect("fog threat");
        assert_eq!(threat.cause, Cause::Fog);
        assert_eq!(threat.source, at(4, 6));
        assert_eq!(threat.threat_army, 10);
    }

    #[test]
    fn fog_arm_ignores_fog_out_of_range() {
        let mut obs = frame(100);
        put(&mut obs, 4, 4 + SEARCH_DEPTH as usize + 1, TYPE_FOG, 0, 0);
        put(&mut obs, 0, 8, TYPE_PLAIN, OWNER_OPP, 5); // contact, far away
        total_armies(&mut obs);
        obs.opp_army = 500;
        assert!(fire(&obs).is_none());
    }

    #[test]
    fn is_silent_until_we_meet_the_enemy() {
        // A fogged cell two steps from our general and a large hidden budget:
        // the fog arm's whole case, and it must not fire while the opponent
        // has never been seen.
        let mut unmet = frame(100);
        put(&mut unmet, 4, 6, TYPE_FOG, 0, 0);
        total_armies(&mut unmet);
        unmet.opp_army = 200;
        assert!(fire(&unmet).is_none());

        // One sighting anywhere on the board — far from our general, and gone
        // again by the frame we evaluate — is enough to arm it.
        let mut contact = frame(99);
        put(&mut contact, 0, 8, TYPE_PLAIN, OWNER_OPP, 5);
        total_armies(&mut contact);
        let mem = remembering(&[&contact, &unmet]);
        let threat = evaluate(&unmet, &mem, &mut Reach::default()).expect("armed by contact");
        assert_eq!(threat.cause, Cause::Fog);
    }

    #[test]
    fn reports_the_worst_of_several_threats() {
        // The fog cell comes first in scan order; the stack we can actually
        // see is the one worth reporting.
        let mut obs = frame(100);
        put(&mut obs, 2, 4, TYPE_FOG, 0, 0);
        put(&mut obs, 4, 6, TYPE_PLAIN, OWNER_OPP, 11);
        total_armies(&mut obs);
        obs.opp_army = 11 + 40;
        let threat = fire(&obs).expect("threat");
        assert_eq!(threat.cause, Cause::VisibleStack);
        assert_eq!(threat.source, at(4, 6));
    }
}
