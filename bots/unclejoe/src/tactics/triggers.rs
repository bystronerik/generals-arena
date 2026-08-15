//! The two trigger predicates.
//!
//! A trigger is neither a decision nor a heuristic. It is a cheap, exact test
//! for "an exact search could have something to prove here", and it is
//! constructed as a **superset**: every position where a kill or a saving
//! defense is provable within the depth budget fires the matching trigger.
//! The converse is not claimed and is not wanted. A fire on a dead position
//! costs one search that then declines; a miss costs the whole point of the
//! bot. So every bound below is loose in that one safe direction.
//!
//! Both predicates are integer scans over the parsed frame plus
//! `board::memory`, O(cells) per turn, far below one forward pass.
//!
//! Range is measured in **moves**, not Manhattan steps: a bounded BFS out of
//! the general over every cell not remembered as a mountain. That is still a
//! superset — army travels one orthogonal step per turn and mountains are the
//! only permanently impassable cells (RULES.md §01), so nothing that could
//! reach the general in `SEARCH_DEPTH` moves is ever excluded — and it is
//! tighter than raw Manhattan distance, which counts armies that a wall
//! stands between.

use crate::board::memory::{is_visible, Memory};
use crate::board::obs::DIRECTIONS;
use crate::io::wire::{Observation, OWNER_ME, OWNER_OPP, TYPE_GENERAL};
use crate::tactics::{DEATHTOUCH_TURN, SEARCH_DEPTH};

/// Distance for a cell the BFS never reached: walled off, or further than the
/// depth budget.
const UNREACHED: i32 = i32::MAX;

/// Why the kill trigger fired. `ArmyBound` when the army we can bring beats
/// the general's last-seen garrison; `Deathtouch` when the turn alone makes
/// any unit lethal (RULES.md §07).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillCause {
    ArmyBound,
    Deathtouch,
}

/// Why the defense trigger fired, worst first — the order they are reported
/// in when several apply on one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreatCause {
    /// A visible enemy stack in range that is at least as big as the garrison.
    VisibleStack,
    /// Past turn 800, a visible enemy stack in range that can move at all.
    Deathtouch,
    /// A cell in range we cannot see, and enough unaccounted enemy army to
    /// fill it.
    Fog,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KillFire {
    /// The enemy general's remembered cell.
    pub target: usize,
    pub cause: KillCause,
    /// Total army on our movable cells within reach of it.
    pub reach_army: i32,
    /// The stale lower bound that `reach_army` was compared against.
    pub last_seen_army: i32,
    /// Turns since that sighting; `0` while the general is in view.
    pub stale_turns: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefenseFire {
    /// Our general's cell.
    pub general: usize,
    pub cause: ThreatCause,
    /// The cell that raised it.
    pub source: usize,
    /// The army credited to that cell: its own if visible, the whole hidden
    /// budget if not.
    pub threat_army: i32,
    pub general_army: i32,
}

/// What one frame's trigger pass found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Fired {
    pub kill: Option<KillFire>,
    pub defense: Option<DefenseFire>,
}

/// The trigger pass and its scratch. Held across turns so the per-move path
/// never grows the heap.
#[derive(Debug, Default)]
pub struct Triggers {
    dist: Vec<i32>,
    queue: Vec<usize>,
}

impl Triggers {
    /// Evaluate both triggers against one frame. Reads memory; changes
    /// nothing but its own scratch.
    pub fn evaluate(&mut self, obs: &Observation, mem: &Memory) -> Fired {
        Fired {
            kill: self.kill(obs, mem),
            defense: self.defense(obs, mem),
        }
    }

    /// **Kill.** The enemy general has been seen, some cell of ours that can
    /// move at all is within `SEARCH_DEPTH` moves of it, and either
    /// deathtouch is live or the army we could bring beats the general's
    /// last-seen garrison.
    ///
    /// Both quantities are deliberately generous. `reach_army` sums whole
    /// cells rather than what they could actually send (a mover leaves one
    /// behind, RULES.md §02) and ignores that only one of them moves per
    /// turn; `last_seen_army` is a lower bound on what defends today. Each
    /// error is in the fire-more direction.
    fn kill(&mut self, obs: &Observation, mem: &Memory) -> Option<KillFire> {
        let target = mem.enemy_general?;
        self.reach_from(obs, mem, target);

        let mut reach_army = 0;
        for cell in 0..obs.h * obs.w {
            // A cell with one army cannot move (RULES.md §02), so it is not a
            // source of anything and does not count.
            if self.dist[cell] <= SEARCH_DEPTH
                && obs.owner_grid[cell] == OWNER_ME
                && obs.army_grid[cell] > 1
            {
                reach_army += obs.army_grid[cell];
            }
        }
        // Armies are non-negative, so a zero total means no movable cell of
        // ours is in range at all.
        if reach_army == 0 {
            return None;
        }

        let cause = if reach_army > mem.last_seen_general_army {
            // Strictly more army takes the cell (RULES.md §05).
            KillCause::ArmyBound
        } else if obs.turn >= DEATHTOUCH_TURN {
            // From turn 800 the garrison stops mattering: one unit that
            // executes onto the tile wins (RULES.md §07).
            KillCause::Deathtouch
        } else {
            return None;
        };
        Some(KillFire {
            target,
            cause,
            reach_army,
            last_seen_army: mem.last_seen_general_army,
            // `last_seen_turn` is set with `enemy_general`, so it is never the
            // pre-sighting sentinel here.
            stale_turns: obs.turn - mem.last_seen_turn,
        })
    }

    /// **Immediate defense.** A cell within `SEARCH_DEPTH` moves of our own
    /// general that could plausibly take it: a visible enemy stack at least
    /// as large as the garrison, any visible enemy stack once deathtouch is
    /// live, or a cell we cannot see while the opponent has enough
    /// unaccounted army to have filled it.
    ///
    /// The fog arm is what makes this more than a look at the frame. Vision
    /// is a 3×3 pool around owned cells (RULES.md §06), so the ring adjacent
    /// to our general is always lit and the fog arm only ever fires at
    /// distance 2 or more — which is exactly where a stack can be sitting one
    /// step outside our sight.
    ///
    /// **Nothing fires before first contact.** With no enemy cell ever seen,
    /// `hidden` is the opponent's whole army and the fog arm reduces to "is
    /// there an unseen cell within `D`", which is true on almost every early
    /// turn: it fired on turn 0 of every game measured, and on 68% of all
    /// defense fires in the corpus replay, before the opponent was ever in
    /// sight. Through turn 13 that is *provably* empty — generals spawn ≥17
    /// BFS steps apart and army moves one step per turn, so no enemy cell can
    /// be within `D` of ours yet. After that it is an **assumption** rather
    /// than a proof, and the only one in this file: a stack that reached our
    /// neighbourhood without ever crossing our vision is possible under the
    /// pessimistic fog model, and we take it as not worth searching for.
    fn defense(&mut self, obs: &Observation, mem: &Memory) -> Option<DefenseFire> {
        mem.first_contact_turn?;
        let general = own_general(obs)?;
        let general_army = obs.army_grid[general];
        let hidden = hidden_army(obs);
        self.reach_from(obs, mem, general);

        let mut worst: Option<DefenseFire> = None;
        for cell in 0..obs.h * obs.w {
            if cell == general || self.dist[cell] > SEARCH_DEPTH {
                continue;
            }
            let visible = is_visible(obs.type_grid[cell]);
            let threat = if visible && obs.owner_grid[cell] == OWNER_OPP {
                let army = obs.army_grid[cell];
                if army >= general_army {
                    Some((ThreatCause::VisibleStack, army))
                } else if obs.turn >= DEATHTOUCH_TURN && army > 1 {
                    Some((ThreatCause::Deathtouch, army))
                } else {
                    None
                }
            } else if !visible && hidden >= general_army {
                Some((ThreatCause::Fog, hidden))
            } else {
                None
            };
            if let Some((cause, threat_army)) = threat {
                if worst.is_none_or(|w| severity(cause) > severity(w.cause)) {
                    worst = Some(DefenseFire {
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

    /// Moves from `origin` to every cell, over everything not remembered as a
    /// mountain, capped at `SEARCH_DEPTH`. Cells past the cap or walled off
    /// keep `UNREACHED`.
    ///
    /// Only *remembered* mountains block. Fog does not: a cell we cannot see
    /// may well be walkable, and pretending otherwise would drop real threats.
    fn reach_from(&mut self, obs: &Observation, mem: &Memory, origin: usize) {
        let (h, w) = (obs.h, obs.w);
        self.dist.clear();
        self.dist.resize(h * w, UNREACHED);
        self.queue.clear();
        self.dist[origin] = 0;
        self.queue.push(origin);

        let mut head = 0;
        while head < self.queue.len() {
            let cell = self.queue[head];
            head += 1;
            let steps = self.dist[cell];
            if steps == SEARCH_DEPTH {
                continue;
            }
            let (row, col) = ((cell / w) as i32, (cell % w) as i32);
            for (dr, dc) in DIRECTIONS {
                let (nr, nc) = (row + dr, col + dc);
                if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                    continue;
                }
                let next = nr as usize * w + nc as usize;
                if mem.mountains[next] || self.dist[next] != UNREACHED {
                    continue;
                }
                self.dist[next] = steps + 1;
                self.queue.push(next);
            }
        }
    }
}

/// Report order when one frame raises several threats: a stack we can see
/// beats one the clock made lethal, which beats a guess about fog.
fn severity(cause: ThreatCause) -> u8 {
    match cause {
        ThreatCause::VisibleStack => 2,
        ThreatCause::Deathtouch => 1,
        ThreatCause::Fog => 0,
    }
}

/// Our own general's cell. Always present while we are alive — we own it, and
/// vision is a 3×3 pool around owned cells — so `None` means a frame we
/// cannot defend anyway.
fn own_general(obs: &Observation) -> Option<usize> {
    (0..obs.h * obs.w)
        .find(|&i| obs.type_grid[i] == TYPE_GENERAL && obs.owner_grid[i] == OWNER_ME)
}

/// Enemy army we cannot account for: the frame states the opponent's total
/// exactly, so everything not sitting on a visible enemy cell is somewhere in
/// the fog. A sound bound on any one hidden cell, since it bounds all of them
/// together.
fn hidden_army(obs: &Observation) -> i32 {
    let visible: i32 = (0..obs.h * obs.w)
        .filter(|&i| obs.owner_grid[i] == OWNER_OPP)
        .map(|i| obs.army_grid[i])
        .sum();
    (obs.opp_army - visible).max(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::wire::{TYPE_FOG, TYPE_MOUNTAIN, TYPE_PLAIN, TYPE_STRUCTURE_IN_FOG};

    const H: usize = 9;
    const W: usize = 9;

    fn at(row: usize, col: usize) -> usize {
        row * W + col
    }

    /// An all-plain, all-neutral, all-visible board with our general at (4, 4)
    /// on 10 army, and no enemy anywhere.
    fn frame(turn: i32) -> Observation {
        let mut obs = Observation::with_dims(H, W);
        obs.turn = turn;
        obs.type_grid.iter_mut().for_each(|t| *t = TYPE_PLAIN);
        put(&mut obs, 4, 4, TYPE_GENERAL, OWNER_ME, 10);
        obs
    }

    fn put(obs: &mut Observation, row: usize, col: usize, cell_type: i32, owner: i32, army: i32) {
        let i = at(row, col);
        obs.type_grid[i] = cell_type;
        obs.owner_grid[i] = owner;
        obs.army_grid[i] = army;
    }

    /// Fill the scalar row the way the engine does: our total, and the
    /// opponent's total with nothing hidden.
    fn total_armies(obs: &mut Observation) {
        obs.my_army = (0..H * W)
            .filter(|&i| obs.owner_grid[i] == OWNER_ME)
            .map(|i| obs.army_grid[i])
            .sum();
        obs.opp_army = (0..H * W)
            .filter(|&i| obs.owner_grid[i] == OWNER_OPP)
            .map(|i| obs.army_grid[i])
            .sum();
    }

    fn fire(obs: &Observation) -> Fired {
        let mut mem = Memory::new(H, W);
        mem.update(obs);
        Triggers::default().evaluate(obs, &mem)
    }

    // ---- kill -----------------------------------------------------------

    #[test]
    fn kill_fires_at_every_depth_in_budget() {
        for depth in 1..=SEARCH_DEPTH as usize {
            let mut obs = frame(100);
            put(&mut obs, 0, 0, TYPE_GENERAL, OWNER_OPP, 5);
            put(&mut obs, 0, depth, TYPE_PLAIN, OWNER_ME, 12);
            total_armies(&mut obs);
            let kill = fire(&obs).kill.expect("a kill inside the budget");
            assert_eq!(kill.cause, KillCause::ArmyBound);
            assert_eq!(kill.target, at(0, 0));
            assert_eq!((kill.reach_army, kill.last_seen_army), (12, 5));
            assert_eq!(kill.stale_turns, 0);
        }
    }

    #[test]
    fn kill_is_silent_past_the_depth_budget() {
        let mut obs = frame(100);
        put(&mut obs, 0, 0, TYPE_GENERAL, OWNER_OPP, 5);
        put(&mut obs, 0, SEARCH_DEPTH as usize + 1, TYPE_PLAIN, OWNER_ME, 99);
        total_armies(&mut obs);
        assert!(fire(&obs).kill.is_none());
    }

    #[test]
    fn kill_is_silent_before_the_general_is_seen() {
        // The general is there, in fog. Memory never learned it, so there is
        // nothing to prove a kill against.
        let mut obs = frame(100);
        put(&mut obs, 0, 0, TYPE_FOG, 0, 0);
        put(&mut obs, 0, 1, TYPE_PLAIN, OWNER_ME, 99);
        total_armies(&mut obs);
        assert!(fire(&obs).kill.is_none());
    }

    #[test]
    fn kill_is_silent_behind_remembered_mountains() {
        // The general's only two neighbours are mountains, and on this frame
        // both are back in fog: only memory can tell they are walls.
        let mut walled = frame(100);
        put(&mut walled, 0, 0, TYPE_GENERAL, OWNER_OPP, 5);
        put(&mut walled, 0, 1, TYPE_FOG, 0, 0);
        put(&mut walled, 1, 0, TYPE_FOG, 0, 0);
        put(&mut walled, 1, 1, TYPE_PLAIN, OWNER_ME, 40);
        total_armies(&mut walled);

        // Without the sighting the wall is invisible and the trigger fires.
        let mut blind = Memory::new(H, W);
        blind.update(&walled);
        assert!(Triggers::default().evaluate(&walled, &blind).kill.is_some());

        // With it, the general is unreachable inside the budget.
        let mut seen = frame(99);
        put(&mut seen, 0, 1, TYPE_MOUNTAIN, 0, 0);
        put(&mut seen, 1, 0, TYPE_MOUNTAIN, 0, 0);
        let mut remembers = Memory::new(H, W);
        remembers.update(&seen);
        remembers.update(&walled);
        assert!(Triggers::default().evaluate(&walled, &remembers).kill.is_none());
    }

    #[test]
    fn kill_is_silent_pre_800_on_insufficient_army() {
        let mut obs = frame(799);
        put(&mut obs, 0, 0, TYPE_GENERAL, OWNER_OPP, 20);
        put(&mut obs, 0, 1, TYPE_PLAIN, OWNER_ME, 15);
        total_armies(&mut obs);
        assert!(fire(&obs).kill.is_none());

        // The same position once deathtouch is live: the garrison stops
        // mattering.
        obs.turn = 800;
        let kill = fire(&obs).kill.expect("deathtouch kill");
        assert_eq!(kill.cause, KillCause::Deathtouch);
    }

    #[test]
    fn kill_needs_a_cell_that_can_move() {
        // Adjacent to a general on no army at all, but one army cannot move.
        let mut obs = frame(900);
        put(&mut obs, 0, 0, TYPE_GENERAL, OWNER_OPP, 0);
        put(&mut obs, 0, 1, TYPE_PLAIN, OWNER_ME, 1);
        total_armies(&mut obs);
        assert!(fire(&obs).kill.is_none());
    }

    #[test]
    fn kill_ages_its_bound_while_the_general_hides() {
        let mut seen = frame(100);
        put(&mut seen, 0, 0, TYPE_GENERAL, OWNER_OPP, 5);
        total_armies(&mut seen);
        let mut mem = Memory::new(H, W);
        mem.update(&seen);

        let mut later = frame(112);
        put(&mut later, 0, 0, TYPE_FOG, 0, 0);
        put(&mut later, 0, 1, TYPE_PLAIN, OWNER_ME, 12);
        total_armies(&mut later);
        mem.update(&later);

        let kill = Triggers::default().evaluate(&later, &mem).kill.expect("kill");
        assert_eq!((kill.last_seen_army, kill.stale_turns), (5, 12));
    }

    // ---- defense --------------------------------------------------------

    #[test]
    fn defense_fires_on_a_visible_stack_at_every_depth_in_budget() {
        for depth in 1..=SEARCH_DEPTH as usize {
            let mut obs = frame(100);
            put(&mut obs, 4, 4 + depth, TYPE_PLAIN, OWNER_OPP, 10);
            total_armies(&mut obs);
            let threat = fire(&obs).defense.expect("a threat inside the budget");
            assert_eq!(threat.cause, ThreatCause::VisibleStack);
            assert_eq!(threat.source, at(4, 4 + depth));
            assert_eq!((threat.threat_army, threat.general_army), (10, 10));
        }
    }

    #[test]
    fn defense_is_silent_on_a_stack_too_small_or_too_far() {
        let mut small = frame(100);
        put(&mut small, 4, 5, TYPE_PLAIN, OWNER_OPP, 9);
        total_armies(&mut small);
        assert!(fire(&small).defense.is_none());

        let mut far = frame(100);
        put(&mut far, 4, 4 + SEARCH_DEPTH as usize + 1, TYPE_PLAIN, OWNER_OPP, 50);
        total_armies(&mut far);
        assert!(fire(&far).defense.is_none());
    }

    #[test]
    fn defense_is_silent_behind_remembered_mountains() {
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
        let mut blind = Memory::new(H, W);
        blind.update(&walled);
        let threat = Triggers::default()
            .evaluate(&walled, &blind)
            .defense
            .expect("a threat through the gap");
        assert_eq!(threat.cause, ThreatCause::VisibleStack);

        // With it, the way around is seven moves — past the budget.
        let mut seen = frame(99);
        for (row, col) in [(3, 5), (4, 5), (5, 5)] {
            put(&mut seen, row, col, TYPE_MOUNTAIN, 0, 0);
        }
        let mut remembers = Memory::new(H, W);
        remembers.update(&seen);
        remembers.update(&walled);
        assert!(Triggers::default().evaluate(&walled, &remembers).defense.is_none());
    }

    #[test]
    fn defense_deathtouch_arm_needs_the_clock() {
        let mut obs = frame(799);
        put(&mut obs, 4, 6, TYPE_PLAIN, OWNER_OPP, 2);
        total_armies(&mut obs);
        assert!(fire(&obs).defense.is_none());

        obs.turn = 800;
        let threat = fire(&obs).defense.expect("deathtouch threat");
        assert_eq!(threat.cause, ThreatCause::Deathtouch);
        assert_eq!(threat.threat_army, 2);
    }

    #[test]
    fn defense_fog_arm_fires_exactly_at_the_threshold() {
        // One fogged cell two steps out, one visible enemy cell far away, and
        // an opponent total that leaves a chosen amount unaccounted for.
        let mut obs = frame(100);
        put(&mut obs, 4, 6, TYPE_FOG, 0, 0);
        put(&mut obs, 0, 8, TYPE_PLAIN, OWNER_OPP, 5);
        total_armies(&mut obs);

        obs.opp_army = 5 + 9; // hidden 9 against a garrison of 10
        assert!(fire(&obs).defense.is_none());

        obs.opp_army = 5 + 10; // hidden 10: a stack that size could be there
        let threat = fire(&obs).defense.expect("fog threat");
        assert_eq!(threat.cause, ThreatCause::Fog);
        assert_eq!(threat.source, at(4, 6));
        assert_eq!(threat.threat_army, 10);
    }

    #[test]
    fn defense_fog_arm_ignores_fog_out_of_range() {
        let mut obs = frame(100);
        put(&mut obs, 4, 4 + SEARCH_DEPTH as usize + 1, TYPE_FOG, 0, 0);
        put(&mut obs, 0, 8, TYPE_PLAIN, OWNER_OPP, 5); // contact, far away
        total_armies(&mut obs);
        obs.opp_army = 500;
        assert!(fire(&obs).defense.is_none());
    }

    #[test]
    fn defense_is_silent_until_we_meet_the_enemy() {
        // A fogged cell two steps from our general and a large hidden budget:
        // the fog arm's whole case, and it must not fire while the opponent
        // has never been seen.
        let mut unmet = frame(100);
        put(&mut unmet, 4, 6, TYPE_FOG, 0, 0);
        total_armies(&mut unmet);
        unmet.opp_army = 200;
        assert!(fire(&unmet).defense.is_none());

        // One sighting anywhere on the board — far from our general, and gone
        // again by the frame we evaluate — is enough to arm it.
        let mut contact = frame(99);
        put(&mut contact, 0, 8, TYPE_PLAIN, OWNER_OPP, 5);
        total_armies(&mut contact);
        let mut mem = Memory::new(H, W);
        mem.update(&contact);
        mem.update(&unmet);
        let threat = Triggers::default()
            .evaluate(&unmet, &mem)
            .defense
            .expect("armed by first contact");
        assert_eq!(threat.cause, ThreatCause::Fog);
    }

    #[test]
    fn defense_reports_the_worst_of_several_threats() {
        // The fog cell comes first in scan order; the stack we can actually
        // see is the one worth reporting.
        let mut obs = frame(100);
        put(&mut obs, 2, 4, TYPE_FOG, 0, 0);
        put(&mut obs, 4, 6, TYPE_PLAIN, OWNER_OPP, 11);
        total_armies(&mut obs);
        obs.opp_army = 11 + 40;
        let threat = fire(&obs).defense.expect("threat");
        assert_eq!(threat.cause, ThreatCause::VisibleStack);
        assert_eq!(threat.source, at(4, 6));
    }
}
