//! The proof search: depth-limited minimax over *(our move, their reply)*
//! pairs, which returns an action only when the goal holds against **every**
//! reply at every ply.
//!
//! It is not a move chooser. Anything short of a proof is a decline, and a
//! decline is not a half-searched preference — it is silence, and the caller
//! keeps whatever the network wanted. That is what makes an override safe to
//! wire ahead of a network that is right far more often than this search will
//! ever fire.
//!
//! # Shape
//!
//! Moves are simultaneous, so one ply is a *pair*: we pick, they pick, and
//! [`Sim`] resolves both under the §02 ladder. The tree is therefore AND/OR —
//! an OR node over our actions, an AND node over theirs — and the recursion
//! is a plain "does some action of ours hold against all of theirs".
//!
//! Iterative deepening, because the shallowest proof is the one worth playing
//! and because a horizon we can finish beats a horizon we abandon. A search
//! that runs out of nodes or clock returns [`Report::capped`] and no move.
//!
//! # What the proof is over
//!
//! The model, not the world. [`Sim`] states its own approximations; two more
//! belong to this file:
//!
//! * **The reply window.** Their actions are generated from cells inside the
//!   window the caller passes — cells within reach of the goal. Replies from
//!   outside it are modeled as a pass. An action that far away cannot reach
//!   the general inside the horizon, and it commutes with ours because the two
//!   share no cell; the residue is a source just outside the window that walks
//!   in and interferes at a later ply, which this search does not see.
//! * **Their builds** are passes, per [`Sim`]'s docs.
//!
//! Neither hole can manufacture a win out of nothing — they can only make a
//! proof land that a wider search would have declined — so both are stated
//! here and in the plan rather than buried.

use std::time::Instant;

use super::sim::{Choice, Move, Outcome, Sim, ME, OPP};

/// Check the wall clock this often. Between checks the node counter is the
/// only brake, which is why both caps exist.
const CLOCK_CHECK: u64 = 1024;

/// What we are trying to prove.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Goal {
    /// Take their general inside the horizon. Reaching the horizon alive is a
    /// failure: an unfinished attack is not a kill.
    Kill,
    /// Keep ours. Reaching the horizon with the general still standing is the
    /// success — there is nothing further to prove.
    Survive,
}

/// The caps. Both exist because they fail differently: the clock is what the
/// turn budget cares about, and the node count is what makes a test
/// reproducible.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub depth: i32,
    pub nodes: u64,
    pub deadline: Instant,
}

/// What a search found, and what it cost.
#[derive(Debug, Clone, Copy)]
pub struct Report {
    /// The proven action: `None` when nothing was proven, `Some(None)` when
    /// the proven action is a pass.
    pub play: Option<Choice>,
    /// The depth the proof landed at, or the deepest one attempted.
    pub depth: i32,
    pub nodes: u64,
    /// Did a cap stop the search? A capped search proves nothing, and says so
    /// rather than reporting a refutation it never finished checking.
    pub capped: bool,
}

/// Prove `goal` from this position, or decline.
///
/// `focus` is the general the goal is about — theirs for a kill, ours for a
/// defense. It orders move generation and nothing else; ordering changes how
/// fast a proof is found, never whether one exists.
pub fn prove(sim: &mut Sim, goal: Goal, focus: usize, window: &[bool], limits: &Limits) -> Report {
    debug_assert!(sim.is_root(), "a search must be handed a board, not a line");
    let mut search = Search {
        goal,
        focus,
        window,
        node_budget: limits.nodes,
        deadline: limits.deadline,
        nodes: 0,
        capped: false,
        ours: vec![Vec::new(); limits.depth.max(1) as usize],
        theirs: vec![Vec::new(); limits.depth.max(1) as usize],
        top: 0,
    };

    let mut reached = 0;
    let mut play = None;
    for depth in 1..=limits.depth {
        search.top = depth;
        reached = depth;
        play = search.our_turn(sim, depth);
        if play.is_some() || search.capped {
            break;
        }
    }
    debug_assert!(sim.is_root(), "the search left a line on the board");
    // A capped search cannot also have proven something: `holds` refuses to
    // report success once a cap has tripped.
    Report { play, depth: reached, nodes: search.nodes, capped: search.capped }
}

/// Does some opponent reply take our general the moment after `candidate`?
///
/// Depth one, and over a **visible-only** [`Sim`]: every reply it enumerates
/// is one the opponent demonstrably has, so a `true` here is a fact about the
/// real board rather than a fact about the pessimistic model. That direction
/// is what a veto needs — it must never remove a candidate that is actually
/// safe. It is deliberately *not* the direction a defense proof needs, which
/// is why the two use different boards.
pub fn refutes(sim: &mut Sim, candidate: Choice, window: &[bool]) -> bool {
    debug_assert!(sim.is_root(), "a refutation is asked of a board, not a line");
    let focus = sim.own_general;
    let mut replies = Vec::new();
    their_moves(sim, window, focus, &mut replies);
    for reply in replies {
        sim.push(candidate, reply);
        let lost = matches!(sim.outcome(), Outcome::OppWins | Outcome::Draw);
        sim.pop();
        if lost {
            return true;
        }
    }
    false
}

struct Search<'a> {
    goal: Goal,
    focus: usize,
    window: &'a [bool],
    node_budget: u64,
    deadline: Instant,
    nodes: u64,
    capped: bool,
    /// One move buffer per ply, so a search that runs to its node budget does
    /// not also run the allocator.
    ours: Vec<Vec<Choice>>,
    theirs: Vec<Vec<Choice>>,
    top: i32,
}

impl Search<'_> {
    /// OR node: is there an action of ours that holds?
    fn our_turn(&mut self, sim: &mut Sim, depth: i32) -> Option<Choice> {
        let level = (self.top - depth) as usize;
        let mut actions = std::mem::take(&mut self.ours[level]);
        our_moves(sim, self.window, self.focus, &mut actions);

        let mut proven = None;
        for &action in actions.iter() {
            if self.holds(sim, action, depth, level) {
                proven = Some(action);
                break;
            }
            if self.capped {
                break;
            }
        }
        self.ours[level] = actions;
        proven
    }

    /// AND node: does `action` hold against every reply?
    fn holds(&mut self, sim: &mut Sim, action: Choice, depth: i32, level: usize) -> bool {
        let mut replies = std::mem::take(&mut self.theirs[level]);
        their_moves(sim, self.window, self.focus, &mut replies);

        let mut all = true;
        for &reply in replies.iter() {
            if self.spend() {
                all = false;
                break;
            }
            sim.push(action, reply);
            let held = match sim.outcome() {
                // Their general fell: that is the kill, and it is also the
                // safest our own general will ever be.
                Outcome::MeWins => true,
                Outcome::OppWins | Outcome::Draw => false,
                Outcome::Ongoing => {
                    if depth <= 1 {
                        self.goal == Goal::Survive
                    } else {
                        self.our_turn(sim, depth - 1).is_some()
                    }
                }
            };
            sim.pop();
            if !held {
                all = false;
                break;
            }
        }
        self.theirs[level] = replies;
        all && !self.capped
    }

    /// One node. `true` once a cap has stopped the search — and it stays
    /// `true`, so an exhausted search unwinds instead of half-proving.
    fn spend(&mut self) -> bool {
        if self.capped {
            return true;
        }
        self.nodes += 1;
        if self.nodes > self.node_budget {
            self.capped = true;
        } else if self.nodes % CLOCK_CHECK == 0 && Instant::now() >= self.deadline {
            self.capped = true;
        }
        self.capped
    }
}

/// Manhattan distance, for move ordering only. Ordering is a speed decision,
/// so it may use a metric the rules do not — nothing here decides what is
/// legal or what is proven.
fn manhattan(sim: &Sim, from: usize, to: usize) -> i32 {
    let (fr, fc) = ((from / sim.w) as i32, (from % sim.w) as i32);
    let (tr, tc) = ((to / sim.w) as i32, (to % sim.w) as i32);
    (fr - tr).abs() + (fc - tc).abs()
}

/// Every action a player has from cells inside the window: each movable cell,
/// four directions, both splits, plus the pass. `half` is skipped where it
/// sends the same army as a full move (an army of two sends one either way).
///
/// Ordered toward the general the goal is about, biggest stack first, with the
/// pass at whichever end `pass_first` asks for. A proof that exists is found
/// either way; ordering only decides how soon.
fn actions_for(
    sim: &Sim,
    player: u8,
    window: &[bool],
    focus: usize,
    pass_first: bool,
    out: &mut Vec<Choice>,
) {
    out.clear();
    out.push(None);
    for from in 0..sim.h * sim.w {
        if !window[from] || sim.owner_at(from) != player || sim.army_at(from) <= 1 {
            continue;
        }
        let splits: &[bool] = if sim.army_at(from) > 2 { &[false, true] } else { &[false] };
        for dir in 0..4u8 {
            let Some(dest) = sim.dest(from, dir) else {
                continue;
            };
            if !sim.passable_for(player, dest) {
                continue;
            }
            for &half in splits {
                out.push(Some(Move { from, dir, half }));
            }
        }
    }
    let pass_key = if pass_first { i32::MIN } else { i32::MAX };
    out.sort_unstable_by_key(|action| match action {
        Some(mv) => {
            let dest = sim.dest(mv.from, mv.dir).unwrap_or(mv.from);
            (manhattan(sim, dest, focus), -sim.army_at(mv.from))
        }
        None => (pass_key, 0),
    });
}

/// Passing is ours to consider too — standing still is sometimes the whole
/// defense, and waiting out a production tick is sometimes the attack. It goes
/// last: with two proofs in hand, the one that does something is the one to
/// play.
fn our_moves(sim: &Sim, window: &[bool], focus: usize, out: &mut Vec<Choice>) {
    actions_for(sim, ME, window, focus, false, out);
}

/// Their pass goes first: if our action does not hold even against a player
/// who does nothing, one node says so.
fn their_moves(sim: &Sim, window: &[bool], focus: usize, out: &mut Vec<Choice>) {
    actions_for(sim, OPP, window, focus, true, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::fixtures::{at, board, step, Setup};
    use std::time::Duration;

    const W: usize = 6;

    fn everywhere(sim: &Sim) -> Vec<bool> {
        vec![true; sim.h * sim.w]
    }

    fn limits(depth: i32) -> Limits {
        Limits {
            depth,
            nodes: 2_000_000,
            deadline: Instant::now() + Duration::from_secs(30),
        }
    }

    /// A kill at each depth in the budget: our stack starts 1, 2, then 3 moves
    /// from a general it outguns, with nothing on the board to stop it.
    ///
    /// Our own general sits far away on a single unit throughout these
    /// fixtures — one army cannot move (§02), so it stays out of the tree and
    /// the tests measure the line under test rather than the branching factor.
    #[test]
    fn finds_a_forced_kill_at_one_two_and_three_plies() {
        for distance in 1..=3usize {
            let mut sim = board(6, W, |b: &mut Setup| {
                b.enemy_general(0, 0, 3);
                b.mine(0, distance, 20);
                b.own_general(5, 5, 1);
            });
            sim.turn = 1;
            let window = everywhere(&sim);
            let target = at(W, 0, 0);
            let report = prove(&mut sim, Goal::Kill, target, &window, &limits(3));
            assert_eq!(report.depth, distance as i32, "{distance}: shortest proof");
            let play = report.play.expect("a kill inside the budget").expect("a move");
            assert_eq!(sim.dest(play.from, play.dir), Some(at(W, 0, distance - 1)));
        }
    }

    /// The same position with a garrison the stack cannot beat: no depth in
    /// the budget helps, and the search says nothing rather than something.
    #[test]
    fn declines_a_kill_that_fails_against_the_best_reply() {
        let mut sim = board(6, W, |b: &mut Setup| {
            b.enemy_general(0, 0, 50);
            b.mine(0, 1, 20);
            b.own_general(5, 5, 1);
        });
        sim.turn = 1;
        let window = everywhere(&sim);
        let report = prove(&mut sim, Goal::Kill, at(W, 0, 0), &window, &limits(3));
        assert!(report.play.is_none());
        assert!(!report.capped, "it finished the search rather than running out");
        assert_eq!(report.depth, 3, "and it finished every depth in the budget");
    }

    /// A reinforcement resolves before our attack (§02 rung 2), so a stack we
    /// outgun today is a stack we do not outgun at the moment we arrive. The
    /// pessimistic model puts exactly this cell wherever it cannot see, which
    /// is why a fogged neighbour defends a general for free.
    #[test]
    fn declines_when_a_hidden_stack_can_reinforce_the_general() {
        let position = |neighbour: i32| {
            let mut sim = board(6, W, |b: &mut Setup| {
                b.enemy_general(0, 0, 3);
                b.opp(1, 0, neighbour);
                b.mine(0, 1, 20);
                b.own_general(5, 5, 1);
            });
            sim.turn = 1;
            let window = everywhere(&sim);
            prove(&mut sim, Goal::Kill, at(W, 0, 0), &window, &limits(1)).play
        };
        // A single unit cannot move at all, so the general stands alone.
        assert!(position(1).is_some());
        // Nineteen of the twenty can join it before our nineteen arrive.
        assert!(position(20).is_none());
    }

    /// Deathtouch makes the garrison irrelevant — and the chase defense makes
    /// it relevant again. Their stack sits on the cell ours must move from, so
    /// every touch we start, they end.
    #[test]
    fn declines_a_touch_the_chase_defense_answers() {
        let mut sim = board(6, W, |b: &mut Setup| {
            b.enemy_general(0, 0, 400);
            b.opp(1, 1, 60);
            b.mine(0, 1, 5);
            b.own_general(5, 5, 1);
        });
        sim.turn = 800;
        let window = everywhere(&sim);
        assert!(prove(&mut sim, Goal::Kill, at(W, 0, 0), &window, &limits(1)).play.is_none());

        // Take the chaser out of the picture and the touch is proven.
        let mut open = board(6, W, |b: &mut Setup| {
            b.enemy_general(0, 0, 400);
            b.mine(0, 1, 5);
            b.own_general(5, 5, 1);
        });
        open.turn = 800;
        let window = everywhere(&open);
        assert!(prove(&mut open, Goal::Kill, at(W, 0, 0), &window, &limits(1)).play.is_some());
    }

    /// The saving defense, both shapes: take the attacker's source, or stand
    /// on the general with enough army to hold it.
    #[test]
    fn finds_the_saving_defense() {
        // Chase: their 12 is next to our general on 3, and our 14 sits on
        // their source's other side. Reinforcing is not enough (12 beats 3+13
        // ... it does not, so the fixture makes the chase the only answer by
        // putting our army out of reach of the general).
        let mut chase = board(6, W, |b: &mut Setup| {
            b.own_general(2, 2, 3);
            b.opp(2, 3, 12);
            b.mine(2, 4, 14);
        });
        chase.turn = 1;
        let window = everywhere(&chase);
        let play = prove(&mut chase, Goal::Survive, at(W, 2, 2), &window, &limits(1))
            .play
            .expect("a defense")
            .expect("a move");
        assert_eq!(play.from, at(W, 2, 4), "the chase, from the third tile");
        assert_eq!(chase.dest(play.from, play.dir), Some(at(W, 2, 3)));

        // Reinforce: the same threat with our army beside the general instead.
        let mut hold = board(6, W, |b: &mut Setup| {
            b.own_general(2, 2, 3);
            b.opp(2, 3, 12);
            b.mine(1, 2, 14);
        });
        hold.turn = 1;
        let window = everywhere(&hold);
        let play = prove(&mut hold, Goal::Survive, at(W, 2, 2), &window, &limits(1))
            .play
            .expect("a defense")
            .expect("a move");
        assert_eq!(hold.dest(play.from, play.dir), Some(at(W, 2, 2)), "onto the general");
    }

    /// Nothing saves a general two stacks can reach, and the search declines
    /// rather than picking the least bad move.
    #[test]
    fn declines_a_defense_that_does_not_exist() {
        let mut sim = board(6, W, |b: &mut Setup| {
            b.own_general(2, 2, 3);
            b.opp(2, 3, 40);
            b.opp(1, 2, 40);
            b.mine(3, 2, 6);
        });
        sim.turn = 1;
        let window = everywhere(&sim);
        assert!(prove(&mut sim, Goal::Survive, at(W, 2, 2), &window, &limits(1)).play.is_none());
    }

    /// A cap is a decline, not a refutation: the same position that proves a
    /// kill with a budget proves nothing without one, and reports why.
    #[test]
    fn a_node_cap_declines_and_says_so() {
        let mut sim = board(6, W, |b: &mut Setup| {
            b.enemy_general(0, 0, 3);
            b.opp(3, 3, 9);
            b.mine(0, 1, 20);
            b.own_general(5, 5, 5);
        });
        sim.turn = 1;
        let window = everywhere(&sim);
        let capped = Limits { nodes: 2, ..limits(3) };
        let report = prove(&mut sim, Goal::Kill, at(W, 0, 0), &window, &capped);
        assert!(report.play.is_none());
        assert!(report.capped);
        assert!(report.nodes <= 3);
    }

    /// The clock does the same job as the node cap, on the deadline instead.
    /// The position is one no kill exists in, so the search runs past the
    /// first clock check with a deadline already behind it.
    #[test]
    fn a_passed_deadline_declines() {
        let mut sim = board(6, W, |b: &mut Setup| {
            b.enemy_general(0, 0, 300);
            b.mine(0, 1, 40);
            b.mine(1, 1, 40);
            b.opp(1, 0, 40);
            b.own_general(5, 5, 1);
        });
        sim.turn = 1;
        let window = everywhere(&sim);
        let expired = Limits {
            depth: 3,
            nodes: 2_000_000,
            deadline: Instant::now() - Duration::from_secs(1),
        };
        let report = prove(&mut sim, Goal::Kill, at(W, 0, 0), &window, &expired);
        assert!(report.play.is_none());
        assert!(report.capped);
        assert!(report.nodes >= CLOCK_CHECK, "the clock is read every {CLOCK_CHECK} nodes");
    }

    /// A structure in fog is closed to us and open to them, so the same cell
    /// that blocks our attack carries their reply. Here it blocks: the way
    /// around is four moves and the budget is three.
    #[test]
    fn an_unresolved_structure_blocks_our_route() {
        let attack = |walled: bool| {
            let mut sim = board(3, W, |b: &mut Setup| {
                b.enemy_general(0, 0, 3);
                b.mine(0, 2, 20);
                b.own_general(2, 5, 1);
                if walled {
                    b.closed_to_us(0, 1);
                }
            });
            sim.turn = 1;
            let window = everywhere(&sim);
            prove(&mut sim, Goal::Kill, at(W, 0, 0), &window, &limits(3))
        };
        assert_eq!(attack(false).depth, 2, "straight through, two moves");
        assert!(attack(true).play.is_none());
    }

    /// The window is the caller's: a source outside it is not generated, and
    /// its threat is invisible to the search. Stated as a test because it is
    /// the search's sharpest edge.
    #[test]
    fn a_reply_outside_the_window_is_not_seen() {
        let mut sim = board(6, W, |b: &mut Setup| {
            b.own_general(2, 2, 3);
            b.opp(2, 3, 12);
            b.mine(2, 4, 14);
        });
        sim.turn = 1;
        let mut window = vec![false; sim.h * sim.w];
        for cell in [at(W, 2, 2), at(W, 2, 4)] {
            window[cell] = true;
        }
        // Their stack is outside the window, so the search believes any move
        // survives — including one that walks away from the general.
        let report = prove(&mut sim, Goal::Survive, at(W, 2, 2), &window, &limits(1));
        assert!(report.play.is_some());
    }

    /// `refutes` flags a candidate that hands over the general and passes one
    /// that does not.
    #[test]
    fn refutes_flags_the_move_that_loses_the_general() {
        let mut sim = board(6, W, |b: &mut Setup| {
            b.own_general(2, 2, 3);
            b.mine(2, 1, 12);
            b.opp(2, 3, 10);
        });
        sim.turn = 1;
        let window = everywhere(&sim);

        let away = Some(step(W, 2, 1, 2)); // (2,1) -> (2,0), leaving the general on 3
        assert!(refutes(&mut sim, away, &window));

        let onto = Some(step(W, 2, 1, 3)); // (2,1) -> (2,2), the general holds 14
        assert!(!refutes(&mut sim, onto, &window));

        // A pass is a candidate too, and this one loses.
        assert!(refutes(&mut sim, None, &window));
    }

    /// `refutes` is depth one by construction: a threat that needs two moves
    /// is not a refutation of anything.
    #[test]
    fn refutes_ignores_a_threat_that_is_still_two_moves_away() {
        let mut sim = board(6, W, |b: &mut Setup| {
            b.own_general(2, 2, 3);
            b.mine(2, 1, 12);
            b.opp(2, 4, 40);
        });
        sim.turn = 1;
        let window = everywhere(&sim);
        assert!(!refutes(&mut sim, Some(step(W, 2, 1, 2)), &window));
    }
}
