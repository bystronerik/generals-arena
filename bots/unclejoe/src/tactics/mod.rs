//! The tactics layer — everything unclejoe does that joe-rs does not.
//!
//! At milestone U3 the layer can **act**, and only ever on a proof. Every turn
//! it evaluates the two tactics' trigger predicates against the frame; on a
//! fire it runs the bounded exact search behind that trigger; and a search
//! that proves its goal replaces the network's move. Everything else — a
//! trigger that stays quiet, a search that declines, a cap that trips — leaves
//! the reply exactly where joe-rs would have left it.
//!
//! # Shape
//!
//! **One file per tactic.** [`kill`] and [`defense`] each own a predicate, the
//! result type it returns, the search call that predicate gates, and the tests
//! for all of it. Everything neither of them owns alone is in [`common`] —
//! distance in moves, and what a frame states about the army it does not show.
//! This file is the composition root: it owns the constants, the shared
//! scratch, the counters, and the order the two tactics run in. A tactic never
//! calls another tactic, and never reaches into another tactic's file for a
//! helper; they meet here and in [`common`], and nowhere else.
//!
//! A tactic's predicate is neither a decision nor a heuristic. It is a cheap,
//! exact test for "an exact search could have something to prove here", built
//! as a **superset**: every position where the goal is provable inside the
//! depth budget fires it, and the converse is not claimed. A fire on a dead
//! position costs one search that then declines; a miss costs the point of the
//! bot. Every bound is therefore loose in that one safe direction — with one
//! named exception, the first-contact gate in [`defense`].
//!
//! # Why the kill runs first
//!
//! Because a proven kill is strictly better than a proven defense. The search
//! counts a mutual capture as a draw and a draw as failure, so a kill it
//! proves is a kill that wins outright — there is no line where surviving
//! would have been worth more. The defense proof then runs against whatever
//! clock is left, which on a declined kill is most of it.
//!
//! The layer reads a `&Memory` and never writes one. `Seat::act` owns the
//! memory update, for the same reason it owns `AugState`'s: per-game state
//! advances every turn regardless of who chooses the move.

pub mod common;
pub mod defense;
pub mod kill;

use std::time::{Duration, Instant};

use crate::board::action::Action5;
use crate::board::memory::Memory;
use crate::io::wire::{Action, Observation, PASS};
use crate::search::minimax::{Limits, Report};
use crate::search::sim::{Choice, Config, Fog, Move, Sim};
use crate::tactics::common::{hidden_army, Reach};

// ---- constants ------------------------------------------------------------
//
// The tactics layer's numbers live here and nowhere else. The tactics plan and
// the strategy spec both quote them, so a change here is a change to two
// documents.

/// `D` — the proof search's depth budget in our own moves, and therefore the
/// radius the triggers scan. Beyond three plies the pessimistic fog bound
/// dominates and proofs stop landing: the binding constraint is knowledge,
/// not compute.
pub const SEARCH_DEPTH: i32 = 3;

/// The defense proof's own horizon, one ply, argued in [`defense::prove`]. The
/// kill uses the full `SEARCH_DEPTH`; defense cannot, because at two plies the
/// fog bound is adjacent to our general and nothing is provable.
pub const DEFENSE_DEPTH: i32 = 1;

/// The turn deathtouch goes live: from here a move that executes onto the
/// enemy general's tile wins whatever the garrison holds (RULES.md §07). The
/// engine's `deathtouch.py` compares `state.time >= 800`, and `obs.turn` is
/// that same clock.
pub const DEATHTOUCH_TURN: i32 = 800;

/// The whole turn, in ms: 20 under the 150 the judge allows (RULES.md §08),
/// which is the slack for host jitter and for emitting the reply. Nothing in
/// this layer may start work that could run past it.
pub const TURN_DEADLINE_MS: u64 = 130;

/// One proof search's wall clock, checked every 1,024 nodes. Two searches can
/// fire on one turn, and each gets its own slice of what is left rather than
/// sharing one — the second is the defense, and a declined kill must not
/// starve it.
pub const PROOF_DEADLINE_MS: u64 = 60;

/// The clock-independent ceiling. It exists so that a search costs the same
/// number of nodes on a fast host and a slow one, which is what makes a
/// decline reproducible.
pub const PROOF_NODE_BUDGET: u64 = 300_000;

/// What a fogged cell can gain while the search runs. A cell we cannot see may
/// be a castle or a general, producing one army every second turn (§04), and
/// one 50-turn tick can land inside the horizon as well — at most three over
/// three plies. Added to the hidden bound so an aged bound is still a bound.
pub const FOG_GROWTH_MARGIN: i32 = SEARCH_DEPTH;

/// The kill-switches, read once at construction.
///
/// `UNCLEJOE_TACTICS=0` is the master: pure pass-through, byte-identical
/// behavior to joe-rs, and what the wire-replay test runs under.
/// `UNCLEJOE_OVERRIDE=0` keeps the triggers and their reports and takes the
/// override away — the U2 shadow build, reachable from any later one.
#[derive(Debug, Clone, Copy)]
struct Switches {
    tactics: bool,
    overrides: bool,
}

impl Switches {
    fn from_env() -> Self {
        let on = |name: &str| !std::env::var(name).is_ok_and(|value| value == "0");
        Self { tactics: on("UNCLEJOE_TACTICS"), overrides: on("UNCLEJOE_OVERRIDE") }
    }
}

/// The layer's per-game state: the switches, the shared scratch, and the
/// counters.
#[derive(Debug)]
pub struct Tactics {
    switches: Switches,
    /// Which seat we are. The §02 ladder's last tiebreak is the seat index, so
    /// the forward model cannot resolve an exact clash without it.
    i_am_p0: bool,
    reach: Reach,
    /// The move-generation window, refilled per search from `reach`.
    window: Vec<bool>,
    turns: u32,
    /// Copied off memory so the summary can report it: the defense tactic is
    /// silent until this is set, so it is the number that explains a low
    /// defense count.
    first_contact_turn: Option<i32>,
    general_known: u32,
    general_visible: u32,
    kill: u32,
    kill_army: u32,
    kill_deathtouch: u32,
    kill_proved: u32,
    kill_capped: u32,
    defense: u32,
    defense_stack: u32,
    defense_deathtouch: u32,
    defense_fog: u32,
    defense_refuted: u32,
    defense_proved: u32,
    defense_capped: u32,
    overrides: u32,
    nodes: u64,
    slowest_proof_ms: f64,
}

impl Tactics {
    pub fn new(i_am_p0: bool) -> Self {
        let switches = Switches::from_env();
        if !switches.tactics {
            eprintln!("[unclejoe] tactics off: the reply is the network's, every turn");
        } else if !switches.overrides {
            eprintln!("[unclejoe] tactics in shadow: triggers report, nothing overrides");
        }
        Self {
            switches,
            i_am_p0,
            reach: Reach::default(),
            window: Vec::new(),
            turns: 0,
            first_contact_turn: None,
            general_known: 0,
            general_visible: 0,
            kill: 0,
            kill_army: 0,
            kill_deathtouch: 0,
            kill_proved: 0,
            kill_capped: 0,
            defense: 0,
            defense_stack: 0,
            defense_deathtouch: 0,
            defense_fog: 0,
            defense_refuted: 0,
            defense_proved: 0,
            defense_capped: 0,
            overrides: 0,
            nodes: 0,
            slowest_proof_ms: 0.0,
        }
    }

    /// One turn of the layer: run both tactics against the frame, and return
    /// the action a proof demands — or nothing, which is the answer on the
    /// overwhelming majority of turns.
    ///
    /// `argmax` is what the network chose. It is not a candidate here: the
    /// defense tactic needs it to ask whether the move about to be played is
    /// the one that loses.
    pub fn decide(
        &mut self,
        obs: &Observation,
        mem: &Memory,
        argmax: Action5,
        t0: Instant,
    ) -> Option<Action> {
        if !self.switches.tactics {
            return None;
        }
        let kill = kill::evaluate(obs, mem, &mut self.reach);
        let defense = defense::evaluate(obs, mem, &mut self.reach);
        self.watch(obs, mem, kill.as_ref(), defense.as_ref());

        if !self.switches.overrides || (kill.is_none() && defense.is_none()) {
            return None;
        }
        let pessimistic = Config {
            fog: Fog::Pessimistic,
            hidden_bound: hidden_army(obs) + FOG_GROWTH_MARGIN,
            deathtouch_turn: DEATHTOUCH_TURN,
            i_am_p0: self.i_am_p0,
        };
        // One board serves both searches: a search leaves it exactly as it
        // found it, because every advance it makes is taken back.
        let mut sim = Sim::from_frame(obs, mem, &pessimistic)?;

        if let Some(fire) = kill {
            if let Some(limits) = self.limits(t0) {
                self.set_window(obs, mem, fire.target, SEARCH_DEPTH + 1);
                let began = Instant::now();
                let report = kill::prove(&fire, &mut sim, &self.window, &limits);
                self.record(began, &report);
                self.kill_capped += u32::from(report.capped);
                if let Some(play) = report.play {
                    self.kill_proved += 1;
                    return Some(self.play(obs, "kill", &report, play));
                }
            }
        }

        if let Some(fire) = defense {
            // The cheap half first: with the network's move safe against every
            // reply we can see, there is nothing here to fix, and most fires
            // end on this line.
            let mut seen =
                Sim::from_frame(obs, mem, &Config { fog: Fog::VisibleOnly, ..pessimistic })?;
            self.set_window(obs, mem, fire.general, DEFENSE_DEPTH + 1);
            if !defense::argmax_loses(as_choice(argmax, obs), &mut seen, &self.window) {
                return None;
            }
            self.defense_refuted += 1;

            let limits = self.limits(t0)?;
            let began = Instant::now();
            let report = defense::prove(&fire, &mut sim, &self.window, &limits);
            self.record(began, &report);
            self.defense_capped += u32::from(report.capped);
            if let Some(play) = report.play {
                self.defense_proved += 1;
                return Some(self.play(obs, "defense", &report, play));
            }
        }
        None
    }

    /// The caps for one search: its own wall-clock slice, cut short by the
    /// turn deadline, and the node ceiling. `None` when the turn is already
    /// spent — an expired search would still cost a thousand nodes before its
    /// first clock check.
    fn limits(&self, t0: Instant) -> Option<Limits> {
        let hard = t0 + Duration::from_millis(TURN_DEADLINE_MS);
        let now = Instant::now();
        if now >= hard {
            return None;
        }
        Some(Limits {
            depth: SEARCH_DEPTH,
            nodes: PROOF_NODE_BUDGET,
            deadline: (now + Duration::from_millis(PROOF_DEADLINE_MS)).min(hard),
        })
    }

    /// Every cell within `radius` moves of `focus`: the window a search
    /// generates both sides' moves from. Range is measured the way the
    /// triggers measure it — moves over cells not remembered as mountains —
    /// because two readings of one rule is the bug [`common`] exists to
    /// prevent.
    ///
    /// One move past the horizon is the right radius. Every cell that can
    /// reach the general inside the horizon is inside it, and so is every cell
    /// of ours that can get there — a source one step further out can only
    /// interfere with a move of ours, and a move of ours that never happens
    /// still leaves the general where it was.
    fn set_window(&mut self, obs: &Observation, mem: &Memory, focus: usize, radius: i32) {
        let cells = obs.h * obs.w;
        self.reach.compute(obs, mem, focus, radius);
        self.window.clear();
        self.window.resize(cells, false);
        for cell in 0..cells {
            self.window[cell] = self.reach.reached(cell);
        }
    }

    /// What one search cost. The clock is read after the fact rather than
    /// handed to the search, which gets its own deadline instead.
    fn record(&mut self, began: Instant, report: &Report) {
        let ms = began.elapsed().as_secs_f64() * 1e3;
        self.slowest_proof_ms = self.slowest_proof_ms.max(ms);
        self.nodes += report.nodes;
    }

    /// Turn a proof into a reply, and say so on stderr: an override is rare
    /// enough that every one of them belongs in the log.
    fn play(&mut self, obs: &Observation, tactic: &str, report: &Report, play: Choice) -> Action {
        self.overrides += 1;
        let action = as_action(play, obs.w);
        eprintln!(
            "[unclejoe] turn {} override {tactic} depth {} nodes {} action {} {} {} {} {}",
            obs.turn,
            report.depth,
            report.nodes,
            action.pass,
            action.row,
            action.col,
            action.dir,
            action.split,
        );
        action
    }

    /// The shadow half, unchanged from U2: count the fires and report each one
    /// with the numbers it fired on. A search that fires on nearly every turn
    /// and one that never fires are different bots, and this is where that
    /// shows up.
    fn watch(
        &mut self,
        obs: &Observation,
        mem: &Memory,
        kill: Option<&kill::Fire>,
        defense: Option<&defense::Fire>,
    ) {
        self.turns += 1;
        self.first_contact_turn = mem.first_contact_turn;
        if mem.enemy_general.is_some() {
            self.general_known += 1;
        }
        if mem.general_visible_now {
            self.general_visible += 1;
        }

        if let Some(k) = kill {
            self.kill += 1;
            match k.cause {
                kill::Cause::ArmyBound => self.kill_army += 1,
                kill::Cause::Deathtouch => self.kill_deathtouch += 1,
            }
            let (row, col) = (k.target / obs.w, k.target % obs.w);
            eprintln!(
                "[unclejoe] turn {} trigger kill cause {} target {row},{col} \
                 reach_army {} last_seen_army {} stale {}",
                obs.turn,
                k.cause.name(),
                k.reach_army,
                k.last_seen_army,
                k.stale_turns,
            );
        }

        if let Some(d) = defense {
            self.defense += 1;
            match d.cause {
                defense::Cause::VisibleStack => self.defense_stack += 1,
                defense::Cause::Deathtouch => self.defense_deathtouch += 1,
                defense::Cause::Fog => self.defense_fog += 1,
            }
            let (row, col) = (d.source / obs.w, d.source % obs.w);
            eprintln!(
                "[unclejoe] turn {} trigger defense cause {} source {row},{col} \
                 threat_army {} general_army {}",
                obs.turn,
                d.cause.name(),
                d.threat_army,
                d.general_army,
            );
        }
    }

    /// One line per game, at EOF: the fire rates U2 exists to measure, and
    /// what U3's searches did with them. A layer that fires often and proves
    /// nothing and a layer that never fires are both failures, and they look
    /// nothing alike here.
    pub fn log_summary(&self) {
        if self.turns == 0 {
            return;
        }
        let pct = |count: u32| 100.0 * f64::from(count) / f64::from(self.turns);
        eprintln!(
            "[unclejoe] tactics shadow: turns {} first_contact {} \
             general_known {} ({:.1}%) \
             general_visible {} ({:.1}%) kill {} ({:.1}%: army {} deathtouch {}) \
             defense {} ({:.1}%: stack {} deathtouch {} fog {})",
            self.turns,
            self.first_contact_turn
                .map_or_else(|| "never".to_string(), |turn| turn.to_string()),
            self.general_known,
            pct(self.general_known),
            self.general_visible,
            pct(self.general_visible),
            self.kill,
            pct(self.kill),
            self.kill_army,
            self.kill_deathtouch,
            self.defense,
            pct(self.defense),
            self.defense_stack,
            self.defense_deathtouch,
            self.defense_fog,
        );
        eprintln!(
            "[unclejoe] tactics proofs: overrides {} ({:.2}%) \
             kill proved {} capped {} \
             defense refuted {} proved {} capped {} \
             nodes {} slowest_proof_ms {:.1}",
            self.overrides,
            pct(self.overrides),
            self.kill_proved,
            self.kill_capped,
            self.defense_refuted,
            self.defense_proved,
            self.defense_capped,
            self.nodes,
            self.slowest_proof_ms,
        );
    }
}

/// The network's action as the forward model sees it. A pass is `None`, and so
/// is a build: to a search about generals a build is a turn spent elsewhere
/// ([`crate::search::sim`] argues why that is the safe reading). An action in
/// the padded region is `None` too — the engine voids it, so the model must.
fn as_choice(action: Action5, obs: &Observation) -> Choice {
    if action.pass_field != 0 {
        return None;
    }
    let (row, col) = (action.row as usize, action.col as usize);
    if row >= obs.h || col >= obs.w {
        return None;
    }
    Some(Move { from: row * obs.w + col, dir: action.dir as u8, half: action.is_half == 1 })
}

/// And back to the wire.
fn as_action(play: Choice, w: usize) -> Action {
    match play {
        None => PASS,
        Some(mv) => Action {
            pass: 0,
            row: (mv.from / w) as u16,
            col: (mv.from % w) as u16,
            dir: mv.dir,
            split: u8::from(mv.half),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn five(action: Action) -> Action5 {
        Action5 {
            pass_field: action.pass as i32,
            row: action.row as i32,
            col: action.col as i32,
            dir: action.dir as i32,
            is_half: action.split as i32,
        }
    }

    #[test]
    fn an_action_survives_the_round_trip_to_the_model_and_back() {
        let obs = Observation::with_dims(9, 9);
        let moves = [(0usize, 0usize, 3u8, false), (8, 8, 0, true), (4, 5, 2, true)];
        for (row, col, dir, half) in moves {
            let choice = Some(Move { from: row * obs.w + col, dir, half });
            assert_eq!(as_choice(five(as_action(choice, obs.w)), &obs), choice);
        }
    }

    #[test]
    fn a_pass_a_build_and_a_pad_cell_are_all_nothing_to_the_model() {
        let obs = Observation::with_dims(9, 9);
        let pass = Action5 { pass_field: 1, row: 0, col: 0, dir: 0, is_half: 0 };
        let build = Action5 { pass_field: 2, row: 3, col: 3, dir: 0, is_half: 0 };
        // The action head is 21×21; a 9×9 board leaves most of it out of play.
        let padded = Action5 { pass_field: 0, row: 12, col: 2, dir: 1, is_half: 0 };
        for action in [pass, build, padded] {
            assert_eq!(as_choice(action, &obs), None, "{action:?}");
        }
        assert_eq!(as_action(None, obs.w), PASS);
    }
}
