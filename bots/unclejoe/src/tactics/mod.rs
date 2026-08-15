//! The tactics layer — everything unclejoe does that joe-rs does not.
//!
//! At milestone U2 the layer only **watches**. It evaluates the two tactics'
//! trigger predicates against every frame and writes what they found to
//! stderr; nothing here touches the reply, so the two binaries still decide
//! every frame identically. That is the point of a shadow milestone: the
//! triggers are supersets by construction, and a superset's fire *rate* is an
//! empirical question — one worth answering before a search runs on the
//! answer, let alone an override.
//!
//! # Shape
//!
//! **One file per tactic.** [`kill`] and [`defense`] each own a predicate,
//! the result type it returns, and the tests for it; each will grow its proof
//! search invocation in U3, beside the predicate that gates it. Everything
//! neither of them owns alone is in [`common`] — distance in moves, and what
//! a frame states about the army it does not show. This file is the
//! composition root: it owns the shared scratch, runs the tactics, and keeps
//! the shadow counters. A tactic never calls another tactic, and never
//! reaches into another tactic's file for a helper; they meet here and in
//! [`common`], and nowhere else.
//!
//! A tactic's predicate is neither a decision nor a heuristic. It is a cheap,
//! exact test for "an exact search could have something to prove here", built
//! as a **superset**: every position where the goal is provable inside the
//! depth budget fires it, and the converse is not claimed. A fire on a dead
//! position costs one search that then declines; a miss costs the point of
//! the bot. Every bound is therefore loose in that one safe direction — with
//! one named exception, the first-contact gate in [`defense`].
//!
//! The layer reads a `&Memory` and never writes one. `Seat::act` owns the
//! memory update, for the same reason it owns `AugState`'s: per-game state
//! advances every turn regardless of who chooses the move.

pub mod common;
pub mod defense;
pub mod kill;

use crate::board::memory::Memory;
use crate::io::wire::Observation;
use crate::tactics::common::Reach;

// ---- constants ------------------------------------------------------------
//
// The tactics layer's numbers live here and nowhere else. The tactics plan
// and the strategy spec both quote them, so a change here is a change to two
// documents. The deadline and node-budget constants join this block with the
// proof search (milestone U3).

/// `D` — the proof search's depth budget in our own moves, and therefore the
/// radius the triggers scan. Beyond three plies the pessimistic fog bound
/// dominates and proofs stop landing: the binding constraint is knowledge,
/// not compute.
pub const SEARCH_DEPTH: i32 = 3;

/// The turn deathtouch goes live: from here a move that executes onto the
/// enemy general's tile wins whatever the garrison holds (RULES.md §07). The
/// engine's `deathtouch.py` compares `state.time >= 800`, and `obs.turn` is
/// that same clock.
pub const DEATHTOUCH_TURN: i32 = 800;

/// The layer's per-game state: the shared reach scratch and the shadow
/// counters.
#[derive(Debug, Default)]
pub struct Tactics {
    reach: Reach,
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
    defense: u32,
    defense_stack: u32,
    defense_deathtouch: u32,
    defense_fog: u32,
}

impl Tactics {
    /// One shadow pass over a frame: run both tactics, count, and log each
    /// fire with the numbers it fired on. Returns nothing, because at this
    /// milestone nothing may act on it.
    pub fn observe(&mut self, obs: &Observation, mem: &Memory) {
        let kill = kill::evaluate(obs, mem, &mut self.reach);
        let defense = defense::evaluate(obs, mem, &mut self.reach);

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

    /// One line per game, at EOF: the fire rates the U2 milestone exists to
    /// measure. A search that fires on nearly every turn and one that never
    /// fires are different bots, and this is where that shows up.
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
    }
}
