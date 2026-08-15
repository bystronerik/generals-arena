//! The tactics layer — everything unclejoe does that joe-rs does not.
//!
//! At milestone U2 the layer only **watches**. It evaluates the two trigger
//! predicates against every frame and writes what they found to stderr;
//! nothing here touches the reply, so the two binaries still decide every
//! frame identically. That is the point of a shadow milestone: the triggers
//! are supersets by construction, and a superset's fire *rate* is an
//! empirical question — one worth answering before a search runs on the
//! answer, let alone an override.
//!
//! The layer reads a `&Memory` and never writes one. `Seat::act` owns the
//! memory update, for the same reason it owns `AugState`'s: per-game state
//! advances every turn regardless of who chooses the move.

pub mod triggers;

use crate::board::memory::Memory;
use crate::io::wire::Observation;
use crate::tactics::triggers::{KillCause, ThreatCause, Triggers};

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

/// The layer's per-game state: the trigger pass and the shadow counters.
#[derive(Debug, Default)]
pub struct Tactics {
    triggers: Triggers,
    turns: u32,
    /// Copied off memory so the summary can report it: the defense trigger is
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
    /// One shadow pass over a frame: evaluate both triggers, count, and log
    /// each fire with the numbers it fired on. Returns nothing, because at
    /// this milestone nothing may act on it.
    pub fn observe(&mut self, obs: &Observation, mem: &Memory) {
        let fired = self.triggers.evaluate(obs, mem);
        self.turns += 1;
        self.first_contact_turn = mem.first_contact_turn;
        if mem.enemy_general.is_some() {
            self.general_known += 1;
        }
        if mem.general_visible_now {
            self.general_visible += 1;
        }

        if let Some(k) = fired.kill {
            self.kill += 1;
            match k.cause {
                KillCause::ArmyBound => self.kill_army += 1,
                KillCause::Deathtouch => self.kill_deathtouch += 1,
            }
            let (row, col) = (k.target / obs.w, k.target % obs.w);
            eprintln!(
                "[unclejoe] turn {} trigger kill cause {} target {row},{col} \
                 reach_army {} last_seen_army {} stale {}",
                obs.turn,
                match k.cause {
                    KillCause::ArmyBound => "army",
                    KillCause::Deathtouch => "deathtouch",
                },
                k.reach_army,
                k.last_seen_army,
                k.stale_turns,
            );
        }

        if let Some(d) = fired.defense {
            self.defense += 1;
            match d.cause {
                ThreatCause::VisibleStack => self.defense_stack += 1,
                ThreatCause::Deathtouch => self.defense_deathtouch += 1,
                ThreatCause::Fog => self.defense_fog += 1,
            }
            let (row, col) = (d.source / obs.w, d.source % obs.w);
            eprintln!(
                "[unclejoe] turn {} trigger defense cause {} source {row},{col} \
                 threat_army {} general_army {}",
                obs.turn,
                match d.cause {
                    ThreatCause::VisibleStack => "stack",
                    ThreatCause::Deathtouch => "deathtouch",
                    ThreatCause::Fog => "fog",
                },
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
