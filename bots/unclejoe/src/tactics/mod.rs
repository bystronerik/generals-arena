//! The tactics layer — everything unclejoe does that joe-rs does not.
//!
//! Three mechanisms, strictly layered, and only the first of them can move the
//! reply today.
//!
//! 1. **The proof-gated override** (U3). Every turn the layer evaluates the two
//!    tactics' trigger predicates against the frame; on a fire it runs the
//!    bounded exact search behind that trigger; and a search that *proves* its
//!    goal replaces the network's move. A trigger that stays quiet, a search
//!    that declines, a cap that trips — each leaves the reply exactly where
//!    joe-rs would have left it.
//! 2. **The candidate filters** ([`filters`], U4). Exact arithmetic that
//!    removes candidates from the policy's shortlist. They never rank.
//! 3. **The afterstate re-rank** (U4, in shadow). The layer hands `Seat::act`
//!    the surviving candidates; `Seat::act` values each one's afterstate with
//!    the network's own value head and reports what a live re-rank *would*
//!    have played. At this milestone it plays the argmax regardless — the
//!    numbers go to `docs/bots/unclejoe/shadow.md`, and U5 decides whether the
//!    loop goes live.
//!
//! # Shape
//!
//! **One file per tactic.** [`kill`] and [`defense`] each own a predicate, the
//! result type it returns, the search call that predicate gates, and the tests
//! for all of it. Everything neither of them owns alone is in [`common`] —
//! distance in moves, and what a frame states about the army it does not show.
//! [`filters`] owns the masks, which belong to no tactic at all. This file is
//! the composition root: it owns the constants, the shared scratch, the
//! counters, and the order everything runs in. A tactic never calls another
//! tactic, and never reaches into another tactic's file for a helper; they
//! meet here and in [`common`], and nowhere else.
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
//! # What this layer does not touch
//!
//! It reads a `&Memory` and never writes one, and it never names the network.
//! `Seat::act` owns the memory update for the same reason it owns `AugState`'s
//! — per-game state advances every turn regardless of who chooses the move —
//! and it owns the forwards because `nn` sits above this layer and must stay
//! byte-identical to joe-rs's.

pub mod common;
pub mod defense;
pub mod filters;
pub mod kill;

use std::time::{Duration, Instant};

use crate::board::action::{decode_action, Action5};
use crate::board::memory::Memory;
use crate::io::wire::{Action, Observation, PASS};
use crate::search::afterstate::Play;
use crate::search::minimax::{self, Limits, Report};
use crate::search::sim::{Choice, Config, Fog, Move, Sim};
use crate::tactics::common::{hidden_army, Reach};
use crate::tactics::filters::Mask;

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

/// How many of the policy's actions the re-rank looks at. The argmax is the
/// first of them, which is what makes the loop anytime: cut it off anywhere
/// and the reply degrades to exactly joe-rs's.
///
/// **Four, because four is what the turn pays for.** One evaluation is one
/// forward, the live path already spends one, and the deadline is 130 ms: at
/// joe-rs's measured ~24 ms per forward that leaves room for four and not
/// five. U4 shipped this at five and measured the loop cutting itself off on
/// 4,008 of 4,037 turns at an average of 3.9 — the anytime cutoff was quietly
/// supplying the real constant, so it is written down here instead.
pub const TOP_K: usize = 4;

/// The floor under the re-rank's reserve, and what it uses before it has
/// measured anything.
///
/// **The reserve itself is measured, not guessed.** An evaluation is a whole
/// forward pass, so a fixed reserve is a claim about how long a forward takes
/// on a host this code has never run on — and U4 shipped that claim at 25 ms
/// against a forward that costs 22 ms typical and 48 ms at its worst, which
/// let the loop start work it could not finish. [`Tactics::reserve_ms`]
/// instead keeps the cost of the slowest evaluation this game and refuses to
/// start another with less than that left; the warmup forward seeds it before
/// the first frame, so this floor binds only if that came back implausibly
/// cheap.
pub const RERANK_RESERVE_FLOOR_MS: u64 = 25;

/// How far a rival's afterstate value must exceed the argmax's before it may
/// take the move. Below this the two positions are a tie and the policy keeps
/// the turn.
///
/// **One bin of the value head**, which is a distribution over 128 bins
/// spanning [−1, +1] (`bin_centers` in the artifact) — so `2/127`. The head's
/// output is a continuous expectation over that softmax, so a smaller
/// difference is representable; what it is not is *evidence*. The head was
/// trained against a 128-bin target, so bin width is the scale at which it
/// learned to separate one outcome from another, and a gap an order of
/// magnitude below that is the softmax leaning between two adjacent bins
/// rather than a preference between two positions.
///
/// U4 measured why this is needed: without it the re-rank overrules the
/// policy on 46.8% of turns and 83% of those ride a sub-bin gap
/// (`docs/bots/unclejoe/shadow.md` §U4).
///
/// Written as a number rather than imported: `tactics` sits below `nn` in the
/// layering and must not name it, exactly as `DEATHTOUCH_TURN` spells out a
/// fact about the engine. The cost is that a joe re-export with a different
/// bin count makes this stale silently — the artifact's bin count belongs on
/// the re-export checklist beside the weights.
pub const RERANK_MIN_GAP: f32 = 2.0 / 127.0;

/// The most §03 crowding surcharge a build may carry and still be valued.
/// Eight admits exactly one own structure at three steps or further; anything
/// closer, or two structures crowding at all, is masked.
///
/// **Uncalibrated.** It is a guess about what a castle is worth, not a fact
/// about the rules, and revising it is strategist work against shadow and
/// round data (tactics-plan.md §3).
pub const CASTLE_SURCHARGE_CAP: i32 = 8;

/// The last turn a build is worth its payback. A castle produces one army
/// every second turn (§04), so a build at price `p` returns what it cost after
/// `2p` turns; past this the game ends first.
///
/// **Uncalibrated**, on the same terms as [`CASTLE_SURCHARGE_CAP`].
pub const CASTLE_LATE_TURN: i32 = 650;

/// The kill-switches, read once at construction.
///
/// `UNCLEJOE_TACTICS=0` is the master: pure pass-through, byte-identical
/// behavior to joe-rs, and what the wire-replay test runs under. The other
/// three take one mechanism away each, so a decomposition arm is a one-line
/// `run.sh` export (tactics-plan.md §6).
///
/// `UNCLEJOE_OVERRIDE=0` keeps the triggers and their reports and takes the
/// override away — the U2 shadow build, reachable from any later one.
/// `UNCLEJOE_MASKS=0` and `UNCLEJOE_RERANK=0` take the filters and the
/// afterstate loop away. At U4 both of those mechanisms are themselves in
/// shadow, so switching one off removes its *measurement* rather than its
/// effect: `UNCLEJOE_RERANK=0` is how a U4 build runs at U3's cost.
#[derive(Debug, Clone, Copy)]
struct Switches {
    tactics: bool,
    overrides: bool,
    masks: bool,
    rerank: bool,
}

impl Switches {
    fn from_env() -> Self {
        let on = |name: &str| !std::env::var(name).is_ok_and(|value| value == "0");
        Self {
            tactics: on("UNCLEJOE_TACTICS"),
            overrides: on("UNCLEJOE_OVERRIDE"),
            masks: on("UNCLEJOE_MASKS"),
            rerank: on("UNCLEJOE_RERANK"),
        }
    }
}

/// What `Seat::act` hands the layer: the network's choice, its logits, and the
/// two grids the pipeline already computed. Passed in rather than recomputed,
/// because `penalties` *is* the engine's legality rule and `cost` *is* §03's
/// price — a second reading of either is a second answer.
#[derive(Debug, Clone, Copy)]
pub struct Policy<'a> {
    pub argmax: Action5,
    pub logits: &'a [f32],
    pub penalties: &'a [f32],
    pub cost: &'a [i32],
}

/// One action the policy named, with the filters' verdict on it.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub action: Action5,
    /// The same action as the forward model takes it.
    pub play: Play,
    pub logit: f32,
    /// Why a filter removed it, or `None` if it survived.
    pub masked: Option<Mask>,
}

/// A candidate that stands for nothing — the filler an empty slate slot holds.
const NOTHING: Candidate = Candidate {
    action: Action5 { pass_field: 1, row: 0, col: 0, dir: 0, is_half: 0 },
    play: Play::Pass,
    logit: f32::NEG_INFINITY,
    masked: None,
};

/// Up to [`TOP_K`] candidates, inline and `Copy`.
///
/// Small on purpose: the per-turn path allocates nothing, and `Seat::act` can
/// hold a copy of the slate while it borrows the layer again to report what
/// the forwards found.
#[derive(Debug, Clone, Copy)]
pub struct Slate {
    items: [Candidate; TOP_K],
    len: usize,
}

impl Slate {
    fn empty() -> Self {
        Self { items: [NOTHING; TOP_K], len: 0 }
    }

    fn push(&mut self, candidate: Candidate) {
        if self.len < TOP_K {
            self.items[self.len] = candidate;
            self.len += 1;
        }
    }
}

impl std::ops::Deref for Slate {
    type Target = [Candidate];

    fn deref(&self) -> &[Candidate] {
        &self.items[..self.len]
    }
}

/// What one turn of the layer decided.
///
/// A quarter of a kilobyte, because a [`Slate`] is inline rather than boxed —
/// which is the trade this type is here to make. It is built once per turn and
/// lives on the stack; a heap allocation on the per-move path would cost more
/// than the copy does.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::large_enum_variant)]
pub enum Decision {
    /// A proof demands this action. The turn is over.
    Override(Action),
    /// Nothing was proven, so the network's move stands. These are the
    /// actions the policy named, in its own order, with the filters' verdicts
    /// attached — for `Seat::act` to value, because the network lives above
    /// this layer. Empty when the layer is switched off.
    Candidates(Slate),
}

/// What one candidate's afterstate was worth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Score {
    /// Not evaluated: a filter removed it, or the clock ran out first.
    Skipped,
    /// What the network's own value head put on the position.
    Value(f32),
    /// The candidate takes their general while the opponent stands still.
    /// There is no position left to value and nothing ranks above it.
    Wins,
}

impl Score {
    pub fn value(self) -> Option<f32> {
        match self {
            Score::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Is this score strictly better than that one? A win beats every number,
    /// a number beats an evaluation that never happened, and equal is not
    /// better — which is what keeps policy order as the tiebreak.
    fn beats(self, other: Score) -> bool {
        match (self, other) {
            (Score::Wins, Score::Wins) => false,
            (Score::Wins, _) => true,
            (_, Score::Wins) => false,
            (Score::Value(a), Score::Value(b)) => a > b,
            (Score::Value(_), Score::Skipped) => true,
            _ => false,
        }
    }

    /// Is this score better than that one by enough to overrule the policy?
    ///
    /// A win is never marginal: a candidate that captures the general inside
    /// the model does not have to clear a threshold about value resolution.
    /// Everything else needs [`RERANK_MIN_GAP`] between two real numbers — an
    /// evaluation that never happened is not a baseline a gap can be measured
    /// against, so it declines rather than guesses.
    fn clears(self, baseline: Score) -> bool {
        match (self, baseline) {
            (Score::Wins, Score::Wins) => false,
            (Score::Wins, _) => true,
            (Score::Value(a), Score::Value(b)) => a - b >= RERANK_MIN_GAP,
            _ => false,
        }
    }
}

/// The action a live re-rank would play, as a slate index.
///
/// Three rules, in order. A filtered candidate is never played. The **default**
/// is the argmax — the policy's own move — unless a filter removed it, in which
/// case it is the first survivor in policy order. And a rival takes the turn
/// off the default only when it beats it by [`RERANK_MIN_GAP`]; short of that
/// the two positions are a tie, and a tie belongs to the policy, which is right
/// far more often than a value comparison at this scale.
///
/// `None` when every candidate was filtered. The caller plays the argmax:
/// filters that remove everything have removed nothing.
fn live_pick(slate: &Slate, scored: &[Score]) -> Option<usize> {
    let default = (0..slate.len()).find(|rank| slate[*rank].masked.is_none())?;
    let mut best = default;
    for rank in default + 1..slate.len() {
        if slate[rank].masked.is_none() && scored[rank].beats(scored[best]) {
            best = rank;
        }
    }
    Some(if best != default && scored[best].clears(scored[default]) {
        best
    } else {
        default
    })
}

/// The best-scored survivor, with no threshold applied — what [`live_pick`]
/// would have returned before [`RERANK_MIN_GAP`] existed. Kept so the shadow
/// report can say what the threshold cost and what it saved.
fn best_survivor(slate: &Slate, scored: &[Score]) -> Option<usize> {
    let first = (0..slate.len()).find(|rank| slate[*rank].masked.is_none())?;
    let mut best = first;
    for rank in first + 1..slate.len() {
        if slate[rank].masked.is_none() && scored[rank].beats(scored[best]) {
            best = rank;
        }
    }
    Some(best)
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
    // U4, all shadow: what the filters would have removed and what the
    // afterstate values would have chosen.
    rerank_turns: u32,
    rerank_forwards: u32,
    rerank_cutoffs: u32,
    rerank_wins: u32,
    would_change: u32,
    /// Turns where a rival out-scored the argmax at all, before the threshold.
    /// The difference between this and `would_change` is what
    /// [`RERANK_MIN_GAP`] refuses.
    rivals_preferred: u32,
    argmax_masked: u32,
    masked_crowding: u32,
    masked_lateness: u32,
    masked_refuted: u32,
    all_masked: u32,
    /// Every turn's `best − argmax` value gap, kept whole rather than
    /// summarized on the fly: U4's go/no-go is a question about the *shape* of
    /// this distribution, and a mean cannot answer it. One f32 per turn, so a
    /// 1,200-turn game costs under 5 KB against a 2 GB cap.
    gaps: Vec<f32>,
    /// The most one afterstate evaluation has cost this game, in ms. This is
    /// the reserve — see [`Tactics::reserve_ms`].
    slowest_eval_ms: f64,
}

impl Tactics {
    pub fn new(i_am_p0: bool) -> Self {
        let switches = Switches::from_env();
        if !switches.tactics {
            eprintln!("[unclejoe] tactics off: the reply is the network's, every turn");
        } else if !switches.overrides {
            eprintln!("[unclejoe] tactics in shadow: triggers report, nothing overrides");
        }
        if switches.tactics {
            eprintln!(
                "[unclejoe] re-rank in shadow: masks {} rerank {} (the reply is the argmax either way)",
                switches.masks, switches.rerank,
            );
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
            rerank_turns: 0,
            rerank_forwards: 0,
            rerank_cutoffs: 0,
            rerank_wins: 0,
            would_change: 0,
            rivals_preferred: 0,
            argmax_masked: 0,
            masked_crowding: 0,
            masked_lateness: 0,
            masked_refuted: 0,
            all_masked: 0,
            gaps: Vec::new(),
            slowest_eval_ms: 0.0,
        }
    }

    /// Is the afterstate loop switched on? `Seat::act` asks before it spends a
    /// forward, because that is the one part of this milestone with a cost.
    pub fn reranking(&self) -> bool {
        self.switches.tactics && self.switches.rerank
    }

    /// Seed the reserve from a forward this host has already paid for.
    ///
    /// `Seat::new` runs one warmup forward on zeros before the first frame, so
    /// this costs nothing extra and answers the question a fixed constant can
    /// only guess at: how long does a forward take *here*. A cold forward errs
    /// high, which is the direction a reserve should err.
    pub fn seed_reserve(&mut self, warmup_ms: f64) {
        self.note_evaluation(warmup_ms);
        eprintln!("[unclejoe] re-rank reserve seeded at {} ms", self.reserve_ms());
    }

    /// What one afterstate evaluation cost, folded into the reserve.
    pub fn note_evaluation(&mut self, ms: f64) {
        if ms.is_finite() {
            self.slowest_eval_ms = self.slowest_eval_ms.max(ms);
        }
    }

    /// How much of the turn to keep back before starting another evaluation:
    /// the worst one this game has cost, never below the floor.
    ///
    /// Measuring rather than guessing is what makes this host-independent. A
    /// slower machine measures a bigger reserve and runs fewer candidates; the
    /// loop is anytime, so running fewer is a degradation and not a fault.
    fn reserve_ms(&self) -> u64 {
        RERANK_RESERVE_FLOOR_MS.max(self.slowest_eval_ms.ceil() as u64)
    }

    /// Is there room in the turn to start another afterstate evaluation?
    ///
    /// This is the whole of the anytime guarantee, and it only holds while the
    /// reserve covers what an evaluation actually costs — which is why the
    /// reserve is measured. The argmax is candidate #1, so a loop cut off at
    /// any point, here or before the first evaluation, plays what joe-rs would
    /// have played.
    pub fn have_time(&self, t0: Instant) -> bool {
        let spent = t0.elapsed().as_millis();
        let left = i128::from(TURN_DEADLINE_MS) - spent as i128;
        left >= i128::from(self.reserve_ms())
    }

    /// The forward model an afterstate advances on: the board as the frame
    /// states it, with fog left empty. Never the pessimistic one — that is a
    /// proof device, and a board carrying the opponent's whole army on every
    /// dark cell is not a position to ask a value head about.
    pub fn afterstate_config(&self) -> Config {
        Config {
            fog: Fog::VisibleOnly,
            hidden_bound: 0,
            deathtouch_turn: DEATHTOUCH_TURN,
            i_am_p0: self.i_am_p0,
        }
    }

    /// One turn of the layer: run both tactics against the frame, and return
    /// the action a proof demands — or, on the overwhelming majority of turns,
    /// the shortlist the network named with the filters' verdicts on it.
    ///
    /// `policy.argmax` is what the network chose. It is a candidate *and* an
    /// argument: the defense tactic needs it to ask whether the move about to
    /// be played is the one that loses.
    pub fn decide(
        &mut self,
        obs: &Observation,
        mem: &Memory,
        policy: &Policy,
        t0: Instant,
    ) -> Decision {
        if !self.switches.tactics {
            return Decision::Candidates(Slate::empty());
        }
        if let Some(proven) = self.override_move(obs, mem, policy.argmax, t0) {
            return Decision::Override(proven);
        }
        Decision::Candidates(self.slate(obs, mem, policy))
    }

    /// The U3 half: a proof, or nothing.
    fn override_move(
        &mut self,
        obs: &Observation,
        mem: &Memory,
        argmax: Action5,
        t0: Instant,
    ) -> Option<Action> {
        let kill = kill::evaluate(obs, mem, &mut self.reach);
        let defense = defense::evaluate(obs, mem, &mut self.reach);
        self.watch(obs, mem, kill.as_ref(), defense.as_ref());

        if !self.switches.overrides || (kill.is_none() && defense.is_none()) {
            return None;
        }
        let pessimistic = self.pessimistic(obs);
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

    /// The policy's [`TOP_K`] best **legal** actions, in its own order, with
    /// every filter's verdict attached.
    ///
    /// Legality is `policy.penalties`: the pipeline's own action mask, which
    /// is the engine's rule and not a second reading of it. The masked entries
    /// carry `-1e9`, so the first candidate here is the same action
    /// `argmax(logits)` named — the argmax is always legal, because the pass
    /// channel always is and no logit comes near a billion.
    fn slate(&mut self, obs: &Observation, mem: &Memory, policy: &Policy) -> Slate {
        let mut slate = Slate::empty();
        let mut taken = [usize::MAX; TOP_K];
        for rank in 0..TOP_K {
            let mut best: Option<usize> = None;
            for idx in 0..policy.logits.len() {
                if policy.penalties[idx] < 0.0 || taken[..rank].contains(&idx) {
                    continue;
                }
                // Strictly greater, so equal logits fall to the lower index —
                // the same first-max rule `board::action::argmax` follows.
                if best.is_none_or(|b| policy.logits[idx] > policy.logits[b]) {
                    best = Some(idx);
                }
            }
            let Some(idx) = best else {
                break;
            };
            taken[rank] = idx;
            let action = decode_action(idx);
            slate.push(Candidate {
                action,
                play: as_play(action, obs),
                logit: policy.logits[idx],
                masked: self.castle_mask(obs, policy, action),
            });
        }
        self.veto(obs, mem, &mut slate);
        slate
    }

    /// The two castle masks, on a build candidate only.
    fn castle_mask(
        &self,
        obs: &Observation,
        policy: &Policy,
        action: Action5,
    ) -> Option<Mask> {
        if !self.switches.masks || action.pass_field != 2 {
            return None;
        }
        let (row, col) = (action.row as usize, action.col as usize);
        if row >= obs.h || col >= obs.w {
            return None;
        }
        filters::castle(obs, policy.cost, row * obs.w + col)
    }

    /// The refutation veto, over the whole slate at once.
    ///
    /// Gated on [`filters::refutation_possible`], which is exact: with no
    /// enemy stack beside our general no candidate can be refuted, and that is
    /// nearly every turn. When the gate does open, the board is the
    /// visible-only one — a veto must never remove a candidate that is
    /// actually safe, so every reply it counts has to be one the opponent
    /// demonstrably has.
    fn veto(&mut self, obs: &Observation, mem: &Memory, slate: &mut Slate) {
        if !self.switches.masks || !filters::refutation_possible(obs, mem) {
            return;
        }
        let Some(general) = mem.own_general else {
            return;
        };
        let config = Config { fog: Fog::VisibleOnly, ..self.pessimistic(obs) };
        let Some(mut seen) = Sim::from_frame(obs, mem, &config) else {
            return;
        };
        self.set_window(obs, mem, general, DEFENSE_DEPTH + 1);
        for slot in 0..slate.len {
            if slate.items[slot].masked.is_some() {
                continue;
            }
            // A build is a pass to this question: it moves no army off the
            // general and hands the opponent the same board a pass would.
            let choice = as_choice(slate.items[slot].action, obs);
            if minimax::refutes(&mut seen, choice, &self.window) {
                slate.items[slot].masked = Some(Mask::Refuted);
            }
        }
    }

    /// The proof board's configuration: fog as the worst it could be, plus the
    /// margin an aged bound needs.
    fn pessimistic(&self, obs: &Observation) -> Config {
        Config {
            fog: Fog::Pessimistic,
            hidden_bound: hidden_army(obs) + FOG_GROWTH_MARGIN,
            deathtouch_turn: DEATHTOUCH_TURN,
            i_am_p0: self.i_am_p0,
        }
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

    /// The U4 shadow half: what the filters removed, what the afterstate
    /// values said, and what a live re-rank would therefore have played.
    /// Nothing here changed the reply — `Seat::act` plays the argmax at this
    /// milestone whatever this reports.
    ///
    /// `scored` is one entry per slate slot. The argmax's entry is always a
    /// value when the loop ran at all, because it is the baseline every gap is
    /// measured against; the rest are values only where a filter let them
    /// through and the clock allowed it.
    pub fn report_rerank(
        &mut self,
        obs: &Observation,
        slate: &Slate,
        scored: &[Score],
        cutoff: bool,
    ) {
        if slate.is_empty() {
            return;
        }
        let mut surviving = 0;
        for (rank, candidate) in slate.iter().enumerate() {
            match candidate.masked {
                Some(Mask::Crowding) => self.masked_crowding += 1,
                Some(Mask::Lateness) => self.masked_lateness += 1,
                Some(Mask::Refuted) => self.masked_refuted += 1,
                None => surviving += 1,
            }
            if rank == 0 && candidate.masked.is_some() {
                self.argmax_masked += 1;
            }
        }
        if surviving == 0 {
            // Every candidate filtered means the filters are moot: U5 plays
            // the argmax, which is what U4 plays anyway.
            self.all_masked += 1;
        }
        let forwards = scored.iter().filter(|s| s.value().is_some()).count() as u32;
        self.rerank_forwards += forwards;
        self.rerank_wins += scored.iter().filter(|s| **s == Score::Wins).count() as u32;
        self.rerank_cutoffs += u32::from(cutoff);
        if forwards == 0 && !scored.contains(&Score::Wins) {
            return;
        }
        self.rerank_turns += 1;

        // What a live loop would play, and what it would have played without
        // the threshold. Both, because the second is the measurement U4 was
        // built to take and the first is the mechanism that measurement
        // argued for; reporting only one of them hides the threshold's effect.
        let pick = live_pick(slate, scored);
        let preferred = best_survivor(slate, scored);
        let changed = pick.is_some_and(|rank| rank != 0);
        self.would_change += u32::from(changed);
        self.rivals_preferred += u32::from(preferred.is_some_and(|rank| rank != 0));

        // The gap stays the *ungated* one — best rival against the argmax —
        // so the distribution the threshold is drawn from keeps being
        // measured after the threshold exists.
        let gap = match (preferred.map(|rank| scored[rank]), scored[0]) {
            (Some(Score::Value(best)), Score::Value(argmax)) => Some(best - argmax),
            _ => None,
        };
        if let Some(gap) = gap {
            self.gaps.push(gap);
        }

        let values: Vec<String> = slate
            .iter()
            .zip(scored)
            .map(|(candidate, score)| match (score, candidate.masked) {
                (Score::Value(v), _) => format!("{v:+.4}"),
                (Score::Wins, _) => "win".to_string(),
                (Score::Skipped, Some(mask)) => mask.name().to_string(),
                (Score::Skipped, None) => "-".to_string(),
            })
            .collect();
        // The logit margin says *what kind* of change this would be: a rival
        // the policy nearly named anyway, or one it ranked far below the
        // argmax. U5's go/no-go cares about the difference — overruling a
        // near-tie is a different bet from overruling a confident policy.
        let margin = pick.map(|rank| slate[rank].logit - slate[0].logit);
        eprintln!(
            "[unclejoe] turn {} rerank k {} forwards {} pick {} preferred {} change {} \
             gap {} logit_margin {} values {}",
            obs.turn,
            slate.len(),
            forwards,
            pick.map_or_else(|| "argmax".to_string(), |rank| rank.to_string()),
            preferred.map_or_else(|| "argmax".to_string(), |rank| rank.to_string()),
            changed,
            gap.map_or_else(|| "n/a".to_string(), |g| format!("{g:+.4}")),
            margin.map_or_else(|| "n/a".to_string(), |m| format!("{m:+.3}")),
            values.join(" "),
        );
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
        self.log_rerank_summary();
    }

    /// The U4 line: the shape of the value-gap distribution, which is what the
    /// go/no-go in tactics-plan.md §7 turns on. A re-rank worth switching on
    /// needs gaps that stand clear of the near-zero mass — so the quantiles
    /// are reported, not the mean, which a pile of zeros would hide behind.
    fn log_rerank_summary(&self) {
        if self.turns == 0 {
            return;
        }
        let pct = |count: u32| 100.0 * f64::from(count) / f64::from(self.turns);
        let mut gaps = self.gaps.clone();
        gaps.sort_by(|a, b| a.partial_cmp(b).expect("a value gap is never NaN"));
        let quantile = |q: f64| {
            gaps.first().map_or(f64::NAN, |_| {
                f64::from(gaps[((gaps.len() as f64 - 1.0) * q) as usize])
            })
        };
        eprintln!(
            "[unclejoe] tactics rerank: turns {} ({:.1}%) forwards {} cutoffs {} wins {} \
             slowest_eval_ms {:.1} reserve_ms {} \
             min_gap {:.4} rivals_preferred {} ({:.1}%) would_change {} ({:.1}%) \
             argmax_masked {} all_masked {} \
             masked crowding {} late {} refuted {} \
             gap p50 {:.4} p90 {:.4} p99 {:.4} max {:.4}",
            self.rerank_turns,
            pct(self.rerank_turns),
            self.rerank_forwards,
            self.rerank_cutoffs,
            self.rerank_wins,
            self.slowest_eval_ms,
            self.reserve_ms(),
            RERANK_MIN_GAP,
            self.rivals_preferred,
            pct(self.rivals_preferred),
            self.would_change,
            pct(self.would_change),
            self.argmax_masked,
            self.all_masked,
            self.masked_crowding,
            self.masked_lateness,
            self.masked_refuted,
            quantile(0.50),
            quantile(0.90),
            quantile(0.99),
            quantile(1.0),
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

/// The network's action as the **afterstate** takes it, which is the same
/// mapping with one difference: a build is a build here, not a pass. The proof
/// searches may treat it as a turn spent elsewhere; a position cannot, because
/// the price comes off a real cell and the castle starts producing (§03).
fn as_play(action: Action5, obs: &Observation) -> Play {
    let (row, col) = (action.row as usize, action.col as usize);
    if row >= obs.h || col >= obs.w {
        return Play::Pass;
    }
    let cell = row * obs.w + col;
    match action.pass_field {
        0 => Play::Act(Move { from: cell, dir: action.dir as u8, half: action.is_half == 1 }),
        2 => Play::Build(cell),
        _ => Play::Pass,
    }
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
    use crate::board::action::encode_action;
    use crate::board::obs::{
        build_cost_from_raw, frame_to_raw, prepare_action_mask, CELLS, N_ACTION_CHANNELS,
    };
    use crate::io::wire::{OWNER_ME, OWNER_OPP, TYPE_GENERAL, TYPE_PLAIN};
    use crate::tactics::common::fixtures::{at, frame, put, remembering, total_armies};
    use std::time::Duration;

    /// The pipeline's own masks and prices for a frame, exactly as `Seat::act`
    /// computes them: a slate test that fabricated its own legality would be
    /// testing the fabrication.
    fn pipeline(obs: &Observation) -> (Vec<f32>, Vec<i32>) {
        let (mut raw, mut cost) = (Vec::new(), Vec::new());
        let (mut move_mask, mut build_mask) = (Vec::new(), Vec::new());
        frame_to_raw(obs, &mut raw);
        build_cost_from_raw(&raw, obs.h, obs.w, &mut cost);
        crate::board::obs::compute_valid_move_mask(&raw, obs.h, obs.w, &mut move_mask);
        crate::board::obs::compute_build_mask_from_raw(
            &raw,
            obs.h,
            obs.w,
            &cost,
            &mut build_mask,
        );
        let mut penalties = vec![0.0; N_ACTION_CHANNELS * CELLS];
        prepare_action_mask(&move_mask, &build_mask, obs.h, obs.w, &mut penalties);
        (penalties, cost)
    }

    /// Logits that name these actions in this order, best first, and leave
    /// everything else far behind.
    fn logits_favouring(order: &[Action5]) -> Vec<f32> {
        let mut logits = vec![-100.0; N_ACTION_CHANNELS * CELLS];
        for (rank, action) in order.iter().enumerate() {
            logits[encode_action(*action)] = 10.0 - rank as f32;
        }
        logits
    }

    fn slate_for(obs: &Observation, order: &[Action5]) -> Slate {
        let (penalties, cost) = pipeline(obs);
        let logits = logits_favouring(order);
        let policy = Policy { argmax: order[0], logits: &logits, penalties: &penalties, cost: &cost };
        let mut tactics = Tactics::new(true);
        tactics.slate(obs, &remembering(&[obs]), &policy)
    }

    fn move_action(row: i32, col: i32, dir: i32) -> Action5 {
        Action5 { pass_field: 0, row, col, dir, is_half: 0 }
    }

    fn build_action(row: i32, col: i32) -> Action5 {
        Action5 { pass_field: 2, row, col, dir: 0, is_half: 0 }
    }

    /// The slate is the policy's own order, and its first entry is the argmax
    /// — which is what makes the re-rank loop anytime.
    #[test]
    fn the_slate_is_the_policy_order_and_the_argmax_leads_it() {
        let mut obs = frame(100);
        put(&mut obs, 4, 4, TYPE_GENERAL, OWNER_ME, 20);
        total_armies(&mut obs);
        let wanted = [move_action(4, 4, 1), move_action(4, 4, 3), move_action(4, 4, 0)];
        let slate = slate_for(&obs, &wanted);

        assert_eq!(slate.len(), TOP_K);
        for (candidate, want) in slate.iter().zip(&wanted) {
            assert_eq!(candidate.action, *want);
        }
        assert_eq!(slate[0].play, Play::Act(Move { from: at(4, 4), dir: 1, half: false }));
    }

    /// An action the pipeline masked never reaches the slate. Here our general
    /// holds one army, so §02 says it cannot move at all and the only legal
    /// action on the board is the pass.
    #[test]
    fn an_illegal_action_is_not_a_candidate() {
        let mut obs = frame(100);
        put(&mut obs, 4, 4, TYPE_GENERAL, OWNER_ME, 1);
        total_armies(&mut obs);
        let slate = slate_for(&obs, &[move_action(4, 4, 1), Action5 {
            pass_field: 1,
            row: 0,
            col: 0,
            dir: 0,
            is_half: 0,
        }]);
        assert!(slate.iter().all(|c| c.action.pass_field == 1), "{slate:?}");
        assert_eq!(slate[0].play, Play::Pass);
    }

    /// A build candidate reaches the afterstate as a build, and the castle
    /// masks are the only thing that judges it.
    #[test]
    fn a_crowded_build_is_masked_and_a_clear_one_is_not() {
        let mut obs = frame(100);
        put(&mut obs, 4, 4, TYPE_GENERAL, OWNER_ME, 20);
        // Two cells rich enough to build on: one beside the general, one clear
        // across the board.
        put(&mut obs, 4, 5, TYPE_PLAIN, OWNER_ME, 80);
        put(&mut obs, 0, 0, TYPE_PLAIN, OWNER_ME, 80);
        total_armies(&mut obs);

        let slate = slate_for(&obs, &[build_action(4, 5), build_action(0, 0)]);
        assert_eq!(slate[0].play, Play::Build(at(4, 5)));
        assert_eq!(slate[0].masked, Some(Mask::Crowding), "adjacent to our general");
        assert_eq!(slate[1].play, Play::Build(at(0, 0)));
        assert_eq!(slate[1].masked, None, "eight steps out, no surcharge");
    }

    /// The veto reads the board *after* our own candidate, which is the whole
    /// point of it: one frame, one enemy stack, and two of our moves that the
    /// same stack answers differently.
    ///
    /// Their 12 beside a garrison of 3 takes it — unless our 14 reinforces
    /// first. Note what the general itself cannot do about this: a move out of
    /// the general's own cell is the *destination* of their attack, so §02
    /// makes their attack a chase and resolves it first, against the garrison
    /// as it stands. Only a third tile can answer.
    #[test]
    fn the_veto_removes_the_move_that_hands_over_the_general() {
        let mut obs = frame(100);
        put(&mut obs, 4, 4, TYPE_GENERAL, OWNER_ME, 3);
        put(&mut obs, 4, 5, TYPE_PLAIN, OWNER_OPP, 12);
        put(&mut obs, 3, 4, TYPE_PLAIN, OWNER_ME, 14);
        total_armies(&mut obs);

        let away = move_action(3, 4, 0); // our 14 walks away from the general
        let hold = move_action(3, 4, 1); // our 14 lands on it instead
        let slate = slate_for(&obs, &[away, hold]);
        assert_eq!(slate[0].action, away);
        assert_eq!(slate[0].masked, Some(Mask::Refuted));
        assert_eq!(slate[1].action, hold);
        assert_eq!(slate[1].masked, None);
    }

    /// A quiet frame costs the veto nothing: no enemy stack stands beside the
    /// general, so nothing is refutable and no search runs.
    #[test]
    fn the_veto_stays_out_of_a_quiet_turn() {
        let mut obs = frame(100);
        put(&mut obs, 4, 4, TYPE_GENERAL, OWNER_ME, 12);
        put(&mut obs, 0, 8, TYPE_PLAIN, OWNER_OPP, 200);
        total_armies(&mut obs);
        let slate = slate_for(&obs, &[move_action(4, 4, 2)]);
        assert!(slate.iter().all(|c| c.masked.is_none()));
    }

    /// A slate of `n` candidates, the first `masked` of them filtered out.
    fn slate_of(scores: &[Score], masked: &[usize]) -> (Slate, Vec<Score>) {
        let mut slate = Slate::empty();
        for (rank, _) in scores.iter().enumerate() {
            let mut candidate = NOTHING;
            candidate.logit = -(rank as f32);
            candidate.masked = masked.contains(&rank).then_some(Mask::Refuted);
            slate.push(candidate);
        }
        let mut padded = scores.to_vec();
        padded.resize(TOP_K, Score::Skipped);
        (slate, padded)
    }

    /// The threshold's whole job: a rival that wins by less than one bin of
    /// the value head does not take the turn off the policy, and one that
    /// wins by more does.
    #[test]
    fn a_rival_needs_a_whole_bin_to_take_the_turn() {
        let under = RERANK_MIN_GAP * 0.9;
        let (slate, scored) = slate_of(&[Score::Value(0.0), Score::Value(under)], &[]);
        assert_eq!(live_pick(&slate, &scored), Some(0), "a sub-bin gap is a tie");
        assert_eq!(best_survivor(&slate, &scored), Some(1), "and it is still measured");

        let (slate, scored) =
            slate_of(&[Score::Value(0.0), Score::Value(RERANK_MIN_GAP)], &[]);
        assert_eq!(live_pick(&slate, &scored), Some(1), "exactly one bin clears");

        // A rival that is merely worse never had a claim either way.
        let (slate, scored) = slate_of(&[Score::Value(0.0), Score::Value(-0.5)], &[]);
        assert_eq!(live_pick(&slate, &scored), Some(0));
    }

    /// The gap is measured against the argmax, not against the runner-up: two
    /// rivals that each edge past the one below them do not add up to a claim
    /// on the turn.
    #[test]
    fn the_gap_is_measured_against_the_policys_own_move() {
        let step = RERANK_MIN_GAP * 0.6;
        let (slate, scored) =
            slate_of(&[Score::Value(0.0), Score::Value(step), Score::Value(2.0 * step)], &[]);
        // The best rival is 1.2 bins clear of the argmax, so it takes the turn.
        assert_eq!(live_pick(&slate, &scored), Some(2));

        // Halve the steps and the same shape no longer clears.
        let (slate, scored) = slate_of(
            &[Score::Value(0.0), Score::Value(step / 2.0), Score::Value(step)],
            &[],
        );
        assert_eq!(live_pick(&slate, &scored), Some(0));
    }

    /// A capture is not a marginal preference. A candidate that takes their
    /// general inside the model has no value to compare and needs none.
    #[test]
    fn a_winning_candidate_ignores_the_threshold() {
        let (slate, scored) = slate_of(&[Score::Value(0.99), Score::Wins], &[]);
        assert_eq!(live_pick(&slate, &scored), Some(1));
    }

    /// A masked argmax promotes the next survivor to default, and the
    /// threshold protects *that* — it is still the policy's highest-ranked
    /// allowed move, and overruling it on a sub-bin difference would be the
    /// same mistake the threshold exists to prevent.
    #[test]
    fn a_masked_argmax_promotes_the_next_survivor_to_default() {
        let under = RERANK_MIN_GAP * 0.1;
        let (slate, scored) =
            slate_of(&[Score::Value(0.5), Score::Value(0.0), Score::Value(under)], &[0]);
        assert_eq!(live_pick(&slate, &scored), Some(1), "the mask removed rank 0 outright");

        // The removed argmax's own value never re-enters the comparison, high
        // as it is: a filtered candidate is not played.
        let (slate, scored) =
            slate_of(&[Score::Value(0.5), Score::Value(0.0), Score::Value(RERANK_MIN_GAP)], &[0]);
        assert_eq!(live_pick(&slate, &scored), Some(2), "measured from the new default");

        // And with every candidate filtered the layer has no opinion at all.
        let (slate, scored) = slate_of(&[Score::Value(0.0), Score::Value(9.0)], &[0, 1]);
        assert_eq!(live_pick(&slate, &scored), None);
    }

    /// Without a baseline there is no gap, so the cutoff keeps the default
    /// rather than guessing that an unevaluated position was worse.
    #[test]
    fn an_unevaluated_baseline_never_gets_overruled() {
        let (slate, scored) = slate_of(&[Score::Skipped, Score::Value(9.0)], &[]);
        assert_eq!(live_pick(&slate, &scored), Some(0));
        assert_eq!(best_survivor(&slate, &scored), Some(1), "the preference is still recorded");
    }

    /// A win beats every number, a number beats an evaluation that never
    /// happened, and equal is not better — so ties fall to policy order.
    #[test]
    fn a_score_ranks_a_win_above_a_number_above_nothing() {
        assert!(Score::Wins.beats(Score::Value(9.0)));
        assert!(!Score::Value(9.0).beats(Score::Wins));
        assert!(!Score::Wins.beats(Score::Wins));
        assert!(Score::Value(0.2).beats(Score::Value(0.1)));
        assert!(!Score::Value(0.1).beats(Score::Value(0.1)));
        assert!(Score::Value(-9.0).beats(Score::Skipped));
        assert!(!Score::Skipped.beats(Score::Skipped));
    }

    /// The anytime guard: a turn that has already spent its budget starts no
    /// evaluation, so the reply degrades to the argmax the network named.
    #[test]
    fn a_spent_turn_starts_no_evaluation() {
        let tactics = Tactics::new(true);
        assert!(tactics.have_time(Instant::now()));
        let room = TURN_DEADLINE_MS - tactics.reserve_ms();
        assert!(!tactics.have_time(Instant::now() - Duration::from_millis(room + 5)));
        assert!(!tactics.have_time(Instant::now() - Duration::from_secs(10)));
    }

    /// The reserve is what an evaluation has actually cost, never below the
    /// floor. This is the fix for U4's measured overrun: a fixed 25 ms reserve
    /// let the loop start a 48 ms forward with 25 ms of turn left.
    #[test]
    fn the_reserve_is_measured_and_floored() {
        let mut tactics = Tactics::new(true);
        assert_eq!(tactics.reserve_ms(), RERANK_RESERVE_FLOOR_MS, "nothing measured yet");

        // A forward faster than the floor does not lower it.
        tactics.note_evaluation(3.0);
        assert_eq!(tactics.reserve_ms(), RERANK_RESERVE_FLOOR_MS);

        // A slow one raises it, and it is the worst that counts, not the last.
        tactics.note_evaluation(47.2);
        assert_eq!(tactics.reserve_ms(), 48, "rounded up, never down");
        tactics.note_evaluation(20.0);
        assert_eq!(tactics.reserve_ms(), 48);
    }

    /// A reserve that grew closes the window sooner — which is the whole point
    /// of measuring it, and the behaviour a slower host gets for free.
    #[test]
    fn a_bigger_reserve_stops_the_loop_earlier() {
        let mut tactics = Tactics::new(true);
        // Half the turn gone: room for a floor-sized evaluation, not for a
        // 70 ms one.
        let half = Instant::now() - Duration::from_millis(TURN_DEADLINE_MS / 2);
        assert!(tactics.have_time(half));
        tactics.note_evaluation(70.0);
        assert!(!tactics.have_time(half));
    }

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
