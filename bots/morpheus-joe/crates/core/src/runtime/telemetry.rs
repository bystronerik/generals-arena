//! Per-turn telemetry, in the shape the Python sibling's probe emits.
//!
//! The Python bot is traced from outside: `arena.instrument.runner` constructs
//! `Agent` in-process and samples `bots/morpheus/probe.py` after every move.
//! Nothing about that works for a subprocess binary, so this bot carries its
//! own probe — plan §7 said it would get one "later", and M6 is later: the
//! milestone's latency and first-move claims are unmeasurable without it.
//!
//! Two properties are copied from the Python path rather than invented:
//!
//! * **The same keys.** Every field here is a field of `probe.py`'s `extras`,
//!   so one reducer reads both bots. Two additions ride along and are ignorable
//!   by anything that does not know them: `components`, the per-component
//!   milliseconds the shipped `offline_p99_ms` table is fitted from, and the
//!   three startup costs on the first line.
//! * **Buffered to the end.** The runner writes its trace in a `finally`, for
//!   the reason stated there: a per-turn flush would put file IO on the move
//!   path, which is exactly what must not perturb the game being measured.
//!   Telemetry that changes the measurement is worse than none.
//!
//! Armed only by `MORPHEUS_JOE_TRACE`. Unset — which is every rated game — this
//! costs one `Option` check per turn and touches no disk.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use crate::runtime::{TurnMetrics, COST_COMPONENTS, FORWARD_CONSUMERS};

// Renamed with the fork, and not cosmetically: the N6 round runs this bot
// and morpheus-rs in the same tournament, and one env var for both would
// have two seats appending to one trace file with no way to tell them
// apart afterwards.
pub const TRACE_ENV: &str = "MORPHEUS_JOE_TRACE";

pub struct Trace {
    path: PathBuf,
    lines: Vec<String>,
    /// A JSON fragment describing the configuration, emitted on the first line.
    /// A sweep writes one trace per candidate config, and a trace that does not
    /// say which knobs produced it is a measurement waiting to be misfiled.
    header: Option<String>,
}

impl Trace {
    /// A trace when `MORPHEUS_JOE_TRACE` names a path, else nothing.
    ///
    /// `%p` in the path becomes this process's id. A tournament runs many
    /// seats at once from one environment, and without it they would all write
    /// the same file — which looks like a trace and is a race.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var(TRACE_ENV).ok()?;
        if path.is_empty() {
            return None;
        }
        let path = path.replace("%p", &std::process::id().to_string());
        Some(Self {
            path: PathBuf::from(path),
            lines: Vec::with_capacity(1024),
            header: None,
        })
    }

    /// Record the configuration this process is playing, for the first line.
    pub fn set_header(&mut self, fragment: String) {
        self.header = Some(fragment);
    }

    /// One line per turn. `startup` is `(load_ms, warmup_ms, init_ms)` and is
    /// written on the first line only — it describes the process, not the move.
    pub fn record(&mut self, turn: i64, m: &TurnMetrics, startup: Option<(f64, f64, f64)>) {
        let mut line = String::with_capacity(1024);
        let _ = write!(line, "{{\"t\":{turn}");
        for (key, value) in [
            ("move_ms", m.move_ms),
            ("search_iters", m.completed_simulations),
            ("completed_simulations", m.completed_simulations),
            ("forward_equivalents", m.forward_equivalents),
            ("belief_ess", m.belief_ess),
            ("recovery", m.recovery),
            ("tree_size", m.tree_size),
            ("cost_belief_ms", m.cost_belief_ms),
            ("cost_root_ms", m.cost_root_ms),
            ("cost_search_ms", m.cost_search_ms),
            ("cost_reply_ms", m.cost_reply_ms),
            ("belief_plus_root_ok", m.belief_plus_root_ok),
            ("proposal_n_unique_info_keys", m.proposal_n_unique_info_keys),
            (
                "proposal_n_unique_policy_inputs",
                m.proposal_n_unique_policy_inputs,
            ),
            (
                "proposal_n_singleton_particles",
                m.proposal_n_singleton_particles,
            ),
            ("proposal_n_policy_batches", m.proposal_n_policy_batches),
            ("search_selection_calls", m.component_calls[9]),
            ("search_leaf_batch_calls", m.component_calls[5]),
            ("search_enemy_prior_calls", m.component_calls[6]),
            ("root_pass_prior_milli", m.root_pass_prior_milli),
            ("root_top_action", m.root_top_action),
            ("root_top_prior_milli", m.root_top_prior_milli),
            ("chosen_action", m.chosen_action),
            ("chosen_is_pass", m.chosen_is_pass),
            ("policy_fallback_is_pass", m.policy_fallback_is_pass),
            ("root_legal_nonpass", m.root_legal_nonpass),
            ("has_root_result", m.has_root_result),
            ("nn_top_action", m.nn_top_action),
            ("nn_top_prior_milli", m.nn_top_prior_milli),
            ("chosen_matches_nn_top", m.chosen_matches_nn_top),
            ("chosen_in_nn_top3", m.chosen_in_nn_top3),
            ("enemy_visible", m.enemy_visible),
        ] {
            let _ = write!(line, ",\"{key}\":{value}");
        }
        let _ = write!(line, ",\"fallback_level\":\"{}\"", m.fallback_level);
        // Calls for all ten, not the three `probe.py` happens to declare. M6
        // could normalize only those three per call and it mattered — the
        // per-turn total charges a faster bot for the extra simulations it fit,
        // so `enemy_prior_batch` read 0.9x per turn and 1.6x per call. M7 fits
        // knobs from these numbers and a per-turn total is the wrong unit for
        // anything that runs once per simulation.
        line.push_str(",\"calls\":{");
        for (i, name) in COST_COMPONENTS.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let _ = write!(line, "\"{name}\":{}", m.component_calls[i]);
        }
        line.push('}');
        // Which consumer spent the forwards, not just how many there were.
        // `forward_equivalents` is one number for four callers, and the joe-net
        // port's N0 has to know how the budget divides before it can say what
        // survives a forward that costs four times as much
        // (docs/bots/morpheus-rs/joe-net-plan.md N0.2).
        line.push_str(",\"forward_by_consumer\":{");
        for (i, name) in FORWARD_CONSUMERS.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let _ = write!(line, "\"{name}\":{}", m.forward_by_consumer[i]);
        }
        line.push('}');
        line.push_str(",\"components\":{");
        for (i, name) in COST_COMPONENTS.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            // Six decimals: `hashing` runs at a microsecond and the shipped
            // table quotes it at 0.001 ms, so anything coarser prints zero.
            let _ = write!(line, "\"{name}\":{:.6}", m.component_ms[i]);
        }
        line.push('}');
        if let Some((load_ms, warmup_ms, init_ms)) = startup {
            let _ = write!(
                line,
                ",\"load_ms\":{load_ms:.3},\"warmup_ms\":{warmup_ms:.3},\"init_ms\":{init_ms:.3}"
            );
            if let Some(header) = &self.header {
                let _ = write!(line, ",\"config\":{header}");
            }
        }
        line.push('}');
        self.lines.push(line);
    }

    /// Write the buffer out. Failures are reported and otherwise ignored:
    /// losing a trace is a lost measurement, never a lost game.
    pub fn flush(&self) {
        if self.lines.is_empty() {
            return;
        }
        if let Some(parent) = self.path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let mut body = self.lines.join("\n");
        body.push('\n');
        if let Err(err) = fs::write(&self.path, body) {
            eprintln!("[morpheus-rs] could not write {}: {err}", self.path.display());
        }
    }
}

/// This process's thread count, when the platform will say.
///
/// Plan §10 wants the Rust analog of
/// `test_play_and_calibration_pin_the_same_thread_count`: the bot is
/// single-threaded by construction, and a dependency that quietly spawns a
/// pool would break the one assumption every latency number here rests on.
///
/// Linux answers through `/proc/self/status`, which is the platform the judge
/// runs and the only one that has to answer. macOS has no `/proc`, and reading
/// its thread count needs `libc` — a dependency this crate does not have and
/// will not take for a check that does not run where it matters. `None` means
/// "not asked", never "passed".
pub fn thread_count() -> Option<usize> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Threads:") {
            return rest.trim().parse().ok();
        }
    }
    None
}
