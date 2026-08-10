//! The runtime controller: deadlines, admission, degradation, telemetry.
//!
//! Port of `bots/morpheus/runtime.py`. rewrite-plan §7 files this as
//! "translation only, no gain" — and that is the point. The controller *is* the
//! deadline behaviour, so it is ported faithfully rather than improved, and its
//! telemetry schema is kept identical so the existing analysis tooling reads
//! both bots.
//!
//! **M4 found the one number here that is a lie by construction.** The
//! controller charges `filter_step` and the `recover_belief` it may trigger to
//! a single component, `particle_transitions`, so its p99 describes a cost
//! distribution that is bimodal by a factor of a hundred (1.1 ms filtered,
//! 127 ms recovered on the Python). That accounting is reproduced, because
//! changing it would silently invalidate every recorded trace — but M7 should
//! reserve for the two paths separately.

use std::time::Instant;

use crate::board::action::{decode_action, encode_action, PASS_INDEX};
use crate::belief::{
    ess, filter_step, initialize_belief, Action5, BeliefConfig, BeliefState, PASS_ACTION,
};
use crate::board::memory::{update_memory, VisibleMemory};
use crate::proposal::{propose_enemy_actions, ProposalPolicy, ProposalTelemetry};
use crate::recovery::recover_belief;
use crate::support::rng::SharedRng;
use crate::search::{
    EnemyPriorRequest, PendingPath, SearchConfig, SearchController, SearchEvaluator, Selection,
};
use crate::tactics::{
    constrain_nn_action, default_shaping_log_clip, enemy_is_visible, play_mask,
    DEFAULT_SHAPING_FLOOR_FRAC, DEFAULT_SHAPING_LAMBDA, OSCILLATION_HISTORY,
};
use crate::io::wire::Observation;

/// Named cost components. Admission keeps a separate rolling p99 for each.
pub const COST_COMPONENTS: [&str; 10] = [
    "belief_tensor",
    "belief_proposal",
    "particle_transitions",
    "hashing",
    "root_inference",
    "leaf_batch",
    "enemy_prior_batch",
    "backup",
    "reply",
    "selection",
];

pub const FORWARD_CONSUMERS: [&str; 4] = ["belief_proposal", "root", "enemy_prior", "leaf_batch"];

pub const DEFAULT_NORMAL_DEADLINE_MS: f64 = 125.0;
pub const DEFAULT_RESERVE_MS: f64 = 25.0;
pub const DEFAULT_FIRST_MOVE_LIMIT_MS: f64 = 8500.0;
pub const DEFAULT_TARGET_SIMULATIONS: usize = 32;
pub const DEFAULT_MIN_SIMULATIONS: usize = 8;
pub const DEFAULT_PENDING_LEAF_BATCH: usize = 4;
pub const DEFAULT_MAX_FORWARD_EQUIVALENTS: usize = 113;
pub const DEFAULT_ADMISSION_GUARD_MS: f64 = 10.0;
pub const DEFAULT_MAX_TREE_NODES: usize = 4096;
pub const DEFAULT_RESIDENT_MEMORY_TARGET_MB: f64 = 256.0;
pub const DEFAULT_P99_WINDOW: usize = 64;
pub const DEFAULT_MAX_PROPOSAL_BATCH: usize = 64;
pub const DEFAULT_SEARCH_DEPTH: usize = 16;

/// Offline qualification p99 placeholders, in the Python's declaration order.
pub const DEFAULT_OFFLINE_P99_MS: [(&str, f64); 10] = [
    ("belief_tensor", 5.0),
    ("belief_proposal", 40.0),
    ("particle_transitions", 20.0),
    ("hashing", 1.0),
    ("root_inference", 8.0),
    ("leaf_batch", 30.0),
    ("enemy_prior_batch", 8.0),
    ("backup", 2.0),
    ("reply", 1.0),
    ("selection", 2.0),
];

pub fn default_offline_p99(name: &str) -> f64 {
    DEFAULT_OFFLINE_P99_MS
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| *value)
        .unwrap_or(1.0)
}

pub fn component_index(name: &str) -> Option<usize> {
    COST_COMPONENTS.iter().position(|&key| key == name)
}

fn consumer_index(name: &str) -> usize {
    FORWARD_CONSUMERS
        .iter()
        .position(|&key| key == name)
        .expect("unknown forward consumer")
}

/// Which degradation band produced the committed action.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum FallbackLevel {
    /// No root result.
    Pass,
    /// Zero completed simulations.
    Policy,
    /// Retired band, kept for old-trace schema compatibility. Three selectors
    /// flipping by simulation count measured 220 flips in a 573-turn game.
    Visit,
    /// One or more completed simulations.
    Average,
}

impl FallbackLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            FallbackLevel::Pass => "pass",
            FallbackLevel::Policy => "policy",
            FallbackLevel::Visit => "visit",
            FallbackLevel::Average => "average",
        }
    }
}

/// Nearest-rank empirical 99th percentile, one-indexed rank `ceil(0.99 n)`.
pub fn nearest_rank_p99(samples: &[f64]) -> Result<f64, String> {
    if samples.is_empty() {
        return Err("nearest-rank p99 needs at least one sample".to_string());
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (0.99 * samples.len() as f64).ceil() as usize;
    Ok(ordered[rank - 1])
}

// ---------------------------------------------------------------- the clock

/// Seconds, monotonic. `advance` exists so a fake clock can be charged the
/// forecast instead of the wall time, which is what makes an admission
/// sequence reproducible across two languages.
pub trait Clock {
    fn now(&self) -> f64;
    fn advance(&mut self, _ms: f64) {}
    fn is_fake(&self) -> bool {
        false
    }
}

pub struct MonotonicClock {
    start: Instant,
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Clock for MonotonicClock {
    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

/// Injectable monotonic clock for deadline tests, in seconds.
#[derive(Default)]
pub struct FakeClock {
    pub t: f64,
}

impl Clock for FakeClock {
    fn now(&self) -> f64 {
        self.t
    }

    fn advance(&mut self, ms: f64) {
        self.t += ms / 1000.0;
    }

    fn is_fake(&self) -> bool {
        true
    }
}

// ------------------------------------------------------------------- config

#[derive(Clone)]
pub struct RuntimeConfig {
    pub normal_deadline_ms: f64,
    pub reserve_ms: f64,
    pub first_move_limit_ms: f64,
    pub target_simulations: usize,
    pub min_simulations: usize,
    pub pending_leaf_batch: usize,
    pub max_forward_equivalents: usize,
    pub admission_guard_ms: f64,
    pub max_tree_nodes: usize,
    pub resident_memory_target_mb: f64,
    pub p99_window: usize,
    pub offline_p99_ms: Vec<(String, f64)>,
    pub n_particles: usize,
    pub max_proposal_batch: usize,
    pub search_depth: usize,
    /// Stop widening when the forecast simulation count falls below this.
    pub widen_freeze_below: usize,
    pub shaping_lambda_pre_contact: f64,
    pub shaping_lambda_post_contact: f64,
    pub shaping_log_clip: f64,
    pub shaping_floor_frac: f64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            normal_deadline_ms: DEFAULT_NORMAL_DEADLINE_MS,
            reserve_ms: DEFAULT_RESERVE_MS,
            first_move_limit_ms: DEFAULT_FIRST_MOVE_LIMIT_MS,
            target_simulations: DEFAULT_TARGET_SIMULATIONS,
            min_simulations: DEFAULT_MIN_SIMULATIONS,
            pending_leaf_batch: DEFAULT_PENDING_LEAF_BATCH,
            max_forward_equivalents: DEFAULT_MAX_FORWARD_EQUIVALENTS,
            admission_guard_ms: DEFAULT_ADMISSION_GUARD_MS,
            max_tree_nodes: DEFAULT_MAX_TREE_NODES,
            resident_memory_target_mb: DEFAULT_RESIDENT_MEMORY_TARGET_MB,
            p99_window: DEFAULT_P99_WINDOW,
            offline_p99_ms: DEFAULT_OFFLINE_P99_MS
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
            n_particles: 64,
            max_proposal_batch: DEFAULT_MAX_PROPOSAL_BATCH,
            search_depth: DEFAULT_SEARCH_DEPTH,
            widen_freeze_below: 16,
            shaping_lambda_pre_contact: DEFAULT_SHAPING_LAMBDA,
            shaping_lambda_post_contact: DEFAULT_SHAPING_LAMBDA,
            shaping_log_clip: default_shaping_log_clip(),
            shaping_floor_frac: DEFAULT_SHAPING_FLOOR_FRAC,
        }
    }
}

impl RuntimeConfig {
    pub fn offline(&self, name: &str) -> Option<f64> {
        self.offline_p99_ms
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| *value)
    }
}

/// One component's estimator seed, with a finite fallback.
///
/// A non-finite seed makes every admission comparison false, so the component
/// never runs again. Falling back is strictly better than passing it through.
fn offline_seed_ms(config: &RuntimeConfig, name: &str) -> f64 {
    let fallback = default_offline_p99(name);
    match config.offline(name) {
        Some(value) if value.is_finite() && value >= 0.0 => value,
        Some(_) => fallback,
        None => fallback,
    }
}

/// Rolling nearest-rank p99 with a deterministic warm-up.
///
/// Before `W` local samples exist the forecast is the maximum of the offline
/// qualification p99 and every observed local sample; after that it is the
/// nearest-rank empirical p99 of the bounded window.
pub struct NearestRankP99Estimator {
    pub window: usize,
    pub offline_p99_ms: f64,
    samples: std::collections::VecDeque<f64>,
}

impl NearestRankP99Estimator {
    pub fn new(window: usize, offline_p99_ms: f64) -> Result<Self, String> {
        if window < 1 {
            return Err("p99 window must be >= 1".to_string());
        }
        if !offline_p99_ms.is_finite() {
            // A non-finite seed poisons every forecast and silently locks the
            // component out of admission forever.
            return Err("offline_p99_ms seed must be finite".to_string());
        }
        Ok(Self {
            window,
            offline_p99_ms,
            samples: std::collections::VecDeque::with_capacity(window),
        })
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn warmed_up(&self) -> bool {
        self.samples.len() >= self.window
    }

    pub fn observe(&mut self, ms: f64) {
        if self.samples.len() == self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(ms);
    }

    pub fn forecast(&self) -> f64 {
        if self.samples.is_empty() {
            return self.offline_p99_ms;
        }
        if !self.warmed_up() {
            let local = self
                .samples
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            return self.offline_p99_ms.max(local);
        }
        let values: Vec<f64> = self.samples.iter().copied().collect();
        nearest_rank_p99(&values).expect("a warmed-up window is non-empty")
    }
}

// ------------------------------------------------------------------ metrics

/// Passive per-turn counters. The probe reads these; they never change play.
#[derive(Clone, Default)]
pub struct TurnMetrics {
    pub move_ms: i64,
    pub completed_simulations: i64,
    pub forward_equivalents: i64,
    pub forward_by_consumer: [i64; 4],
    /// `ess * 1000`.
    pub belief_ess: i64,
    pub recovery: i64,
    pub tree_size: i64,
    pub fallback_level: &'static str,
    pub cost_belief_ms: i64,
    pub cost_root_ms: i64,
    pub cost_search_ms: i64,
    pub cost_reply_ms: i64,
    pub belief_plus_root_ok: i64,
    pub component_ms: [f64; 10],
    /// Real component runs this turn, excluding estimator padding.
    pub component_calls: [i64; 10],
    pub proposal_n_particles: i64,
    pub proposal_n_singleton_particles: i64,
    pub proposal_n_unique_info_keys: i64,
    pub proposal_n_unique_policy_inputs: i64,
    pub proposal_n_policy_batches: i64,
    pub root_pass_prior_milli: i64,
    pub root_top_action: i64,
    pub root_top_prior_milli: i64,
    pub chosen_action: i64,
    pub chosen_is_pass: i64,
    pub policy_fallback_is_pass: i64,
    pub root_legal_nonpass: i64,
    pub has_root_result: i64,
    /// "Who's deciding": the same fields measured on the *unshaped* network
    /// prior, so agreement between the network's own top action and what the
    /// bot played is observable without re-running inference.
    pub nn_top_action: i64,
    pub nn_top_prior_milli: i64,
    pub chosen_matches_nn_top: i64,
    pub chosen_in_nn_top3: i64,
    pub enemy_visible: i64,
}

impl TurnMetrics {
    fn fresh() -> Self {
        Self {
            fallback_level: FallbackLevel::Pass.as_str(),
            root_top_action: -1,
            chosen_action: -1,
            nn_top_action: -1,
            ..Default::default()
        }
    }
}

/// Derive the integer probe fields from the legal-normalized root prior.
///
/// `prior` is the shaped prior the bot searched on; `unshaped` is the raw
/// network prior before the blend. Comparing the chosen action against the
/// *unshaped* top is what makes "who's deciding" answerable.
#[allow(clippy::too_many_arguments)]
pub fn prior_probe_fields(
    metrics: &mut TurnMetrics,
    prior: Option<&[f64]>,
    mask: Option<&[bool]>,
    chosen: Action5,
    policy_fallback: Option<Action5>,
    has_root_result: bool,
    unshaped: Option<&[f64]>,
    enemy_visible: bool,
) {
    let chosen_idx = encode_action(chosen);
    metrics.chosen_action = chosen_idx as i64;
    metrics.chosen_is_pass = (chosen == PASS_ACTION || chosen_idx == PASS_INDEX) as i64;
    metrics.has_root_result = has_root_result as i64;
    metrics.enemy_visible = enemy_visible as i64;
    if let Some(fallback) = policy_fallback {
        metrics.policy_fallback_is_pass =
            (fallback == PASS_ACTION || encode_action(fallback) == PASS_INDEX) as i64;
    }
    let (prior, mask) = match (prior, mask) {
        (Some(prior), Some(mask)) if prior.len() == mask.len() => (prior, mask),
        _ => return,
    };
    if mask[PASS_INDEX] {
        metrics.root_pass_prior_milli = (prior[PASS_INDEX] * 1000.0).round() as i64;
    }
    let live = mask.iter().filter(|&&m| m).count() as i64;
    metrics.root_legal_nonpass = live - mask[PASS_INDEX] as i64;
    let top = argmax_masked(prior, mask);
    metrics.root_top_action = top as i64;
    metrics.root_top_prior_milli = (prior[top] * 1000.0).round() as i64;

    let unshaped = match unshaped {
        Some(unshaped) if unshaped.len() == mask.len() && mask.iter().any(|&m| m) => unshaped,
        _ => return,
    };
    let nn_top = argmax_masked(unshaped, mask);
    metrics.nn_top_action = nn_top as i64;
    metrics.nn_top_prior_milli = (unshaped[nn_top] * 1000.0).round() as i64;
    metrics.chosen_matches_nn_top = (chosen_idx == nn_top) as i64;
    // `np.argpartition(-scored, k-1)[:k]` — the *set* of the top k, in no
    // particular order, which is all the membership test needs.
    let k = 3.min(mask.iter().filter(|&&m| m).count());
    let mut scored: Vec<(usize, f64)> = (0..mask.len())
        .map(|i| (i, if mask[i] { unshaped[i] } else { -1.0 }))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    metrics.chosen_in_nn_top3 = scored[..k].iter().any(|&(i, _)| i == chosen_idx) as i64;
}

/// `int(np.argmax(np.where(mask, values, -1.0)))` — first maximum wins.
fn argmax_masked(values: &[f64], mask: &[bool]) -> usize {
    let score = |i: usize| if mask[i] { values[i] } else { -1.0 };
    let mut best = 0usize;
    for i in 1..values.len() {
        if score(i) > score(best) {
            best = i;
        }
    }
    best
}

/// Highest-prior playable action; pass only when it is the sole masked action.
pub fn highest_prior_legal(prior: &[f64], mask: &[bool]) -> Action5 {
    if !mask.iter().any(|&m| m) {
        return PASS_ACTION;
    }
    decode_action(argmax_masked(prior, mask)).unwrap_or(PASS_ACTION)
}

/// The deterministic degradation path.
///
/// `has_root_result` false means root inference never completed, which is
/// distinct from zero completed simulations — that case still has a policy
/// fallback. One selector covers every simulated turn: at low simulation counts
/// the average strategy is dominated by the prior, so it degrades to the old
/// visit/prior band without switching decision rules turn to turn.
pub fn select_degraded_action(
    completed_simulations: u64,
    has_root_result: bool,
    policy_fallback: Option<Action5>,
    search: &SearchController,
) -> (Action5, FallbackLevel) {
    if !has_root_result {
        return (PASS_ACTION, FallbackLevel::Pass);
    }
    if completed_simulations == 0 {
        return (
            policy_fallback.unwrap_or(PASS_ACTION),
            FallbackLevel::Policy,
        );
    }
    let action = search.best_action().unwrap_or(PASS_ACTION);
    if action == PASS_ACTION {
        if let Some(fallback) = policy_fallback {
            if fallback != PASS_ACTION {
                return (fallback, FallbackLevel::Policy);
            }
        }
    }
    (action, FallbackLevel::Average)
}

// --------------------------------------------------------------- controller

/// Owns first-move setup, the normal-turn work order, and admission control.
pub struct RuntimeController {
    pub seat: usize,
    pub h: usize,
    pub w: usize,
    pub config: RuntimeConfig,
    pub clock: Box<dyn Clock>,
    pub rng: SharedRng,
    /// Fixed per-component forecasts, overriding the estimators.
    pub fixed_forecasts_ms: Option<Vec<(String, f64)>>,
    /// When true, each admitted component advances the injected clock by its
    /// forecast. Production uses real elapsed time instead.
    pub charge_fixed_forecasts: bool,
    /// Sample the policy net for the belief proposal. `deployment.json` ships
    /// this off; see `DeploymentConfig::use_policy_proposal`.
    pub use_policy_proposal: bool,
    pub search: SearchController,
    pub memory: Option<VisibleMemory>,
    pub belief: Option<BeliefState>,
    pub metrics: TurnMetrics,
    estimators: Vec<NearestRankP99Estimator>,
    setup_done: bool,
    last_action: Action5,
    recent_actions: std::collections::VecDeque<Action5>,
    pending_recovery: bool,
    turn_start: f64,
    proposal_telemetry: ProposalTelemetry,
    forward_equivalents: i64,
    forward_by_consumer: [i64; 4],
}

impl RuntimeController {
    pub fn new(
        seat: usize,
        h: usize,
        w: usize,
        config: RuntimeConfig,
        clock: Box<dyn Clock>,
        rng: SharedRng,
    ) -> Self {
        let estimators = COST_COMPONENTS
            .iter()
            .map(|name| {
                NearestRankP99Estimator::new(config.p99_window, offline_seed_ms(&config, name))
                    .expect("the seed is finite by construction of offline_seed_ms")
            })
            .collect();
        let search = SearchController::new(
            seat,
            SearchConfig {
                depth: config.search_depth,
                pending_batch: config.pending_leaf_batch,
                max_nodes: config.max_tree_nodes,
                n_particles: config.n_particles,
                ..SearchConfig::default()
            },
            rng.handle(),
        );
        Self {
            seat,
            h,
            w,
            config,
            clock,
            rng,
            fixed_forecasts_ms: None,
            charge_fixed_forecasts: false,
            use_policy_proposal: false,
            search,
            memory: None,
            belief: None,
            metrics: TurnMetrics::fresh(),
            estimators,
            setup_done: false,
            last_action: PASS_ACTION,
            recent_actions: std::collections::VecDeque::with_capacity(OSCILLATION_HISTORY),
            pending_recovery: false,
            turn_start: 0.0,
            proposal_telemetry: ProposalTelemetry::default(),
            forward_equivalents: 0,
            forward_by_consumer: [0; 4],
        }
    }

    // ------------------------------------------------------------- the clock

    fn now(&self) -> f64 {
        self.clock.now()
    }

    fn deadline_for_turn(&self, first_move: bool) -> f64 {
        let budget_ms = if first_move {
            self.config.first_move_limit_ms
        } else {
            self.config.normal_deadline_ms
        };
        self.turn_start + budget_ms / 1000.0
    }

    pub fn forecast_ms(&self, component: &str) -> f64 {
        if let Some(fixed) = &self.fixed_forecasts_ms {
            if let Some((_, value)) = fixed.iter().find(|(key, _)| key == component) {
                return *value;
            }
        }
        let at = component_index(component).expect("unknown cost component");
        self.estimators[at].forecast()
    }

    /// True when the forecast p99 plus the guard fits before the deadline.
    pub fn can_admit(&self, component: &str, deadline: f64) -> bool {
        let remaining_ms = (deadline - self.now()) * 1000.0;
        remaining_ms >= self.forecast_ms(component) + self.config.admission_guard_ms
    }

    fn observe(&mut self, component: &str, ms: f64, count_call: bool) {
        if let Some(at) = component_index(component) {
            self.estimators[at].observe(ms);
            self.metrics.component_ms[at] += ms;
            if count_call {
                self.metrics.component_calls[at] += 1;
            }
        }
    }

    /// Advance an injected clock by the component forecast (test mode).
    fn charge(&mut self, component: &str) {
        if !self.charge_fixed_forecasts {
            return;
        }
        let advance = self.forecast_ms(component);
        self.clock.advance(advance);
    }

    /// Time one block: real elapsed, or the forecast under a charged clock.
    fn timed<T>(&mut self, component: &str, body: impl FnOnce(&mut Self) -> T) -> T {
        let t0 = self.now();
        self.charge(component);
        let result = body(self);
        let elapsed_ms = if self.charge_fixed_forecasts {
            self.forecast_ms(component)
        } else {
            (self.now() - t0) * 1000.0
        };
        self.observe(component, elapsed_ms, true);
        result
    }

    // -------------------------------------------------------- the first move

    /// Load-time work that fits the first-frame grace window.
    ///
    /// Network load and warm-up belong to the caller. This allocates memory,
    /// the belief, and bounded tree storage, and classifies the initial terrain
    /// through the memory update.
    pub fn first_move_setup(&mut self, obs: &Observation) {
        if self.setup_done {
            return;
        }
        self.memory = Some(update_memory(&VisibleMemory::empty(self.h, self.w), obs));
        let belief_cfg = BeliefConfig {
            n_particles: self.config.n_particles,
            ..Default::default()
        };
        let mut rng = self.rng.handle();
        self.belief = Some(
            initialize_belief(obs, self.seat, &mut rng, belief_cfg).unwrap_or(
                // Tiny boards in protocol tests cannot place an enemy general.
                BeliefState {
                    seat: self.seat,
                    particles: Vec::new(),
                    config: belief_cfg,
                    collapsed: true,
                },
            ),
        );
        self.search.tree.clear();
        self.setup_done = true;
    }

    // ------------------------------------------------------------- one turn

    /// Run one turn. Pass is stored before any optional work.
    pub fn decide<E: SearchEvaluator + ?Sized>(
        &mut self,
        evaluator: &mut E,
        obs: &Observation,
    ) -> Action5 {
        self.turn_start = self.now();
        let first_move = !self.setup_done;
        let mut deadline = self.deadline_for_turn(first_move);

        // 1. Protocol-safe fallback.
        let mut policy_fallback: Option<Action5> = None;
        let mut has_root_result = false;
        let mut root_prior_for_probe: Option<Vec<f64>> = None;
        let mut root_mask_for_probe: Option<Vec<bool>> = None;
        // The first move allocates; later turns track the update.
        let mut belief_update_ok = first_move;
        self.search.tree.completed_simulations = 0;
        self.forward_equivalents = 0;
        self.forward_by_consumer = [0; 4];
        self.metrics = TurnMetrics::fresh();
        let mut recovery_flag = 0i64;
        // A skipped root inference must not report last turn's network prior.
        evaluator.clear_last_unshaped_prior();

        if first_move {
            let admitted = self.can_admit("belief_tensor", deadline);
            if admitted {
                let obs = obs.clone();
                self.timed("belief_tensor", |this| this.first_move_setup(&obs));
            } else {
                // Must still allocate: this is the first frame.
                self.first_move_setup(obs);
            }
            belief_update_ok = true;
            deadline = self.deadline_for_turn(true);
        } else {
            let memory = update_memory(
                self.memory.as_ref().expect("memory exists after setup"),
                obs,
            );
            self.memory = Some(memory);
            let have_particles = self.belief.as_ref().map_or(false, |b| b.n() > 0);
            if have_particles {
                let remaining_ms = (deadline - self.now()) * 1000.0;
                let guard = self.config.admission_guard_ms;
                let mut admit_belief = self.can_admit("belief_proposal", deadline)
                    && self.can_admit("particle_transitions", deadline);
                if !admit_belief {
                    // Warm-up lockout: one high-diversity spike makes the
                    // forecast `max(sample)` until the window fills, so the
                    // belief would never remeasure. Probe again while the
                    // offline p99 still fits.
                    let prop = &self.estimators[component_index("belief_proposal").unwrap()];
                    let tr = &self.estimators[component_index("particle_transitions").unwrap()];
                    admit_belief = !prop.warmed_up()
                        && remaining_ms >= prop.offline_p99_ms + guard
                        && remaining_ms >= tr.offline_p99_ms + guard;
                }
                if admit_belief {
                    self.run_belief_update(evaluator, obs);
                    belief_update_ok = true;
                } else {
                    // Keep the last valid set, lower the reported ESS, defer
                    // recovery to a turn that can afford it.
                    recovery_flag = 1;
                    self.pending_recovery = true;
                }
            } else if self.pending_recovery && self.can_admit("belief_proposal", deadline) {
                self.pending_recovery = false;
            }
        }

        let belief = self.belief.clone().unwrap_or(BeliefState {
            seat: self.seat,
            particles: Vec::new(),
            config: BeliefConfig {
                n_particles: self.config.n_particles,
                ..Default::default()
            },
            collapsed: true,
        });
        let memory = self.memory.clone().expect("memory exists by now");

        // 3–5. Legal masks, root network evaluation, policy fallback.
        // The Python assigns `action = PASS` here and again from the policy
        // fallback; both writes are dead, because step 8 always recomputes the
        // committed action from the same inputs. The fallback is still built
        // and kept, since `select_degraded_action` reads it.
        if self.can_admit("root_inference", deadline) {
            let last_action = self.last_action;
            let root_exists = self.search.tree.root.is_some();
            let obs_owned = obs.clone();
            let memory_owned = memory.clone();
            let belief_owned = belief.clone();
            self.timed("root_inference", |this| {
                if first_move || !root_exists {
                    this.search
                        .ensure_root(evaluator, &obs_owned, &memory_owned, &belief_owned);
                } else {
                    this.search.reuse_or_reset(
                        evaluator,
                        last_action,
                        &obs_owned,
                        &memory_owned,
                        &belief_owned,
                    );
                }
            });
            has_root_result = true;
            self.forward_equivalents += 1;
            self.forward_by_consumer[consumer_index("root")] += 1;
            let prior_full = match self.search.last_root_prior.clone() {
                Some(prior) => prior,
                None => {
                    let (prior, _) = evaluator.evaluate(obs, &memory, &belief, true, true);
                    self.forward_equivalents += 1;
                    self.forward_by_consumer[consumer_index("root")] += 1;
                    prior
                }
            };
            let mask = play_mask(obs, &memory, None);
            policy_fallback = Some(highest_prior_legal(&prior_full, &mask));
            root_prior_for_probe = Some(prior_full);
            root_mask_for_probe = Some(mask.to_vec());
            // Hashing already ran inside ensure_root / reuse_or_reset. Record
            // elapsed rather than a forecast so the estimator stays finite.
            self.timed("hashing", |_| {});
        }

        // 6–7. Search, in batches of up to `pending_leaf_batch`.
        let search_t0 = self.now();
        if has_root_result && belief.n() > 0 {
            self.run_search(evaluator, &belief, deadline);
        }
        self.metrics.cost_search_ms = ((self.now() - search_t0) * 1000.0).round() as i64;
        if self.charge_fixed_forecasts {
            let at = component_index("leaf_batch").unwrap();
            self.metrics.cost_search_ms = self.metrics.component_ms[at].round() as i64;
        }

        // 8. Serialize the best available action along the degradation path.
        let (mut action, mut level) = select_degraded_action(
            self.search.tree.completed_simulations,
            has_root_result,
            policy_fallback,
            &self.search,
        );
        // Hard rules only — keep the search/network choice otherwise so
        // training stays influential. Cheap enemy scouting lives in the root
        // prior reshape.
        let constrained = constrain_nn_action(
            obs,
            &memory,
            action,
            Some(self.last_action),
            &self.recent_actions.iter().copied().collect::<Vec<_>>(),
            self.search.last_root_prior.as_deref(),
            Some(&belief),
        );
        if constrained != action {
            action = constrained;
            level = FallbackLevel::Policy;
        }

        // Reply cost accounting: wall elapsed, since serialization is at the
        // caller.
        if self.can_admit("reply", deadline) {
            self.timed("reply", |_| {});
        }

        self.last_action = action;
        if action[0] == 0 {
            if self.recent_actions.len() == OSCILLATION_HISTORY {
                self.recent_actions.pop_front();
            }
            self.recent_actions.push_back(action);
        }
        evaluator.set_previous_action(Some(action));
        // Finite samples for components that may not run every turn.
        let enemy_prior_at = component_index("enemy_prior_batch").unwrap();
        if self.metrics.component_calls[enemy_prior_at] == 0
            && self.metrics.component_ms[enemy_prior_at] == 0.0
        {
            self.observe("enemy_prior_batch", 0.0, false);
        }
        let enemy_visible = enemy_is_visible(obs, &memory);
        let unshaped = evaluator.last_unshaped_prior().map(|p| p.to_vec());
        prior_probe_fields(
            &mut self.metrics,
            root_prior_for_probe.as_deref(),
            root_mask_for_probe.as_deref(),
            action,
            policy_fallback,
            has_root_result,
            unshaped.as_deref(),
            enemy_visible,
        );
        self.publish_metrics(
            level,
            recovery_flag,
            (belief_update_ok && has_root_result) as i64,
        );
        action
    }

    /// Propose, filter, and recover — the block charged to two components.
    fn run_belief_update<E: SearchEvaluator + ?Sized>(
        &mut self,
        evaluator: &mut E,
        obs: &Observation,
    ) {
        let belief = self.belief.clone().expect("checked by the caller");
        let max_batch = self.config.max_proposal_batch;
        let proposal_policy = if self.use_policy_proposal {
            evaluator.as_proposal_policy()
        } else {
            None
        };

        // The proposal's forwards are counted where they happen; the Python
        // wraps the policy in a counting shim for exactly this.
        let mut counting = proposal_policy.map(|inner| CountingPolicy { inner, forwards: 0 });
        let t0 = self.now();
        self.charge("belief_proposal");
        let mut telemetry = ProposalTelemetry::default();
        let enemy_actions = {
            let mut rng = self.rng.handle();
            let policy: Option<&mut (dyn ProposalPolicy + '_)> = match counting.as_mut() {
                Some(shim) => Some(shim),
                None => None,
            };
            propose_enemy_actions(&belief, &mut rng, policy, max_batch, Some(&mut telemetry))
        };
        if let Some(shim) = counting.as_ref() {
            self.forward_by_consumer[consumer_index("belief_proposal")] += shim.forwards;
            self.forward_equivalents += shim.forwards;
        }
        let mut propose_ms = (self.now() - t0) * 1000.0;
        if self.charge_fixed_forecasts {
            propose_ms = self.forecast_ms("belief_proposal");
        }
        self.observe("belief_proposal", propose_ms, true);
        self.proposal_telemetry = telemetry;

        let t1 = self.now();
        self.charge("particle_transitions");
        let last_action = self.last_action;
        let mut next = {
            let mut rng = self.rng.handle();
            filter_step(&belief, last_action, obs, &enemy_actions, &mut rng)
                .unwrap_or_else(|_| belief.clone())
        };
        if !next.particles.iter().any(|p| p.weight > 0.0) {
            let memory = self.memory.clone().expect("memory exists on a normal turn");
            let mut rng = self.rng.handle();
            let policy: Option<&mut (dyn ProposalPolicy + '_)> = match counting.as_mut() {
                Some(shim) => Some(shim.inner()),
                None => None,
            };
            next = recover_belief(&belief, obs, &memory, &mut rng, policy);
        }
        let mut filter_ms = (self.now() - t1) * 1000.0;
        if self.charge_fixed_forecasts {
            filter_ms = self.forecast_ms("particle_transitions");
        }
        self.observe("particle_transitions", filter_ms, true);
        self.belief = Some(next);
    }

    /// The batched search loop, with admission checked before every stage.
    fn run_search<E: SearchEvaluator + ?Sized>(
        &mut self,
        evaluator: &mut E,
        belief: &BeliefState,
        deadline: f64,
    ) {
        let target = self.config.target_simulations as u64;
        while self.search.tree.completed_simulations < target {
            let remaining = (target - self.search.tree.completed_simulations) as usize;
            let per_sim = (self.forecast_ms("selection")
                + self.forecast_ms("leaf_batch") / self.config.pending_leaf_batch.max(1) as f64
                + self.forecast_ms("backup"))
            .max(1e-6);
            let remaining_ms =
                (deadline - self.now()) * 1000.0 - self.config.admission_guard_ms;
            // `int(remaining_ms // per_sim)` in the Python: floor division on
            // floats, truncated. Negative remaining time therefore forecasts a
            // negative number of further simulations, which is the point — it
            // is what freezes widening on an already-overrun turn.
            let forecast_total = self.search.tree.completed_simulations as f64
                + (remaining_ms / per_sim).floor();
            let freeze_widening = forecast_total < self.config.widen_freeze_below as f64;

            if !self.can_admit("selection", deadline) {
                break;
            }
            if !self.can_admit("leaf_batch", deadline) {
                break;
            }
            if self.forward_equivalents >= self.config.max_forward_equivalents as i64 {
                break;
            }

            let batch_n = self.config.pending_leaf_batch.min(remaining);
            let mut paths: Vec<PendingPath> = Vec::new();
            let mut to_resume: Vec<EnemyPriorRequest> = Vec::new();

            'batch: while paths.len() < batch_n {
                let resume = if to_resume.is_empty() {
                    None
                } else {
                    Some(to_resume.remove(0))
                };
                let outcome =
                    match self.select_once(deadline, target, paths.len(), freeze_widening, resume) {
                        Some(outcome) => outcome,
                        None => break,
                    };
                match outcome {
                    Selection::NeedsPrior(request) => {
                        let mut missing = vec![*request];
                        // Collect more missing priors before one batched forward.
                        while paths.len() + to_resume.len() + missing.len() < batch_n {
                            match self.select_once(
                                deadline,
                                target,
                                paths.len(),
                                freeze_widening,
                                None,
                            ) {
                                None => break,
                                Some(Selection::NeedsPrior(next)) => missing.push(*next),
                                Some(Selection::Path(path)) => {
                                    paths.push(*path);
                                    break;
                                }
                            }
                        }
                        if !self.materialize(evaluator, &mut missing, belief, deadline) {
                            // Discard partial selections — no statistics moved.
                            for req in missing.iter().chain(to_resume.iter()) {
                                self.unpin(req);
                            }
                            to_resume.clear();
                            break 'batch;
                        }
                        to_resume.extend(missing);
                    }
                    Selection::Path(path) => paths.push(*path),
                }
            }

            // Finish any remaining resumed paths that fit the batch.
            while !to_resume.is_empty() && paths.len() < batch_n {
                let resume = to_resume.remove(0);
                let outcome = match self.select_once(
                    deadline,
                    target,
                    paths.len(),
                    freeze_widening,
                    Some(resume),
                ) {
                    Some(outcome) => outcome,
                    None => break,
                };
                match outcome {
                    Selection::NeedsPrior(request) => {
                        let mut one = vec![*request];
                        if !self.materialize(evaluator, &mut one, belief, deadline) {
                            for req in one.iter().chain(to_resume.iter()) {
                                self.unpin(req);
                            }
                            to_resume.clear();
                            break;
                        }
                        for req in one.into_iter().rev() {
                            to_resume.insert(0, req);
                        }
                    }
                    Selection::Path(path) => paths.push(*path),
                }
            }

            // Drop unfinished prior requests; keep pins only for kept paths.
            for req in &to_resume {
                self.unpin(req);
            }
            to_resume.clear();

            if paths.is_empty() {
                break;
            }

            // Inference is not cancelable: check once before the whole batch.
            if !self.can_admit("leaf_batch", deadline) {
                // Discard the selected paths — no statistics moved.
                break;
            }

            let t0 = self.now();
            self.charge("leaf_batch");
            let values = self.search.evaluate_leaves(evaluator, &mut paths, belief);
            let mut elapsed = (self.now() - t0) * 1000.0;
            if self.charge_fixed_forecasts {
                elapsed = self.forecast_ms("leaf_batch");
            }
            self.observe("leaf_batch", elapsed, true);
            self.forward_equivalents += paths.len() as i64;
            self.forward_by_consumer[consumer_index("leaf_batch")] += paths.len() as i64;

            for (path, value) in paths.iter().zip(&values) {
                if !self.can_admit("backup", deadline) {
                    // The remaining paths in this batch are discarded.
                    break;
                }
                let t0 = self.now();
                self.charge("backup");
                self.search.backup_path(path, *value);
                let mut elapsed = (self.now() - t0) * 1000.0;
                if self.charge_fixed_forecasts {
                    elapsed = self.forecast_ms("backup");
                }
                self.observe("backup", elapsed, true);
            }
        }
    }

    fn unpin(&mut self, request: &EnemyPriorRequest) {
        let node = self.search.tree.node_mut(request.node());
        node.pending_pins.retain(|pin| pin != &request.info_hash);
    }

    fn select_once(
        &mut self,
        deadline: f64,
        target: u64,
        pending: usize,
        freeze_widening: bool,
        resume: Option<EnemyPriorRequest>,
    ) -> Option<Selection> {
        if !self.can_admit("selection", deadline) {
            return None;
        }
        if self.search.tree.completed_simulations + pending as u64 >= target {
            return None;
        }
        let t0 = self.now();
        self.charge("selection");
        let outcome = self.search.select_path(freeze_widening, resume);
        let mut elapsed = (self.now() - t0) * 1000.0;
        if self.charge_fixed_forecasts {
            elapsed = self.forecast_ms("selection");
        }
        self.observe("selection", elapsed, true);
        Some(outcome)
    }

    fn materialize<E: SearchEvaluator + ?Sized>(
        &mut self,
        evaluator: &mut E,
        requests: &mut Vec<EnemyPriorRequest>,
        belief: &BeliefState,
        deadline: f64,
    ) -> bool {
        if requests.is_empty() {
            return true;
        }
        if !self.can_admit("enemy_prior_batch", deadline) {
            return false;
        }
        if self.forward_equivalents >= self.config.max_forward_equivalents as i64 {
            return false;
        }
        let t0 = self.now();
        self.charge("enemy_prior_batch");
        // The requests are consumed by the call and rebuilt for the resume,
        // because both halves need the same owned observation and memory.
        let cloned: Vec<EnemyPriorRequest> = requests
            .iter()
            .map(|req| EnemyPriorRequest {
                nodes: req.nodes.clone(),
                edges: req.edges.clone(),
                particle: req.particle.clone(),
                state: std::rc::Rc::clone(&req.state),
                depth: req.depth,
                freeze_widening: req.freeze_widening,
                info_hash: req.info_hash,
                enemy_obs: req.enemy_obs.clone(),
                enemy_mem: req.enemy_mem.clone(),
            })
            .collect();
        let prior_n = self
            .search
            .materialize_enemy_priors(evaluator, cloned, belief);
        let mut prior_ms = (self.now() - t0) * 1000.0;
        if self.charge_fixed_forecasts {
            prior_ms = if prior_n > 0 {
                self.forecast_ms("enemy_prior_batch")
            } else {
                0.0
            };
        }
        self.observe("enemy_prior_batch", prior_ms, true);
        if prior_n > 0 {
            self.forward_equivalents += prior_n as i64;
            self.forward_by_consumer[consumer_index("enemy_prior")] += prior_n as i64;
        }
        true
    }

    fn publish_metrics(
        &mut self,
        action_level: FallbackLevel,
        recovery_flag: i64,
        belief_plus_root_ok: i64,
    ) {
        let move_ms = ((self.now() - self.turn_start) * 1000.0).round() as i64;
        let mut ess_milli = 0i64;
        if let Some(belief) = &self.belief {
            if belief.n() > 0 {
                ess_milli = (ess(&belief.weights()) * 1000.0).round() as i64;
                if recovery_flag != 0 {
                    let cap = (0.25 * self.config.n_particles as f64 * 1000.0).round() as i64;
                    ess_milli = ess_milli.min(cap);
                }
            }
        }
        let idx = |name: &str| component_index(name).unwrap();
        self.metrics.move_ms = move_ms;
        self.metrics.completed_simulations = self.search.tree.completed_simulations as i64;
        self.metrics.forward_equivalents = self.forward_equivalents;
        self.metrics.forward_by_consumer = self.forward_by_consumer;
        self.metrics.belief_ess = ess_milli;
        self.metrics.recovery = recovery_flag;
        self.metrics.tree_size = self.search.tree.nodes.len() as i64;
        self.metrics.fallback_level = action_level.as_str();
        self.metrics.belief_plus_root_ok = belief_plus_root_ok;
        self.metrics.cost_belief_ms = (self.metrics.component_ms[idx("belief_proposal")]
            + self.metrics.component_ms[idx("belief_tensor")]
            + self.metrics.component_ms[idx("particle_transitions")])
        .round() as i64;
        self.metrics.cost_root_ms = self.metrics.component_ms[idx("root_inference")].round() as i64;
        self.metrics.cost_reply_ms = self.metrics.component_ms[idx("reply")].round() as i64;
        self.metrics.proposal_n_particles = self.proposal_telemetry.n_particles as i64;
        self.metrics.proposal_n_singleton_particles =
            self.proposal_telemetry.n_singleton_particles as i64;
        self.metrics.proposal_n_unique_info_keys =
            self.proposal_telemetry.n_unique_info_keys as i64;
        self.metrics.proposal_n_unique_policy_inputs =
            self.proposal_telemetry.n_unique_policy_inputs as i64;
        self.metrics.proposal_n_policy_batches = self.proposal_telemetry.n_policy_batches as i64;
    }
}

/// Counts the forwards a proposal policy performs, as `runtime.py`'s shim does.
struct CountingPolicy<'a, 'b> {
    inner: &'a mut (dyn ProposalPolicy + 'b),
    forwards: i64,
}

impl<'a, 'b> CountingPolicy<'a, 'b> {
    fn inner(&mut self) -> &mut (dyn ProposalPolicy + 'b) {
        self.inner
    }
}

impl ProposalPolicy for CountingPolicy<'_, '_> {
    fn policy_logits(&mut self, batch: &[&[f32]]) -> Vec<Vec<f64>> {
        self.forwards += batch.len() as i64;
        self.inner.policy_logits(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nearest_rank_p99_is_the_ceil_rank_not_an_interpolation() {
        let samples: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        assert_eq!(nearest_rank_p99(&samples).unwrap(), 99.0);
        assert_eq!(nearest_rank_p99(&[5.0]).unwrap(), 5.0);
        assert!(nearest_rank_p99(&[]).is_err());
    }

    #[test]
    fn a_cold_estimator_forecasts_its_offline_seed() {
        let est = NearestRankP99Estimator::new(4, 12.0).unwrap();
        assert_eq!(est.forecast(), 12.0);
    }

    #[test]
    fn a_warming_estimator_takes_the_max_of_the_seed_and_the_samples() {
        let mut est = NearestRankP99Estimator::new(4, 12.0).unwrap();
        est.observe(3.0);
        assert_eq!(est.forecast(), 12.0);
        est.observe(40.0);
        assert_eq!(est.forecast(), 40.0);
        assert!(!est.warmed_up());
    }

    #[test]
    fn a_warmed_estimator_uses_the_window_and_forgets_the_seed() {
        let mut est = NearestRankP99Estimator::new(4, 1000.0).unwrap();
        for value in [1.0, 2.0, 3.0, 4.0] {
            est.observe(value);
        }
        assert!(est.warmed_up());
        assert_eq!(est.forecast(), 4.0);
        // The window is bounded: the oldest sample falls out.
        est.observe(0.5);
        assert_eq!(est.forecast(), 4.0);
    }

    #[test]
    fn a_non_finite_seed_is_refused_rather_than_forecast() {
        assert!(NearestRankP99Estimator::new(4, f64::INFINITY).is_err());
        assert!(NearestRankP99Estimator::new(0, 1.0).is_err());
    }

    #[test]
    fn a_non_finite_offline_value_falls_back_to_the_part07_default() {
        let mut config = RuntimeConfig::default();
        config.offline_p99_ms = vec![("leaf_batch".to_string(), f64::NAN)];
        assert_eq!(offline_seed_ms(&config, "leaf_batch"), 30.0);
        // A component the config never mentions gets the same treatment.
        assert_eq!(offline_seed_ms(&config, "backup"), 2.0);
    }

    #[test]
    fn the_degradation_path_prefers_a_non_pass_fallback_to_emitting_pass() {
        let rng = SharedRng::new(Box::new(crate::support::rng::SmallRng::seed_from_u64(1)));
        let search = SearchController::new(0, SearchConfig::default(), rng);
        let (action, level) = select_degraded_action(0, false, Some([0, 1, 1, 0, 0]), &search);
        assert_eq!(action, PASS_ACTION);
        assert_eq!(level, FallbackLevel::Pass);
        let (action, level) = select_degraded_action(0, true, Some([0, 1, 1, 0, 0]), &search);
        assert_eq!(action, [0, 1, 1, 0, 0]);
        assert_eq!(level, FallbackLevel::Policy);
    }

    #[test]
    fn the_highest_prior_legal_action_ignores_masked_out_mass() {
        let mut mask = vec![false; PASS_INDEX + 1];
        mask[5] = true;
        let mut prior = vec![0.0; PASS_INDEX + 1];
        prior[3] = 0.9; // illegal, and the largest
        prior[5] = 0.1;
        assert_eq!(highest_prior_legal(&prior, &mask), decode_action(5).unwrap());
    }
}
