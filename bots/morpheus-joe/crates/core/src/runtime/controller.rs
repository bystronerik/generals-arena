//! The deadline: what to attempt, in what order, and what to skip.
//!
//! [`RuntimeController`] owns first-move setup, the normal-turn work order,
//! and admission control — every component is forecast before it runs and
//! skipped if it does not fit under the reserve. It is ported faithfully
//! rather than improved, because the controller *is* the deadline behaviour.
//!
//! **M4 found the one number here that is a lie by construction.** The
//! controller charges `filter_step` and the `recover_belief` it may trigger to
//! a single component, `particle_transitions`, so its p99 describes a cost
//! distribution that is bimodal by a factor of a hundred (1.1 ms filtered,
//! 127 ms recovered on the Python). That accounting is reproduced, because
//! changing it would silently invalidate every recorded trace — but a retune
//! should reserve for the two paths separately.

use crate::belief::{
    filter_step, initialize_belief, Action5, BeliefConfig, BeliefState, PASS_ACTION,
};
use crate::board::memory::{update_memory, VisibleMemory};
use crate::belief::proposal::{propose_enemy_actions, ProposalTelemetry};
use crate::belief::recovery::recover_belief;
use crate::support::rng::SharedRng;
use crate::search::{
    EnemyPriorRequest, PendingPath, SearchConfig, SearchController, SearchEvaluator, Selection,
};
use crate::tactics::{
    constrain_nn_action, enemy_is_visible, play_mask, OSCILLATION_HISTORY,
};
use crate::io::wire::Observation;

use super::*;

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
    pub search: SearchController,
    pub memory: Option<VisibleMemory>,
    pub belief: Option<BeliefState>,
    pub metrics: TurnMetrics,
    estimators: Vec<NearestRankP99Estimator>,
    setup_done: bool,
    last_action: Action5,
    recent_actions: std::collections::VecDeque<Action5>,
    pending_recovery: bool,
    pub(super) turn_start: f64,
    pub(super) proposal_telemetry: ProposalTelemetry,
    pub(super) forward_equivalents: i64,
    pub(super) forward_by_consumer: [i64; 4],
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

    pub(super) fn now(&self) -> f64 {
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
                    self.run_belief_update(obs);
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
    ///
    /// Network-free end to end since N1 (joe-net-plan §3): the proposal draws
    /// uniformly over each particle's legal mask and recovery falls back to the
    /// same distribution, so nothing here spends a forward and the
    /// `belief_proposal` consumer's count is structurally zero.
    fn run_belief_update(&mut self, obs: &Observation) {
        let belief = self.belief.clone().expect("checked by the caller");
        let t0 = self.now();
        self.charge("belief_proposal");
        let mut telemetry = ProposalTelemetry::default();
        let enemy_actions = {
            let mut rng = self.rng.handle();
            propose_enemy_actions(&belief, &mut rng, Some(&mut telemetry))
        };
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
            next = recover_belief(&belief, obs, &memory, &mut rng);
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
}

