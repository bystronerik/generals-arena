//! Passive per-turn counters. The probe reads these; they never change play.
//!
//! [`TurnMetrics`] is the telemetry schema, kept field-for-field identical to
//! the Python's so the existing analysis tooling reads both bots.
//! `publish_metrics` is the controller's method that fills it at the end of a
//! turn, and it lives here rather than in `controller.rs` because what it
//! writes is this file's subject.

use crate::board::action::{encode_action, PASS_INDEX};
use crate::belief::{
    ess, Action5, PASS_ACTION,
};

use super::*;

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
    pub(super) fn fresh() -> Self {
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

impl RuntimeController {
    pub(super) fn publish_metrics(
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
