//! The knobs, their defaults, and the component vocabulary.
//!
//! [`COST_COMPONENTS`] is the list every estimator, every metric field and
//! every recorded trace is keyed on, so its order is a wire format, not a
//! detail. The `DEFAULT_*` values are what `deployment.json` overrides, and
//! [`RuntimeConfig`] is the resolved result the controller reads.

use crate::tactics::{
    default_shaping_log_clip,
    DEFAULT_SHAPING_FLOOR_FRAC, DEFAULT_SHAPING_LAMBDA,
};

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

pub(super) fn consumer_index(name: &str) -> usize {
    FORWARD_CONSUMERS
        .iter()
        .position(|&key| key == name)
        .expect("unknown forward consumer")
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
pub(super) fn offline_seed_ms(config: &RuntimeConfig, name: &str) -> f64 {
    let fallback = default_offline_p99(name);
    match config.offline(name) {
        Some(value) if value.is_finite() && value >= 0.0 => value,
        Some(_) => fallback,
        None => fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_non_finite_offline_value_falls_back_to_the_part07_default() {
        let mut config = RuntimeConfig::default();
        config.offline_p99_ms = vec![("leaf_batch".to_string(), f64::NAN)];
        assert_eq!(offline_seed_ms(&config, "leaf_batch"), 30.0);
        // A component the config never mentions gets the same treatment.
        assert_eq!(offline_seed_ms(&config, "backup"), 2.0);
    }
}
