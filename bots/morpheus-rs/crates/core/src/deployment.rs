//! The named deployment configuration.
//!
//! Port of `bots/morpheus/deployment.py`. **The file format is unchanged**, so
//! the existing operator tooling reads both bots' `deployment.json` and a
//! knob sweep can be expressed once. Every estimator and safety field is
//! explicit — no silent defaults at acceptance time.
//!
//! rewrite-plan §1 is blunt about what this file currently is: the shipped
//! Python config records *"Qualification verdict is no on this host"*, and its
//! knobs were fitted to component costs that M0 then measured wrong by 1.4× to
//! 13×. The parser is faithful so the two bots can be compared at identical
//! settings at M6; producing the first *qualified* configuration is M7.

use std::path::{Path, PathBuf};

use crate::evaluator::ShapingKnobs;
use crate::io::json::{parse, Json};
use crate::runtime::{
    default_offline_p99, RuntimeConfig, COST_COMPONENTS, DEFAULT_OFFLINE_P99_MS,
};
use crate::tactics::{default_shaping_log_clip, DEFAULT_SHAPING_FLOOR_FRAC, DEFAULT_SHAPING_LAMBDA};

pub const EVALUATOR_KINDS: [&str; 2] = ["network", "uniform"];
pub const P99_ESTIMATOR_TYPE: &str = "nearest_rank_empirical";
pub const P99_WARMUP_RULE: &str = "max_of_offline_p99_and_local_samples_until_window_full";

/// Which evaluator plays: the exported net, or the ablation's flat prior.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EvaluatorKind {
    Network,
    Uniform,
}

/// The complete coupled online configuration for one checkpoint family.
#[derive(Clone, Debug)]
pub struct DeploymentConfig {
    pub inference_runtime: String,
    pub n_particles: usize,
    pub target_simulations: usize,
    pub min_simulations: usize,
    pub pending_leaf_batch: usize,
    pub max_proposal_batch: usize,
    pub search_depth: usize,
    pub max_forward_equivalents: usize,
    pub max_tree_nodes: usize,
    pub widen_freeze_below: usize,
    pub shaping_lambda_pre_contact: f64,
    pub shaping_lambda_post_contact: f64,
    pub shaping_log_clip: f64,
    pub shaping_floor_frac: f64,
    pub evaluator: EvaluatorKind,
    pub normal_deadline_ms: f64,
    pub reserve_ms: f64,
    pub first_move_limit_ms: f64,
    pub admission_guard_ms: f64,
    pub resident_memory_target_mb: f64,
    pub p99_estimator_type: String,
    pub p99_warmup_rule: String,
    pub p99_window: usize,
    pub offline_p99_ms: Vec<(String, f64)>,
    pub warmup_batch_shapes: Vec<usize>,
    /// Belief proposal source. `false` advances particles on uniform legal
    /// enemy actions; `true` samples the policy net from the enemy's view.
    /// Measured off: the learned proposal showed no benefit (−0.07 ± 0.13
    /// paired over 100 games/arm) and skipping its forward returns ~13 ms/turn.
    pub use_policy_proposal: bool,
    pub qualification_host: String,
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            inference_runtime: "torch.jit.script+float32".to_string(),
            n_particles: 32,
            target_simulations: 8,
            min_simulations: 8,
            pending_leaf_batch: 4,
            max_proposal_batch: 16,
            search_depth: 8,
            max_forward_equivalents: 113,
            max_tree_nodes: 4096,
            widen_freeze_below: 16,
            shaping_lambda_pre_contact: DEFAULT_SHAPING_LAMBDA,
            shaping_lambda_post_contact: DEFAULT_SHAPING_LAMBDA,
            shaping_log_clip: default_shaping_log_clip(),
            shaping_floor_frac: DEFAULT_SHAPING_FLOOR_FRAC,
            evaluator: EvaluatorKind::Network,
            normal_deadline_ms: 125.0,
            reserve_ms: 25.0,
            first_move_limit_ms: 8500.0,
            admission_guard_ms: 10.0,
            resident_memory_target_mb: 256.0,
            p99_estimator_type: P99_ESTIMATOR_TYPE.to_string(),
            p99_warmup_rule: P99_WARMUP_RULE.to_string(),
            p99_window: 64,
            offline_p99_ms: DEFAULT_OFFLINE_P99_MS
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
            warmup_batch_shapes: vec![1, 4, 16],
            use_policy_proposal: false,
            qualification_host: String::new(),
        }
    }
}

fn number(data: &Json, key: &str, current: f64) -> f64 {
    data.get(key).and_then(Json::as_f64).unwrap_or(current)
}

fn count(data: &Json, key: &str, current: usize) -> usize {
    data.get(key)
        .and_then(Json::as_i64)
        .map(|v| v.max(0) as usize)
        .unwrap_or(current)
}

fn text(data: &Json, key: &str, current: &str) -> String {
    data.get(key)
        .and_then(Json::as_str)
        .unwrap_or(current)
        .to_string()
}

impl DeploymentConfig {
    /// Parse a mapping, applying every validation the Python applies.
    ///
    /// Unknown keys are ignored, as in the Python: the file carries provenance
    /// notes the bot has no use for, and refusing them would make a comment a
    /// startup failure.
    pub fn from_json(data: &Json) -> Result<Self, String> {
        let mut cfg = Self::default();
        cfg.inference_runtime = text(data, "inference_runtime", &cfg.inference_runtime);
        cfg.n_particles = count(data, "n_particles", cfg.n_particles);
        cfg.target_simulations = count(data, "target_simulations", cfg.target_simulations);
        cfg.min_simulations = count(data, "min_simulations", cfg.min_simulations);
        cfg.pending_leaf_batch = count(data, "pending_leaf_batch", cfg.pending_leaf_batch);
        cfg.max_proposal_batch = count(data, "max_proposal_batch", cfg.max_proposal_batch);
        cfg.search_depth = count(data, "search_depth", cfg.search_depth);
        cfg.max_forward_equivalents =
            count(data, "max_forward_equivalents", cfg.max_forward_equivalents);
        cfg.max_tree_nodes = count(data, "max_tree_nodes", cfg.max_tree_nodes);
        cfg.widen_freeze_below = count(data, "widen_freeze_below", cfg.widen_freeze_below);
        cfg.shaping_lambda_pre_contact = number(
            data,
            "shaping_lambda_pre_contact",
            cfg.shaping_lambda_pre_contact,
        );
        cfg.shaping_lambda_post_contact = number(
            data,
            "shaping_lambda_post_contact",
            cfg.shaping_lambda_post_contact,
        );
        cfg.shaping_log_clip = number(data, "shaping_log_clip", cfg.shaping_log_clip);
        cfg.shaping_floor_frac = number(data, "shaping_floor_frac", cfg.shaping_floor_frac);
        cfg.normal_deadline_ms = number(data, "normal_deadline_ms", cfg.normal_deadline_ms);
        cfg.reserve_ms = number(data, "reserve_ms", cfg.reserve_ms);
        cfg.first_move_limit_ms = number(data, "first_move_limit_ms", cfg.first_move_limit_ms);
        cfg.admission_guard_ms = number(data, "admission_guard_ms", cfg.admission_guard_ms);
        cfg.resident_memory_target_mb = number(
            data,
            "resident_memory_target_mb",
            cfg.resident_memory_target_mb,
        );
        cfg.p99_estimator_type = text(data, "p99_estimator_type", &cfg.p99_estimator_type);
        cfg.p99_warmup_rule = text(data, "p99_warmup_rule", &cfg.p99_warmup_rule);
        cfg.p99_window = count(data, "p99_window", cfg.p99_window);
        cfg.qualification_host = text(data, "qualification_host", &cfg.qualification_host);
        if let Some(Json::Bool(flag)) = data.get("use_policy_proposal") {
            cfg.use_policy_proposal = *flag;
        }
        if let Some(shapes) = data.get("warmup_batch_shapes").and_then(Json::as_array) {
            cfg.warmup_batch_shapes = shapes
                .iter()
                .filter_map(Json::as_i64)
                .map(|v| v.max(0) as usize)
                .collect();
        }
        if let Some(Json::Object(map)) = data.get("offline_p99_ms") {
            cfg.offline_p99_ms = map
                .iter()
                .filter_map(|(k, v)| v.as_f64().map(|value| (k.clone(), value)))
                .collect();
        }
        if let Some(kind) = data.get("evaluator").and_then(Json::as_str) {
            cfg.evaluator = match kind {
                "network" => EvaluatorKind::Network,
                "uniform" => EvaluatorKind::Uniform,
                other => {
                    return Err(format!(
                        "evaluator must be one of {EVALUATOR_KINDS:?}: {other:?}"
                    ))
                }
            };
        }

        let missing: Vec<&str> = COST_COMPONENTS
            .iter()
            .copied()
            .filter(|name| !cfg.offline_p99_ms.iter().any(|(key, _)| key == name))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "deployment offline_p99_ms missing components: {missing:?}"
            ));
        }
        let mut bad: Vec<&str> = cfg
            .offline_p99_ms
            .iter()
            .filter(|(_, value)| !value.is_finite() || *value < 0.0)
            .map(|(key, _)| key.as_str())
            .collect();
        bad.sort_unstable();
        if !bad.is_empty() {
            return Err(format!(
                "deployment offline_p99_ms must be finite and >= 0: {bad:?}"
            ));
        }
        if cfg.p99_estimator_type != P99_ESTIMATOR_TYPE {
            return Err(format!(
                "unsupported p99_estimator_type {:?}",
                cfg.p99_estimator_type
            ));
        }
        if cfg.p99_warmup_rule.is_empty() {
            return Err("p99_warmup_rule must be explicit".to_string());
        }
        if cfg.p99_window < 1 {
            return Err("p99_window must be >= 1".to_string());
        }
        if cfg.admission_guard_ms < 0.0 {
            return Err("admission_guard_ms must be >= 0".to_string());
        }
        for (name, value) in [
            ("shaping_lambda_pre_contact", cfg.shaping_lambda_pre_contact),
            (
                "shaping_lambda_post_contact",
                cfg.shaping_lambda_post_contact,
            ),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be finite in [0, 1]: {value}"));
            }
        }
        // An infinite clip is the retired unbounded regime, not a valid config.
        if !cfg.shaping_log_clip.is_finite() || cfg.shaping_log_clip <= 0.0 {
            return Err(format!(
                "shaping_log_clip must be finite and > 0: {}",
                cfg.shaping_log_clip
            ));
        }
        if !(0.0..1.0).contains(&cfg.shaping_floor_frac) {
            return Err(format!(
                "shaping_floor_frac must be in [0, 1): {}",
                cfg.shaping_floor_frac
            ));
        }
        Ok(cfg)
    }

    pub fn to_runtime_config(&self) -> RuntimeConfig {
        // Every named component gets an entry, then the file's own values
        // overwrite — so a component the file omits is a hard error above
        // rather than a silent 1.0 here.
        let mut offline: Vec<(String, f64)> = COST_COMPONENTS
            .iter()
            .map(|name| {
                let value = self
                    .offline_p99_ms
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|(_, v)| *v)
                    .unwrap_or_else(|| default_offline_p99(name));
                (name.to_string(), value)
            })
            .collect();
        for (name, value) in &self.offline_p99_ms {
            match offline.iter_mut().find(|(key, _)| key == name) {
                Some(slot) => slot.1 = *value,
                None => offline.push((name.clone(), *value)),
            }
        }
        RuntimeConfig {
            normal_deadline_ms: self.normal_deadline_ms,
            reserve_ms: self.reserve_ms,
            first_move_limit_ms: self.first_move_limit_ms,
            target_simulations: self.target_simulations,
            min_simulations: self.min_simulations,
            pending_leaf_batch: self.pending_leaf_batch,
            max_forward_equivalents: self.max_forward_equivalents,
            admission_guard_ms: self.admission_guard_ms,
            max_tree_nodes: self.max_tree_nodes,
            resident_memory_target_mb: self.resident_memory_target_mb,
            p99_window: self.p99_window,
            offline_p99_ms: offline,
            n_particles: self.n_particles,
            max_proposal_batch: self.max_proposal_batch,
            search_depth: self.search_depth,
            widen_freeze_below: self.widen_freeze_below,
            shaping_lambda_pre_contact: self.shaping_lambda_pre_contact,
            shaping_lambda_post_contact: self.shaping_lambda_post_contact,
            shaping_log_clip: self.shaping_log_clip,
            shaping_floor_frac: self.shaping_floor_frac,
        }
    }

    pub fn shaping_knobs(&self) -> ShapingKnobs {
        ShapingKnobs {
            lambda_pre_contact: self.shaping_lambda_pre_contact,
            lambda_post_contact: self.shaping_lambda_post_contact,
            log_clip: self.shaping_log_clip,
            floor_frac: self.shaping_floor_frac,
        }
    }
}

pub const DEPLOYMENT_NAME: &str = "deployment.json";

/// Where `deployment.json` sits, for each shape the binary is launched in.
///
/// Three, and all three are real. In the repo the binary is
/// `target/release/morpheus-rs` and the file is two directories up; in the
/// **static** submission both sit at the zip root; in the **vendored**
/// submission the binary is under `target/release/` again. Mirrors
/// `Session::default_artifact_dir`, and for the same reason: the static
/// variant is R4's fallback, and a fallback that silently plays placeholder
/// knobs is not a fallback.
pub fn deployment_candidates() -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(DEPLOYMENT_NAME));
            if let Some(up) = dir.parent().and_then(Path::parent) {
                candidates.push(up.join(DEPLOYMENT_NAME));
            }
        }
    }
    candidates.push(PathBuf::from(DEPLOYMENT_NAME));
    candidates
}

/// The first candidate that exists.
pub fn default_deployment_path() -> Option<PathBuf> {
    deployment_candidates().into_iter().find(|p| p.is_file())
}

pub fn load_deployment(path: &Path) -> Result<DeploymentConfig, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("deployment config missing: {}: {e}", path.display()))?;
    let data = parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if !matches!(data, Json::Object(_)) {
        return Err(format!("deployment root must be an object: {}", path.display()));
    }
    DeploymentConfig::from_json(&data)
}

/// Load `deployment.json` when present, else the Part 07 placeholders.
///
/// The fallback is loud on stderr. Placeholder knobs are a *different bot* —
/// four times the particles, twice the search depth, a 125 ms deadline instead
/// of 140 — so a submission that lost the file would play something nobody
/// measured, and would do it silently.
pub fn try_load_deployment() -> DeploymentConfig {
    match default_deployment_path() {
        Some(path) => match load_deployment(&path) {
            Ok(cfg) => cfg,
            Err(err) => {
                eprintln!("[morpheus-rs] {err}; falling back to placeholder knobs");
                DeploymentConfig::default()
            }
        },
        None => {
            eprintln!(
                "[morpheus-rs] no {DEPLOYMENT_NAME} among {:?}; playing placeholder knobs",
                deployment_candidates()
            );
            DeploymentConfig::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_offline() -> String {
        let entries: Vec<String> = COST_COMPONENTS
            .iter()
            .map(|name| format!("\"{name}\": 1.0"))
            .collect();
        format!("{{{}}}", entries.join(", "))
    }

    fn config_text(extra: &str) -> String {
        format!(
            "{{\"offline_p99_ms\": {}{}{}}}",
            full_offline(),
            if extra.is_empty() { "" } else { ", " },
            extra
        )
    }

    #[test]
    fn a_complete_config_parses_and_maps_to_a_runtime_config() {
        let text = config_text("\"n_particles\": 8, \"target_simulations\": 16");
        let cfg = DeploymentConfig::from_json(&parse(&text).unwrap()).unwrap();
        assert_eq!(cfg.n_particles, 8);
        let runtime = cfg.to_runtime_config();
        assert_eq!(runtime.target_simulations, 16);
        assert_eq!(runtime.offline_p99_ms.len(), COST_COMPONENTS.len());
    }

    #[test]
    fn a_missing_component_is_refused_by_name() {
        let text = "{\"offline_p99_ms\": {\"backup\": 1.0}}";
        let err = DeploymentConfig::from_json(&parse(text).unwrap()).unwrap_err();
        assert!(err.contains("missing components"), "{err}");
        assert!(err.contains("belief_tensor"), "{err}");
    }

    #[test]
    fn an_infinite_shaping_clip_is_the_retired_regime_and_is_refused() {
        let text = config_text("\"shaping_log_clip\": 0.0");
        let err = DeploymentConfig::from_json(&parse(&text).unwrap()).unwrap_err();
        assert!(err.contains("shaping_log_clip"), "{err}");
    }

    #[test]
    fn an_unknown_evaluator_kind_is_refused() {
        let text = config_text("\"evaluator\": \"oracle\"");
        let err = DeploymentConfig::from_json(&parse(&text).unwrap()).unwrap_err();
        assert!(err.contains("evaluator"), "{err}");
    }

    #[test]
    fn lambda_outside_the_unit_interval_is_refused() {
        let text = config_text("\"shaping_lambda_post_contact\": 1.5");
        let err = DeploymentConfig::from_json(&parse(&text).unwrap()).unwrap_err();
        assert!(err.contains("shaping_lambda_post_contact"), "{err}");
    }

    #[test]
    fn the_shipped_python_deployment_parses_unchanged() {
        // The format is shared by both bots; a field the Rust side cannot read
        // would silently play different knobs than the file names.
        let text = config_text(
            "\"admission_guard_ms\": 0.0, \"search_depth\": 2, \
             \"warmup_batch_shapes\": [1, 4, 8], \"use_policy_proposal\": false, \
             \"belief_quality_threshold\": null, \"qualification_host\": \"macOS\"",
        );
        let cfg = DeploymentConfig::from_json(&parse(&text).unwrap()).unwrap();
        assert_eq!(cfg.search_depth, 2);
        assert_eq!(cfg.warmup_batch_shapes, vec![1, 4, 8]);
        assert!(!cfg.use_policy_proposal);
        assert_eq!(cfg.qualification_host, "macOS");
    }
}
