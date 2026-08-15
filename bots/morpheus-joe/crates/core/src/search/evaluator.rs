//! The policy/value interface the search plugs into, and every implementation.
//!
//! Port of `bots/morpheus/evaluator.py` plus the two stubs the search module
//! carried. Three implementations in one file because they only make sense
//! against each other: [`ShapedUniformEvaluator`] is the network-free arm of
//! the Part 17 C1 ablation — same play mask, same blend, same hard rules, only
//! the learned prior and value gone — and [`UniformEvaluator`] and
//! [`ScriptedEvaluator`] are the deterministic stubs the `search` parity
//! surface needs, because a network's answers agree to 6.6e-7 and that is
//! enough to reorder a near-tie and turn a search comparison into an argument
//! about the last bit of a softmax.
//!
//! **`NetworkEvaluator` is not here yet, and that is N1's whole point.** The
//! one that used to be — morpheus's 249,316-parameter CNN behind a 49-plane
//! tensor build — went with the net it wrapped. Its replacement runs joe's
//! frozen transformer and needs three things this phase does not build: the
//! observation bridge from morpheus's `Observation` into joe's `AugState`
//! (N2), the 4,410 -> 3,970 channel remap and the 128-bin value decode (N3,
//! landing in [`crate::nn::head`]). Until then `EvaluatorKind::Network` still
//! **loads and warms joe's net** — so the artifact, the schema check and the
//! startup budget are all exercised — and then plays through
//! [`ShapedUniformEvaluator`], which makes the seat a running, finishing,
//! deliberately weak bot rather than a bot that cannot start.
//!
//! Two roles the old evaluator carried are gone for good rather than pending:
//!
//! * **Belief proposal.** `as_proposal_policy` is deleted with the rest of the
//!   policy-proposal path; see `belief/proposal.rs` for why joe's net could not
//!   fill it even with budget to spare (joe-net-plan §3.3).
//! * **Enemy priors.** `policy_priors_many` survives as a trait method and its
//!   uniform implementation, but §3.4 rules that it stays uniform: joe's net
//!   would need a hypothesized enemy `AugState` that nothing in the belief
//!   reconstructs, and the turn has room for 3–5 forwards in total.
//!
//! **Batching is a loop here, not a tensor dimension.** M3 measured
//! TorchScript's batched call going superlinear on one x86 core (4x the work
//! for 11.8x the time) while a loop of single forwards stayed flat, and joe's
//! engine has no batch axis at all — its `Scratch` is a single `RefCell`. So
//! `evaluate_many` runs the batch sequentially and the caller's "leaf batch"
//! stays a scheduling unit, which is what the runtime charges it as. §5.2
//! scopes any change to that as upstream work in joe-rs, because §8.2 forbids
//! editing joe's files here.

use crate::belief::{Action5, BeliefState};
use crate::board::action::{legal_mask, N_ACTIONS};
use crate::board::memory::VisibleMemory;
use crate::io::wire::Observation;
use crate::search::EvalItem;
use crate::tactics::{
    apply_pre_contact_prior, default_shaping_log_clip, enemy_is_visible, play_mask,
    DEFAULT_SHAPING_FLOOR_FRAC, DEFAULT_SHAPING_LAMBDA,
};

/// Injectable policy/value interface for the search.
pub trait SearchEvaluator {
    fn evaluate(
        &mut self,
        obs: &Observation,
        memory: &VisibleMemory,
        belief: &BeliefState,
        from_root: bool,
        shape: bool,
    ) -> (Vec<f64>, f64);

    /// Batched evaluation. The default loops, as the Python's fallback does.
    fn evaluate_many(&mut self, items: &[EvalItem]) -> Vec<(Vec<f64>, f64)> {
        items
            .iter()
            .map(|item| {
                self.evaluate(
                    &item.obs,
                    &item.memory,
                    &item.belief,
                    item.from_root,
                    item.shape,
                )
            })
            .collect()
    }

    /// Policy-only priors for enemy tables. Enemy tables discard the value, so
    /// a network evaluator skips the WDL head here.
    fn policy_priors_many(&mut self, items: &[EvalItem]) -> Vec<Vec<f64>> {
        items
            .iter()
            .map(|item| {
                self.evaluate(&item.obs, &item.memory, &item.belief, false, false)
                    .0
            })
            .collect()
    }

    fn set_previous_action(&mut self, _action: Option<Action5>) {}

    /// The raw legal-normalized network prior from the last root evaluation,
    /// before the shaping blend — the "who's deciding" probe reads this.
    fn last_unshaped_prior(&self) -> Option<&[f64]> {
        None
    }

    fn clear_last_unshaped_prior(&mut self) {}
}

/// Deterministic stub: a uniform legal prior and a constant value.
pub struct UniformEvaluator {
    pub value: f64,
}

impl SearchEvaluator for UniformEvaluator {
    fn evaluate(
        &mut self,
        obs: &Observation,
        memory: &VisibleMemory,
        _belief: &BeliefState,
        _from_root: bool,
        _shape: bool,
    ) -> (Vec<f64>, f64) {
        let mask = legal_mask(obs, memory, None);
        let live = mask.iter().filter(|&&m| m).count();
        let mut prior = vec![0.0f64; N_ACTIONS];
        if live == 0 {
            prior[crate::board::action::PASS_INDEX] = 1.0;
        } else {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    prior[i] = 1.0 / live as f64;
                }
            }
        }
        (prior, self.value)
    }
}

/// Test helper: a fixed prior vector and value, masked and renormalized.
///
/// The Python's `ScriptedEvaluator`. It exists here for the same reason: the
/// `search` parity surface needs an evaluator whose answers are *identical* on
/// both sides, and the network's are not — they agree to 6.6e-7, which is
/// enough to reorder a near-tie in the candidate list and turn a search
/// comparison into an argument about the last bit of a softmax. With a scripted
/// prior, any disagreement in the tree is the tree's.
pub struct ScriptedEvaluator {
    pub prior: Vec<f64>,
    pub value: f64,
}

impl SearchEvaluator for ScriptedEvaluator {
    fn evaluate(
        &mut self,
        obs: &Observation,
        memory: &VisibleMemory,
        _belief: &BeliefState,
        _from_root: bool,
        _shape: bool,
    ) -> (Vec<f64>, f64) {
        let mask = legal_mask(obs, memory, None);
        let masked: Vec<f64> = (0..N_ACTIONS)
            .map(|i| if mask[i] { self.prior[i] } else { 0.0 })
            .collect();
        let total = crate::support::rng::npsum(&masked);
        if total <= 0.0 {
            let live = mask.iter().filter(|&&m| m).count().max(1);
            let prior = (0..N_ACTIONS)
                .map(|i| if mask[i] { 1.0 / live as f64 } else { 0.0 })
                .collect();
            return (prior, self.value);
        }
        (masked.iter().map(|&v| v / total).collect(), self.value)
    }
}

/// The four shaping knobs, exactly as `deployment.json` carries them.
#[derive(Clone, Copy)]
pub struct ShapingKnobs {
    pub lambda_pre_contact: f64,
    pub lambda_post_contact: f64,
    pub log_clip: f64,
    pub floor_frac: f64,
}

impl Default for ShapingKnobs {
    fn default() -> Self {
        Self {
            lambda_pre_contact: DEFAULT_SHAPING_LAMBDA,
            lambda_post_contact: DEFAULT_SHAPING_LAMBDA,
            log_clip: default_shaping_log_clip(),
            floor_frac: DEFAULT_SHAPING_FLOOR_FRAC,
        }
    }
}

impl ShapingKnobs {
    fn lambda_for(&self, enemy_visible: bool) -> f64 {
        if enemy_visible {
            self.lambda_post_contact
        } else {
            self.lambda_pre_contact
        }
    }
}

fn shape_root(
    knobs: &ShapingKnobs,
    prior: &[f64],
    obs: &Observation,
    memory: &VisibleMemory,
    mask: &[bool],
    belief: &BeliefState,
    previous_action: Option<Action5>,
) -> Vec<f64> {
    let lam = knobs.lambda_for(enemy_is_visible(obs, memory));
    apply_pre_contact_prior(
        prior,
        obs,
        memory,
        mask,
        Some(belief),
        lam,
        knobs.log_clip,
        knobs.floor_frac,
        0.0,
        previous_action,
    )
}

/// The network-free arm: a flat prior over the play mask and a zero value.
#[derive(Default)]
pub struct ShapedUniformEvaluator {
    pub knobs: ShapingKnobs,
    pub previous_action: Option<Action5>,
    pub last_unshaped: Option<Vec<f64>>,
}

impl ShapedUniformEvaluator {
    fn uniform(mask: &[bool]) -> Vec<f64> {
        let live = mask.iter().filter(|&&m| m).count();
        let mut prior = vec![0.0f64; N_ACTIONS];
        if live == 0 {
            prior[N_ACTIONS - 1] = 1.0;
            return prior;
        }
        for (i, &m) in mask.iter().enumerate() {
            if m {
                prior[i] = 1.0 / live as f64;
            }
        }
        prior
    }
}

impl SearchEvaluator for ShapedUniformEvaluator {
    fn evaluate(
        &mut self,
        obs: &Observation,
        memory: &VisibleMemory,
        belief: &BeliefState,
        from_root: bool,
        shape: bool,
    ) -> (Vec<f64>, f64) {
        if !from_root {
            let mask = legal_mask(obs, memory, None);
            return (Self::uniform(&mask), 0.0);
        }
        let mask = play_mask(obs, memory, None);
        let prior = Self::uniform(&mask);
        if !shape {
            return (prior, 0.0);
        }
        self.last_unshaped = Some(prior.clone());
        let shaped = shape_root(
            &self.knobs,
            &prior,
            obs,
            memory,
            &mask,
            belief,
            self.previous_action,
        );
        (shaped, 0.0)
    }

    fn policy_priors_many(&mut self, items: &[EvalItem]) -> Vec<Vec<f64>> {
        items
            .iter()
            .map(|item| Self::uniform(&legal_mask(&item.obs, &item.memory, None)))
            .collect()
    }

    fn set_previous_action(&mut self, action: Option<Action5>) {
        self.previous_action = action;
    }

    fn last_unshaped_prior(&self) -> Option<&[f64]> {
        self.last_unshaped.as_deref()
    }

    fn clear_last_unshaped_prior(&mut self) {
        self.last_unshaped = None;
    }
}
