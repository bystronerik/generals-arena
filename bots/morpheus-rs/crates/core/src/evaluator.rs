//! Network-backed search adapters for the frozen export.
//!
//! Port of `bots/morpheus/evaluator.py`. Two evaluators, both shaped:
//! [`NetworkEvaluator`] runs the exported net, and [`ShapedUniformEvaluator`]
//! is the network-free arm of the Part 17 C1 ablation — same play mask, same
//! blend, same hard rules, only the learned prior and value gone, so the
//! contrast measures exactly what the network contributes.
//!
//! Belief proposal and enemy priors use policy-only inference; the root and
//! leaves use policy plus WDL. The auxiliary heads never run on the online
//! search path.
//!
//! **Batching is a loop here, not a tensor dimension.** M3 measured
//! TorchScript's batched call going superlinear on one x86 core (4× the work
//! for 11.8× the time) while a loop of single forwards stayed flat, and the
//! bespoke engine has no batch axis at all. So `evaluate_many` runs the batch
//! sequentially and the caller's "leaf batch" stays a scheduling unit, which is
//! what the runtime charges it as.

use crate::action::legal_mask;
use crate::belief::{Action5, BeliefState};
use crate::inference::Session;
use crate::network::{backup_value, legal_normalized_policy, Heads, IN_CHANNELS, N_ACTIONS};
use crate::particle_summary::summarize_belief;
use crate::proposal::ProposalPolicy;
use crate::search::{EvalItem, SearchEvaluator};
use crate::tactics::{
    apply_pre_contact_prior, default_shaping_log_clip, enemy_is_visible, play_mask,
    DEFAULT_SHAPING_FLOOR_FRAC, DEFAULT_SHAPING_LAMBDA,
};
use crate::tensor::build_tensor;
use crate::io::wire::Observation;
use crate::memory::VisibleMemory;

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

/// `SearchEvaluator` over the exported network.
pub struct NetworkEvaluator {
    pub session: Session,
    pub knobs: ShapingKnobs,
    pub previous_action: Option<Action5>,
    /// The raw legal-normalized prior from the most recent root evaluation,
    /// kept for the "who's deciding" probe.
    pub last_unshaped: Option<Vec<f64>>,
}

impl NetworkEvaluator {
    pub fn new(session: Session, knobs: ShapingKnobs) -> Self {
        Self {
            session,
            knobs,
            previous_action: None,
            last_unshaped: None,
        }
    }

    fn tensor(&self, obs: &Observation, memory: &VisibleMemory, belief: &BeliefState) -> Vec<f32> {
        let summary = if belief.n() > 0 {
            Some(summarize_belief(belief))
        } else {
            None
        };
        build_tensor(
            obs,
            memory,
            summary.as_ref(),
            self.previous_action,
            self.session.army_scale,
        )
    }
}

impl SearchEvaluator for NetworkEvaluator {
    fn evaluate(
        &mut self,
        obs: &Observation,
        memory: &VisibleMemory,
        belief: &BeliefState,
        from_root: bool,
        shape: bool,
    ) -> (Vec<f64>, f64) {
        let x = self.tensor(obs, memory, belief);
        let (logits, wdl) = {
            let out = self.session.forward(&x, Heads::PolicyWdl);
            (out.flat_logits(), out.wdl_logits)
        };
        let mask = play_mask(obs, memory, None);
        let mut prior = legal_normalized_policy(&logits, &mask);
        if shape {
            self.last_unshaped = Some(prior.clone());
            prior = shape_root(
                &self.knobs,
                &prior,
                obs,
                memory,
                &mask,
                belief,
                self.previous_action,
            );
        }
        (prior, backup_value(wdl, from_root))
    }

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

    /// Enemy tables discard the value, so this skips the WDL head — and it
    /// masks with the plain `legal_mask`, not the play mask, because the enemy
    /// is not playing by Morpheus's house rules.
    fn policy_priors_many(&mut self, items: &[EvalItem]) -> Vec<Vec<f64>> {
        items
            .iter()
            .map(|item| {
                let x = self.tensor(&item.obs, &item.memory, &item.belief);
                let logits = self.session.forward(&x, Heads::Policy).flat_logits();
                let mask = legal_mask(&item.obs, &item.memory, None);
                legal_normalized_policy(&logits, &mask)
            })
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

    fn as_proposal_policy(&mut self) -> Option<&mut (dyn ProposalPolicy + '_)> {
        Some(self)
    }
}

/// The belief-proposal `PolicyFn`: `(B, 49, 21, 21) -> (B, 3970)` logits.
///
/// `deployment.json` ships `use_policy_proposal` off, so this is unreachable on
/// the deployed path; it exists because M7 may re-qualify the knob and because
/// a proposal surface with no implementation cannot be measured.
impl ProposalPolicy for NetworkEvaluator {
    fn policy_logits(&mut self, batch: &[&[f32]]) -> Vec<Vec<f64>> {
        batch
            .iter()
            .map(|x| {
                debug_assert_eq!(x.len(), IN_CHANNELS * 441);
                self.session
                    .forward(x, Heads::Policy)
                    .flat_logits()
                    .iter()
                    .map(|&v| v as f64)
                    .collect()
            })
            .collect()
    }
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
