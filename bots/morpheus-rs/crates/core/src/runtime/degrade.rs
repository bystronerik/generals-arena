//! What to send when the deadline arrives before the answer does.
//!
//! Three bands, worst first: pass, the highest-prior legal action, and the
//! search's own best-by-average. Emitting pass is the failure this file
//! exists to avoid — a legal move chosen badly beats no move at all — and
//! [`FallbackLevel`] records which band the committed action came from.

use crate::board::action::decode_action;
use crate::belief::{
    Action5, PASS_ACTION,
};
use crate::search::SearchController;

use super::*;

/// `int(np.argmax(np.where(mask, values, -1.0)))` — first maximum wins.
pub(super) fn argmax_masked(values: &[f64], mask: &[bool]) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::action::PASS_INDEX;
    use crate::search::SearchConfig;
    use crate::support::rng::SharedRng;

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
