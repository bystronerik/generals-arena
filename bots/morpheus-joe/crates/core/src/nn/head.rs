//! The arithmetic between the network and the search.
//!
//! What is left of `network.rs` after the port. The 249,316-parameter CNN, its
//! eleven heads, the 49-plane input contract, and the WDL backup value all
//! went with N1; the forward pass now lives in the `joenet` crate, where it is
//! joe's and is not edited (joe-net-plan §2).
//!
//! [`legal_normalized_policy`] survives **unchanged, deliberately**. It is
//! written against a 3,970-long `f32` logit vector and a 3,970-long boolean
//! mask, which is exactly what §7.1's remap of joe's 4,410 raw logits
//! produces, so N3 changes what feeds it and not the function itself
//! (joe-net-plan §7.2). Copying it forward now rather than deleting and
//! re-deriving it at N3 is the point: this is oracle-checked code, and the
//! `prior` parity surface that proved it is retired with the Torch oracle
//! (§8.1).
//!
//! N3 lands two new functions beside it — `remap_joe_logits` (4,410 -> 3,970,
//! §7.1) and the 128-bin value decode (§7.3) — and a new `prior` surface with
//! a NumPy/JAX oracle to check them, because that is where the port's own bugs
//! will live.

use crate::board::action::N_ACTIONS;

/// Softmax over legal actions only; illegal entries stay exactly zero.
///
/// `masked_fill(finfo(f32).min)` then softmax then `* mask`, as in
/// `network.py`. The arithmetic is f32 and the *result* is f64, mirroring a
/// caller that softmaxes an f32 tensor and casts the answer. **The width is
/// part of the contract**: computing this in f64 is more accurate, disagrees
/// in the eighth decimal, and that is enough to reorder a near-tie. The same
/// rule governs the 128-bin value decode N3 adds here (§7.3), because joe
/// computes it in f32 too.
pub fn legal_normalized_policy(logits: &[f32], mask: &[bool]) -> Vec<f64> {
    assert_eq!(logits.len(), N_ACTIONS);
    assert_eq!(mask.len(), N_ACTIONS);
    let mut max = f32::NEG_INFINITY;
    for i in 0..N_ACTIONS {
        let v = if mask[i] { logits[i] } else { f32::MIN };
        if v > max {
            max = v;
        }
    }
    let mut exps = vec![0f32; N_ACTIONS];
    let mut total = 0f32;
    for i in 0..N_ACTIONS {
        let v = if mask[i] { logits[i] } else { f32::MIN };
        let e = (v - max).exp();
        exps[i] = e;
        total += e;
    }
    // The Python multiplies the softmax by the mask rather than reasoning
    // about `exp(finfo.min - max)`; an illegal action is exactly zero, not
    // merely negligible, and the harness checks that distinction.
    (0..N_ACTIONS)
        .map(|i| if mask[i] { (exps[i] / total) as f64 } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::action::PASS_INDEX;

    #[test]
    fn legal_policy_zeroes_illegal_actions_and_sums_to_one() {
        let mut logits = vec![0f32; N_ACTIONS];
        for (i, v) in logits.iter_mut().enumerate() {
            *v = (i % 13) as f32 * 0.1;
        }
        let mut mask = vec![false; N_ACTIONS];
        mask[7] = true;
        mask[9] = true;
        mask[PASS_INDEX] = true;
        let prior = legal_normalized_policy(&logits, &mask);
        let total: f64 = prior.iter().sum();
        // 1e-6, not 1e-12: the softmax divides in f32 on purpose (the width is
        // copied from `torch.softmax`, see the function's docs), so the masses
        // sum to one to about a float's worth of precision and no further.
        assert!((total - 1.0).abs() < 1e-6, "{total}");
        for i in 0..N_ACTIONS {
            if !mask[i] {
                assert_eq!(prior[i], 0.0);
            } else {
                assert!(prior[i] > 0.0);
            }
        }
    }
}
