//! What is left of the network surfaces after the port: the ranking corner of
//! the proposal.
//!
//! The other two went with morpheus's own net and its Torch oracle (joe-net-plan
//! §8.1, §8.3). `net` ran a 49-plane tensor through all eleven heads and three
//! entry points; `prior` checked the legal-normalized prior and the WDL backup
//! value between the network and the search. Both were compared against
//! TorchScript, which no longer describes anything this bot runs, and both were
//! the only surfaces in the harness with a float tolerance rather than an
//! equality. **This file now holds no float tolerance at all.**
//!
//! Nothing replaces them here at N1. Joe's forward pass is proved transitively
//! instead — byte-identity against the copies joe-rs's JAX-oracle corpus
//! already covers, asserted by `tests/test_joe_source_fanout.py` (§8.2). N3
//! adds a **new** `prior` surface with its own Python/JAX oracle over the
//! remap and the bin -> scalar value conversion, which is where this port's own
//! bugs will live and is the most valuable single test in the plan (§8.4).

use crate::board::action::N_ACTIONS;
use crate::belief::proposal::{
    singleton_probs, softmax_masked, top_legal_actions,
};
use crate::parity::ints::Ints;
use crate::parity::Ctx;

/// logits + mask + k -> the masked softmax, the singleton fallback,
/// and the ranked candidate list
///
/// The corner of the proposal M4 ports but nothing else reaches.
/// `softmax_masked` runs only under the policy proposal, which
/// `deployment.json` ships turned **off**; and `top_legal_actions`
/// is otherwise exercised through recovery on a *uniform*
/// distribution, where every legal action ties and the check is
/// therefore only about NumPy's tie-breaking. Feeding real logits
/// here checks the ranking itself, and covers the branch that would
/// have to work before M7 could re-qualify `use_policy_proposal`.
pub(in crate::parity) fn toplegal(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let logits = ints.f64s(N_ACTIONS)?;
    let mask_ints = ints.ints(N_ACTIONS)?;
    let mask: Vec<bool> = mask_ints.iter().map(|v| *v != 0).collect();
    let k = ints.n()?;

    let probs = softmax_masked(&logits, &mask);
    let nonzero: Vec<usize> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p != 0.0)
        .map(|(i, _)| i)
        .collect();
    out.push(nonzero.len() as i64);
    for index in &nonzero {
        out.push(*index as i64);
        out.push(probs[*index].to_bits() as i64);
    }
    let singles = singleton_probs(&mask);
    out.push(singles.iter().position(|&p| p != 0.0).unwrap_or(0) as i64);

    let top = top_legal_actions(&probs, &mask, k);
    out.push(top.len() as i64);
    for action in &top {
        out.extend(action.iter().map(|&v| v as i64));
    }
    Ok(())
}
