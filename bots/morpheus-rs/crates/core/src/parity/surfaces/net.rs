//! The forward pass and the two priors read off it.
//!
//! The only surfaces in the harness with a float tolerance rather than an
//! equality: `net` is compared at 1e-5 and measures 4.05e-6.

use crate::board::action::N_ACTIONS;
use crate::nn::network;
use crate::belief::proposal::{
    singleton_probs, softmax_masked, top_legal_actions,
};
use crate::parity::codec::*;
use crate::parity::ints::Ints;
use crate::parity::Ctx;

/// a 49×441 tensor -> every head, through all three entry points
///
/// Running `Policy`, `PolicyWdl`, and `All` on the same tensor and
/// emitting all three answers costs two extra forwards per case and
/// checks something no single entry point can: that the switch in
/// `forward_into` returns early without changing what it already
/// wrote. The Python side runs its three separate TorchScript
/// modules, so the comparison is entry point against entry point.
pub(in crate::parity) fn net(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    ctx: &mut Ctx,
) -> Result<(), String> {
    let session = ctx.net.get_or_insert_with(|| {
        crate::nn::inference::Session::load_default()
            .unwrap_or_else(|e| panic!("loading the artifact: {e}"))
    });
    let tensor = ints.floats(network::IN_CHANNELS * network::CELLS)?;

    let policy_only = session.forward(&tensor, network::Heads::Policy).clone();
    push_f32(out, &policy_only.policy);
    push_f32(out, &[policy_only.pass_logit]);

    let pw = session.forward(&tensor, network::Heads::PolicyWdl).clone();
    push_f32(out, &pw.policy);
    push_f32(out, &[pw.pass_logit]);
    push_f32(out, &pw.wdl_logits);

    let all = session.forward(&tensor, network::Heads::All);
    push_f32(out, &all.policy);
    push_f32(out, &[all.pass_logit]);
    push_f32(out, &all.wdl_logits);
    push_f32(out, &all.hidden_owner);
    push_f32(out, &all.enemy_army_bins);
    push_f32(out, &all.enemy_general);
    push_f32(out, &all.hidden_castle);
    push_f32(
        out,
        &[
            all.land_margin,
            all.army_margin,
            all.castle_margin,
            all.turns_to_termination,
        ],
    );
    Ok(())
}

/// logits + mask + WDL -> the legal-normalized prior and the backup
/// value, in f64
///
/// The arithmetic between the network and the search, split out
/// from the network itself so a divergence lands on one of them.
/// It is also the only surface that checks the flat policy layout
/// end to end: a channel-major/row-major swap leaves every head
/// bit-identical and moves every prior mass to the wrong action.
pub(in crate::parity) fn prior(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let logits = ints.floats(network::N_ACTIONS)?;
    let mask_ints = ints.ints(network::N_ACTIONS)?;
    let mask: Vec<bool> = mask_ints.iter().map(|v| *v != 0).collect();
    let wdl = ints.floats(3)?;
    let from_root = ints.next()? != 0;
    let prior = network::legal_normalized_policy(&logits, &mask);
    push_f64(out, &prior);
    push_f64(
        out,
        &[network::backup_value([wdl[0], wdl[1], wdl[2]], from_root)],
    );
    Ok(())
}

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
