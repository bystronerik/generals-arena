//! The two host-conditional NumPy behaviours, isolated.
//!
//! `npsum` is pairwise summation and `argsort` is the descending stable order
//! NumPy produces. Both are reproduced rather than approximated, because the
//! belief's resample threshold and the candidate ordering are decided on
//! their last bits.

use crate::support::rng::argsort_desc_numpy;
use crate::parity::codec::*;
use crate::parity::ints::Ints;
use crate::parity::Ctx;

/// a vector of f64 -> its NumPy pairwise sum
///
/// A one-line surface guarding a two-line function, because the
/// function is not portable by construction: `np.sum` reduces in
/// eight interleaved lanes, and a host whose NumPy vectorizes that
/// reduction differently would change the ESS, which would change
/// whether a resample happened, which would desynchronize the
/// replay stream three calls later. Checking the sum directly turns
/// that into one named failure instead.
pub(in crate::parity) fn npsum(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let count = ints.n()?;
    let values = ints.f64s(count)?;
    push_f64(out, &[crate::support::rng::npsum(&values)]);
    Ok(())
}

/// a vector of f64 -> `np.argsort(-scores)`
///
/// `top_legal_actions` ranks a distribution that, on the deployed
/// uniform proposal, is entirely ties — so the recovery candidate
/// order is decided by NumPy's introsort internals and by nothing
/// else. See `rng::argsort_desc_numpy` for why this surface is
/// host-conditional and why that is the oracle's property, not the
/// port's.
pub(in crate::parity) fn argsort(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let count = ints.n()?;
    let values = ints.f64s(count)?;
    // Length first: an empty permutation is a legitimate answer,
    // and a bare empty line would be dropped as blank by the
    // harness's line reader rather than compared.
    out.push(count as i64);
    out.extend(argsort_desc_numpy(&values).iter().map(|&i| i as i64));
    Ok(())
}
