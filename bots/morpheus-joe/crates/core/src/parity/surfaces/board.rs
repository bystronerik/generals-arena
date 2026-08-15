//! The deterministic layer, where the tolerance is zero.
//!
//! `order` is its own surface rather than a detail of `transition` because
//! move order is mostly *unobservable* downstream: when the two moves do not
//! interact, either order gives the same board. A mutation that broke the
//! NumPy index wrap survived the end-to-end transition check over hundreds of
//! recorded positions and is caught immediately by this one.

use crate::board::action::{
    decode_action, encode_action, legal_mask, live_build_cost, N_ACTIONS, PAD,
};
use crate::board::hashing::{
    child_edge_key, enemy_info_hash_prehashed, info_state_key_prehashed, memory_digest,
    observation_payload, roll_history_digest,
};
use crate::board::memory::update_memory;
use crate::board::observe::emit_observation;
use crate::board::symmetry;
use crate::board::transition::{determine_move_order, Actions};
use crate::parity::codec::*;
use crate::parity::ints::Ints;
use crate::parity::Ctx;

/// state + both actions -> next state + info
pub(in crate::parity) fn transition(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let state = read_state(ints)?;
    let actions: Actions = [ints.action()?, ints.action()?];
    let (next, info) = crate::board::transition::transition(&state, &actions);
    write_state(out, &next);
    out.extend([
        info.army[0],
        info.army[1],
        info.land[0],
        info.land[1],
        info.is_done as i64,
        info.winner as i64,
        info.time as i64,
    ]);
    Ok(())
}

/// state + both actions -> which seat resolves first
///
/// Its own surface rather than a detail of `transition`, because
/// move order is mostly *unobservable* downstream: when the two
/// moves do not interact, either order gives the same board. A
/// mutation that broke the NumPy index wrap here survived the
/// end-to-end transition check over hundreds of recorded positions
/// and is caught immediately by this one.
pub(in crate::parity) fn order(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let state = read_state(ints)?;
    let actions: Actions = [ints.action()?, ints.action()?];
    out.push(determine_move_order(&state, &actions) as i64);
    Ok(())
}

/// state + seat -> the fogged observation that seat receives
pub(in crate::parity) fn observe(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let state = read_state(ints)?;
    let seat = ints.n()?;
    write_observation(out, &emit_observation(&state, seat));
    Ok(())
}

/// observation + memory -> the 3970-long legal mask
pub(in crate::parity) fn mask(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let memory = read_memory(ints)?;
    let mask = legal_mask(&obs, &memory, None);
    out.reserve(N_ACTIONS);
    out.extend(mask.iter().map(|&v| v as i64));
    Ok(())
}

/// observation + memory -> the live build-cost grid
pub(in crate::parity) fn cost(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let memory = read_memory(ints)?;
    let cost = live_build_cost(&obs, &memory);
    out.extend(cost[..obs.h * obs.w].iter().map(|&v| v as i64));
    Ok(())
}

/// observation + memory -> memory after folding the observation in
pub(in crate::parity) fn memory(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let memory = read_memory(ints)?;
    write_memory(out, &update_memory(&memory, &obs));
    Ok(())
}

/// observation + memory + prev-history digest + action -> every
/// digest the search keys on
pub(in crate::parity) fn hash(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let obs = read_observation(ints)?;
    let memory = read_memory(ints)?;
    let mut prev = [0u8; 32];
    for slot in prev.iter_mut() {
        *slot = ints.next()? as u8;
    }
    let action = ints.action()?;
    let payload = observation_payload(&obs);
    let mem_digest = memory_digest(&memory);
    for digest in [
        mem_digest,
        info_state_key_prehashed(obs.turn, &mem_digest, &payload, &prev),
        enemy_info_hash_prehashed(&payload, &mem_digest),
        child_edge_key(action, &payload),
        roll_history_digest(&prev, action, &payload),
    ] {
        out.extend(digest.iter().map(|&b| b as i64));
    }
    Ok(())
}

/// symmetry index -> the coordinate map, direction map, and the
/// permutation it induces on the 3970 policy logits
pub(in crate::parity) fn symmetry(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let which = ints.n()?;
    let sym = symmetry::ALL[which % symmetry::ALL.len()];
    for r in 0..PAD {
        for c in 0..PAD {
            let (nr, nc) = sym.transform_rc(r, c);
            out.push(nr as i64);
            out.push(nc as i64);
        }
    }
    for d in 0..4 {
        out.push(sym.transform_dir(d) as i64);
    }
    // Where each logit index lands. A permutation compares as
    // integers, so this stays tier 1 even though it serves the
    // float tensor path.
    for index in 0..N_ACTIONS {
        let action = decode_action(index).unwrap_or([1, 0, 0, 0, 0]);
        out.push(encode_action(symmetry::transform_action(action, sym)) as i64);
    }
    Ok(())
}
