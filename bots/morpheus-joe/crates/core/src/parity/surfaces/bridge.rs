//! The observation bridge, against a corpus this bot did not have to record.
//!
//! One surface, `sequence`, and it is N2's whole gate. It replays a recorded
//! game frame by frame through [`ObsBridge`] and reports two numbers per turn:
//!
//! | column | question |
//! | --- | --- |
//! | CRC-32 of the augmented tensor | is the bridge's `AugState` joe's? |
//! | disagreeing build-cost cells | is Q10 still closed? |
//!
//! then the final `AugState` in full.
//!
//! # Why a CRC and not the tensor
//!
//! joe-rs's `sequence` surface reports exactly this digest, and joe-rs's
//! recorded corpus stores it per turn under `all_aug_hash` — captured from the
//! **Python/JAX** pipeline, not from Rust. Emitting the same digest in the same
//! order makes that corpus this bridge's oracle for free, and that is the whole
//! trick: the fork never had to record 39×441 floats a turn for three games,
//! and the thing it is compared against was produced by joe itself. The
//! polynomial is zlib's, mirrored by `zlib.crc32` on the capture side.
//!
//! The digest is taken **before** normalization, which is why
//! [`ObsBridge::augment`] stops one step short of [`ObsBridge::advance`]. The
//! corpus hashes the unnormalized tensor and the play path normalizes it, so
//! one of the two has to be reachable on its own.
//!
//! A digest hides *where* a divergence is, on purpose: the answer is one
//! integer per turn instead of 17,199, and the first turn that differs is what
//! matters. Localizing inside that turn is joe-rs's `obs` surface's job, and
//! joe-rs still has it.
//!
//! # The second column, which joe-rs has no reason to carry
//!
//! §6.2 stops the port from consulting joe's mask, so from N3 on only
//! morpheus's `live_build_cost` prices a castle — and the two implementations
//! then drift silently, because nothing reads joe's any more. Q10 asks whether
//! they agree cell for cell. Both implement 35 base plus `max(0, 14 − 2·d)`
//! within radius 6, and both are integer kernels, so the honest test is not a
//! hand-built frame but every turn of a real game: this column is the count of
//! cells where they differ, and the gate is that it is zero everywhere.
//!
//! They can differ in principle. Joe prices the **visible** own structures of
//! the current frame; morpheus folds in latched knowledge — `own_general` and
//! `known_castle` from `VisibleMemory`, which is why the memory is advanced
//! here exactly as the runtime advances it. Owning a cell implies seeing it,
//! so the latch should never add a structure the frame does not already show.
//! "Should" is what this column replaces.

use crate::board::action::live_build_cost;
use crate::board::memory::{update_memory, VisibleMemory};
use crate::io::wire::Observation;
use crate::nn::bridge::ObsBridge;
use crate::parity::codec::push_f32;
use crate::parity::ints::Ints;
use crate::parity::Ctx;

/// CRC-32 (zlib's 0xEDB88320) over the little-endian bytes of the f32 values.
///
/// Mirrors joe-rs's `sequence` digest and `zlib.crc32` in the capture tool.
/// The table is rebuilt per call because a call costs one game and the table
/// costs 256 iterations.
fn crc32(values: &[f32]) -> i64 {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *entry = c;
    }
    let mut crc = !0u32;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
    }
    (!crc) as i64
}

/// One wire-shaped frame into a morpheus `Observation` whose dims are set.
///
/// The layout is joe-rs's, not [`crate::parity::codec::read_observation`]'s:
/// scalars first and no repeated dimensions, because the case header already
/// gave them once for the whole game.
fn read_frame(ints: &mut Ints, obs: &mut Observation) -> Result<(), String> {
    obs.turn = ints.next()? as i32;
    obs.my_land = ints.next()? as i32;
    obs.my_army = ints.next()? as i32;
    obs.opp_land = ints.next()? as i32;
    obs.opp_army = ints.next()? as i32;
    let n = obs.h * obs.w;
    for i in 0..n {
        obs.type_grid[i] = ints.next()? as u8;
    }
    for i in 0..n {
        obs.owner_grid[i] = ints.next()? as u8;
    }
    for i in 0..n {
        obs.army_grid[i] = ints.next()? as i32;
    }
    Ok(())
}

/// `h w turns` + one frame per turn -> per turn `(tensor digest, build-cost
/// disagreements)`, then the final `AugState`, then the temporal input
///
/// The state is serialized in `AugState`'s own field order, which is what the
/// corpus's `final_state_*` arrays are named after, so
/// `tests/test_morpheus_joe_bridge.py::STATE_LAYOUT` reads it positionally and
/// compares field by field. (That driver is this bot's own: joe-rs's
/// `parity_lib.py` reads the same order for the same reason, and the two
/// harnesses are deliberately not shared — a second module of the same name
/// would silently hand one bot the other's.)
pub(in crate::parity) fn sequence(
    ints: &mut Ints,
    out: &mut Vec<i64>,
    _ctx: &mut Ctx,
) -> Result<(), String> {
    let h = ints.n()?;
    let w = ints.n()?;
    let turns = ints.n()?;

    let mut obs = Observation::with_dims(h, w);
    let mut bridge = ObsBridge::new(h, w)?;
    let mut memory = VisibleMemory::empty(h, w);

    for _ in 0..turns {
        read_frame(ints, &mut obs)?;
        // Unnormalized: the corpus digests the tensor before the /50.
        bridge.augment(&obs);
        out.push(crc32(bridge.aug()));

        // The runtime's own order — the memory absorbs the frame, then prices
        // it — so a disagreement here is a disagreement the bot would have.
        memory = update_memory(&memory, &obs);
        let morpheus_cost = live_build_cost(&obs, &memory);
        let joe_cost = bridge.build_cost();
        let disagreements = (0..h * w).filter(|&i| joe_cost[i] != morpheus_cost[i]).count();
        out.push(disagreements as i64);
    }

    let state = bridge.state();
    push_f32(out, &state.army_stack);
    push_f32(out, &state.enemy_stack);
    push_f32(out, &state.last_army);
    push_f32(out, &state.last_enemy_army);
    for plane in [
        &state.castles,
        &state.generals,
        &state.mountains,
        &state.seen,
        &state.enemy_seen,
    ] {
        out.extend(plane.iter().map(|&v| v as i64));
    }
    push_f32(out, &state.last_enemy_army_seen_value);
    push_f32(out, &state.last_enemy_army_seen_timestep);
    push_f32(out, &state.opponent_army_history);
    push_f32(out, &state.opponent_land_history);
    out.push(state.temporal_step as i64);

    // The `(2, 512)` buffer the forward is actually handed, which is the one
    // thing in the bridge's output that the state above does not cover.
    //
    // It is a copy of those same two ring buffers, so it needs no oracle — the
    // check is that it *is* the copy. That distinction is the whole of §6.1
    // and it fails silently: halves swapped, or a stale buffer from the turn
    // before, leaves a state the corpus still agrees with and a network input
    // that is wrong.
    push_f32(out, bridge.temporal());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The check value every CRC-32 implementation agrees on: `"123456789"`
    /// hashes to 0xCBF43926. Fed as bytes through the f32 path, four at a
    /// time, so the byte order is checked too — the capture side hashes the
    /// same little-endian bytes.
    #[test]
    fn crc32_matches_the_standard_check_value() {
        let bytes = b"123456789\0\0\0";
        let values: Vec<f32> = bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Three whole words: "1234", "5678", "9\0\0\0" — the trailing NULs are
        // part of this input, so compare against zlib over the same twelve
        // bytes rather than over the nine-character string.
        assert_eq!(crc32(&values) as u32, 0x77D5_5834);
        assert_eq!(crc32(&[]) as u32, 0);
    }
}
