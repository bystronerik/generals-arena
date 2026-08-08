//! Information-set and enemy-table digests for the search.
//!
//! Port of `bots/morpheus/hashing.py`. Keys are full SHA-256 digests over
//! byte payloads whose layout is part of the contract: two implementations
//! that hash the same board through different byte widths produce different
//! node keys, and a search that disagrees about node identity is a different
//! search. So the widths below are load-bearing, and none of them is obvious.
//!
//! **`VisibleMemory` is not uniformly typed.** The Python arrays are `bool`
//! (one byte), `int32` (four, little-endian) and `int8` (one) depending on the
//! field, and `tobytes()` writes exactly that. `remembered_owner` and
//! `remembered_castle_owner` are `int8` while `last_seen_turn` and
//! `remembered_army` are `int32`; this crate stores all four as `i32`, so the
//! two `int8` fields are narrowed here on the way into the hash. Getting that
//! wrong yields a digest that is stable, plausible, and wrong.
//!
//! **`observation_payload` is not a hash**, despite `observation_hash` being
//! its name on the Python side. It is the raw little-endian `int32` payload,
//! hashed only when it is folded into a key. Kept as raw bytes because the
//! belief filter compares payloads directly for exact observation matching.

use crate::action::encode_action;
use crate::memory::VisibleMemory;
use crate::sha256::{sha256, Sha256};
use crate::wire::Observation;

pub const ZERO_DIGEST: [u8; 32] = [0u8; 32];

/// SHA-256 over every `VisibleMemory` plane in declaration order, then H, W.
pub fn memory_digest(memory: &VisibleMemory) -> [u8; 32] {
    let n = memory.cells();
    let mut hasher = Sha256::new();

    // Six `bool` planes: one byte per cell, 0 or 1.
    for plane in [
        &memory.known_mountain[..n],
        &memory.known_passable_base[..n],
        &memory.known_castle[..n],
        &memory.own_general[..n],
        &memory.known_enemy_general[..n],
        &memory.ever_visible[..n],
    ] {
        for &value in plane {
            hasher.update(&[value as u8]);
        }
    }
    // int32, little-endian.
    for &value in &memory.last_seen_turn[..n] {
        hasher.update(&value.to_le_bytes());
    }
    // int8 — narrowed, not widened. See the module note.
    for &value in &memory.remembered_owner[..n] {
        hasher.update(&[value as i8 as u8]);
    }
    for &value in &memory.remembered_army[..n] {
        hasher.update(&value.to_le_bytes());
    }
    for &value in &memory.remembered_was_castle[..n] {
        hasher.update(&[value as u8]);
    }
    for &value in &memory.remembered_castle_owner[..n] {
        hasher.update(&[value as i8 as u8]);
    }
    hasher.update(&(memory.h as i32).to_le_bytes());
    hasher.update(&(memory.w as i32).to_le_bytes());
    hasher.finish()
}

/// The observation's raw `int32` little-endian payload — scalars then grids.
///
/// Not a digest. The belief filter's exact-match test compares these bytes
/// directly, and hashing them here would throw away the information it needs.
pub fn observation_payload(obs: &Observation) -> Vec<u8> {
    let n = obs.h * obs.w;
    let mut out = Vec::with_capacity((7 + 3 * n) * 4);
    for scalar in [
        obs.h as i32,
        obs.w as i32,
        obs.turn,
        obs.my_land,
        obs.my_army,
        obs.opp_land,
        obs.opp_army,
    ] {
        out.extend_from_slice(&scalar.to_le_bytes());
    }
    for &value in &obs.type_grid[..n] {
        out.extend_from_slice(&(value as i32).to_le_bytes());
    }
    for &value in &obs.owner_grid[..n] {
        out.extend_from_slice(&(value as i32).to_le_bytes());
    }
    for &value in &obs.army_grid[..n] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Advance the rolling action-observation history digest.
pub fn roll_history_digest(prev: &[u8; 32], action: [i32; 5], obs_payload: &[u8]) -> [u8; 32] {
    let index = encode_action(action) as i32;
    sha256(&[prev, &index.to_le_bytes(), obs_payload])
}

/// Node key from already-computed digests, so memory is hashed once per turn.
pub fn info_state_key_prehashed(
    turn: i32,
    mem_digest: &[u8; 32],
    obs_payload: &[u8],
    history_digest: &[u8; 32],
) -> [u8; 32] {
    sha256(&[&turn.to_le_bytes(), mem_digest, obs_payload, history_digest])
}

/// Node key: turn ‖ memory digest ‖ observation payload ‖ history digest.
pub fn info_state_key(
    turn: i32,
    memory: &VisibleMemory,
    obs_payload: &[u8],
    history_digest: &[u8; 32],
) -> [u8; 32] {
    info_state_key_prehashed(turn, &memory_digest(memory), obs_payload, history_digest)
}

/// Enemy table key from already-computed observation payload and memory digest.
pub fn enemy_info_hash_prehashed(obs_payload: &[u8], mem_digest: &[u8; 32]) -> [u8; 32] {
    sha256(&[obs_payload, mem_digest])
}

pub fn enemy_info_hash(enemy_obs: &Observation, enemy_memory: &VisibleMemory) -> [u8; 32] {
    enemy_info_hash_prehashed(&observation_payload(enemy_obs), &memory_digest(enemy_memory))
}

/// Child map key: the action index plus the resulting observation payload.
pub fn child_edge_key(action: [i32; 5], obs_payload: &[u8]) -> [u8; 32] {
    let index = encode_action(action) as i32;
    sha256(&[&index.to_le_bytes(), obs_payload])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::MAX_CELLS;

    fn memory() -> VisibleMemory {
        let mut m = VisibleMemory::empty(2, 3);
        m.known_mountain[0] = true;
        m.ever_visible[1] = true;
        m.last_seen_turn[1] = 7;
        m.remembered_owner[1] = 2;
        m.remembered_army[1] = 300;
        m.remembered_castle_owner[2] = 1;
        m
    }

    #[test]
    fn a_digest_depends_on_every_plane() {
        let base = memory_digest(&memory());
        let mut planes: Vec<[u8; 32]> = Vec::new();
        for mutate in [
            (|m: &mut VisibleMemory| m.known_mountain[3] = true) as fn(&mut VisibleMemory),
            |m| m.known_passable_base[3] = true,
            |m| m.known_castle[3] = true,
            |m| m.own_general[3] = true,
            |m| m.known_enemy_general[3] = true,
            |m| m.ever_visible[3] = true,
            |m| m.last_seen_turn[3] = 5,
            |m| m.remembered_owner[3] = 1,
            |m| m.remembered_army[3] = 9,
            |m| m.remembered_was_castle[3] = true,
            |m| m.remembered_castle_owner[3] = 2,
        ] {
            let mut m = memory();
            mutate(&mut m);
            let digest = memory_digest(&m);
            assert_ne!(digest, base, "a plane change left the digest untouched");
            planes.push(digest);
        }
        // And each plane is distinguishable from the others: a layout that
        // collided two fields would still pass the check above.
        for i in 0..planes.len() {
            for j in i + 1..planes.len() {
                assert_ne!(planes[i], planes[j], "planes {i} and {j} collide");
            }
        }
    }

    #[test]
    fn dimensions_are_part_of_the_key() {
        let small = VisibleMemory::empty(2, 3);
        let wide = VisibleMemory::empty(3, 2);
        assert_ne!(memory_digest(&small), memory_digest(&wide));
    }

    #[test]
    fn the_owner_planes_are_hashed_as_signed_bytes() {
        // int8, not int32: a widened field would make -1 and 255 collide with
        // different neighbours and silently change every node key.
        let mut m = VisibleMemory::empty(1, 2);
        m.remembered_owner[0] = -1;
        let a = memory_digest(&m);
        m.remembered_owner[0] = 255;
        assert_eq!(a, memory_digest(&m), "not narrowed to one byte");
    }

    #[test]
    fn an_observation_payload_is_raw_int32_little_endian() {
        let mut obs = Observation::with_dims(1, 2);
        obs.turn = 3;
        obs.army_grid[1] = 258; // 0x0102
        let payload = observation_payload(&obs);
        assert_eq!(payload.len(), (7 + 3 * 2) * 4, "seven scalars, three grids");
        assert_eq!(&payload[0..4], &1i32.to_le_bytes());
        assert_eq!(&payload[8..12], &3i32.to_le_bytes());
        assert_eq!(&payload[payload.len() - 4..], &258i32.to_le_bytes());
    }

    #[test]
    fn history_rolls_forward_and_order_matters() {
        let obs = Observation::with_dims(1, 1);
        let payload = observation_payload(&obs);
        let a = roll_history_digest(&ZERO_DIGEST, [1, 0, 0, 0, 0], &payload);
        let b = roll_history_digest(&a, [0, 0, 0, 3, 0], &payload);
        let swapped = roll_history_digest(
            &roll_history_digest(&ZERO_DIGEST, [0, 0, 0, 3, 0], &payload),
            [1, 0, 0, 0, 0],
            &payload,
        );
        assert_ne!(a, b);
        assert_ne!(b, swapped, "history is order-sensitive");
    }

    #[test]
    fn node_keys_separate_turn_memory_observation_and_history() {
        let m = memory();
        let obs = Observation::with_dims(1, 1);
        let payload = observation_payload(&obs);
        let base = info_state_key(1, &m, &payload, &ZERO_DIGEST);
        assert_ne!(base, info_state_key(2, &m, &payload, &ZERO_DIGEST));
        assert_ne!(base, info_state_key(1, &VisibleMemory::empty(2, 3), &payload, &ZERO_DIGEST));
        let mut other = Observation::with_dims(1, 1);
        other.turn = 9;
        assert_ne!(base, info_state_key(1, &m, &observation_payload(&other), &ZERO_DIGEST));
        assert_ne!(base, info_state_key(1, &m, &payload, &[7u8; 32]));
    }

    #[test]
    fn prehashing_memory_gives_the_same_key() {
        let m = memory();
        let obs = Observation::with_dims(1, 1);
        let payload = observation_payload(&obs);
        assert_eq!(
            info_state_key(3, &m, &payload, &ZERO_DIGEST),
            info_state_key_prehashed(3, &memory_digest(&m), &payload, &ZERO_DIGEST),
        );
    }

    #[test]
    fn a_full_board_memory_hashes_the_live_region_only() {
        // The planes are `MAX_CELLS` long whatever the board size; hashing the
        // slack would make an 18x18 board's key depend on uninitialised space.
        let mut a = VisibleMemory::empty(18, 18);
        let mut b = VisibleMemory::empty(18, 18);
        a.last_seen_turn[MAX_CELLS - 1] = 999;
        b.last_seen_turn[MAX_CELLS - 1] = -7;
        assert_eq!(memory_digest(&a), memory_digest(&b));
    }
}
