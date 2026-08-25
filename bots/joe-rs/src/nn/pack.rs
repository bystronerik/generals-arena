//! rANS decoder for the `joe-net-v2` packed artifact (rans0 codec).
//!
//! The rans_byte variant of the public-domain ryg_rans reference: u32
//! state, normalization interval lower bound `L = 1 << 23`, byte-at-a-time
//! renormalization. The Python encoder (`tools/rans.py`) is LIFO — it
//! walks symbols in reverse and writes backwards — so this decoder reads
//! the stream forward, seeded by the 4-byte little-endian state that heads
//! the payload. Frequency tables travel in the container; this module
//! validates and consumes them, it never builds or normalizes one, so
//! there is no cross-language normalization to keep in sync
//! (docs/bots/joe-rs/rans-plan.md).
//!
//! Loudness over leniency, same as the safetensors loader: a truncated
//! payload, a frequency sum that is not `1 << scale_bits`, unconsumed
//! trailing bytes, or a final state that is not `L` are hard errors.
//! Decoding runs once at intake next to a full cargo build, so clarity
//! beats cycles: `u64` intermediates wherever overflow is not obvious.

pub const RANS_L: u32 = 1 << 23;

/// One parsed `rans0` part: `scale_bits`, the 256-entry frequency table,
/// and the payload (initial state + stream). Wire layout, all
/// little-endian: `scale_bits: u8`, `freqs: [u16; 256]`,
/// `payload_len: u32`, `payload: [u8; payload_len]`.
pub struct RansPart<'a> {
    pub scale_bits: u8,
    pub freqs: [u16; 256],
    pub payload: &'a [u8],
}

/// Parse one part from the front of `blob`; returns the part and the
/// number of bytes consumed.
pub fn parse_part(blob: &[u8]) -> Result<(RansPart<'_>, usize), String> {
    const HEADER: usize = 1 + 256 * 2 + 4;
    if blob.len() < HEADER {
        return Err(format!("rans0 part header: {} bytes, need {HEADER}", blob.len()));
    }
    let scale_bits = blob[0];
    if !(8..=15).contains(&scale_bits) {
        return Err(format!("rans0 scale_bits {scale_bits} outside 8..=15"));
    }
    let mut freqs = [0u16; 256];
    for (i, f) in freqs.iter_mut().enumerate() {
        *f = u16::from_le_bytes([blob[1 + 2 * i], blob[2 + 2 * i]]);
    }
    let sum: u64 = freqs.iter().map(|&f| f as u64).sum();
    if sum != 1u64 << scale_bits {
        return Err(format!("rans0 frequency sum {sum} != {}", 1u64 << scale_bits));
    }
    let len_at = 1 + 256 * 2;
    let payload_len =
        u32::from_le_bytes([blob[len_at], blob[len_at + 1], blob[len_at + 2], blob[len_at + 3]])
            as usize;
    let end = HEADER + payload_len;
    if blob.len() < end {
        return Err(format!("rans0 payload: {} bytes past header, need {payload_len}",
                           blob.len() - HEADER));
    }
    Ok((RansPart { scale_bits, freqs, payload: &blob[HEADER..end] }, end))
}

/// Decode exactly `n` symbols from a parsed part.
pub fn decode(part: &RansPart, n: usize) -> Result<Vec<u8>, String> {
    let k = part.scale_bits as u32;
    let m = 1usize << k;
    let mask = (m - 1) as u32;

    // Cumulative table and the slot -> symbol lookup (transient, <= 32 KiB).
    let mut cum = [0u32; 257];
    for s in 0..256 {
        cum[s + 1] = cum[s] + part.freqs[s] as u32;
    }
    let mut slot2sym = vec![0u8; m];
    for s in 0..256 {
        for slot in cum[s]..cum[s + 1] {
            slot2sym[slot as usize] = s as u8;
        }
    }

    let payload = part.payload;
    if payload.len() < 4 {
        return Err(format!("rans0 payload {} bytes, need the 4-byte state", payload.len()));
    }
    let mut x = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let mut pos = 4usize;
    let mut out = vec![0u8; n];
    for (i, slot_out) in out.iter_mut().enumerate() {
        let slot = x & mask;
        let s = slot2sym[slot as usize];
        let f = part.freqs[s as usize] as u64;
        x = (f * (x >> k) as u64 + slot as u64 - cum[s as usize] as u64) as u32;
        while x < RANS_L {
            if pos >= payload.len() {
                return Err(format!("rans0 payload truncated at symbol {i}/{n}"));
            }
            x = (x << 8) | payload[pos] as u32;
            pos += 1;
        }
        *slot_out = s;
    }
    if pos != payload.len() {
        return Err(format!("rans0: {} unconsumed payload bytes", payload.len() - pos));
    }
    if x != RANS_L {
        return Err(format!("rans0 final state {x:#x} != L"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_blob(blob: &[u8], n: usize) -> Result<Vec<u8>, String> {
        let (part, used) = parse_part(blob)?;
        assert_eq!(used, blob.len(), "fixture blob has trailing bytes");
        decode(&part, n)
    }

    // Golden fixtures encoded by tools/rans.py (rans-plan.md P2); the
    // generator run is recorded in the plan. G1: skewed two-symbol
    // stream. G2: full 256-symbol alphabet. G3: single symbol — its
    // payload is the hand-verifiable case: encoding never leaves the
    // initial state, so the stream is exactly LE32(L).
    include!("../../tests/fixtures/rans_vectors.rs");

    #[test]
    fn golden_streams_decode_byte_exact() {
        for (raw, blob) in [(G1_RAW, G1_BLOB), (G2_RAW, G2_BLOB), (G3_RAW, G3_BLOB)] {
            assert_eq!(decode_blob(blob, raw.len()).unwrap(), raw);
        }
    }

    #[test]
    fn single_symbol_payload_is_the_initial_state() {
        let (part, _) = parse_part(G3_BLOB).unwrap();
        assert_eq!(part.payload, RANS_L.to_le_bytes());
    }

    #[test]
    fn truncated_payload_refuses() {
        let mut blob = G1_BLOB.to_vec();
        let n = blob.len();
        blob.truncate(n - 1);
        let len_at = 1 + 512;
        let new_len = (G1_BLOB.len() - 517 - 1) as u32;
        blob[len_at..len_at + 4].copy_from_slice(&new_len.to_le_bytes());
        let err = decode_blob(&blob, G1_RAW.len()).unwrap_err();
        assert!(err.contains("truncated"), "{err}");
    }

    #[test]
    fn bad_frequency_sum_refuses() {
        let mut blob = G1_BLOB.to_vec();
        blob[1] = blob[1].wrapping_add(1); // corrupt freqs[0] low byte
        let err = decode_blob(&blob, G1_RAW.len()).unwrap_err();
        assert!(err.contains("frequency sum"), "{err}");
    }

    #[test]
    fn corrupt_state_refuses() {
        let mut blob = G1_BLOB.to_vec();
        let state_at = 1 + 512 + 4;
        blob[state_at + 3] ^= 0x40;
        assert!(decode_blob(&blob, G1_RAW.len()).is_err());
    }

    #[test]
    fn short_header_refuses() {
        assert!(parse_part(&[14u8; 12]).is_err());
        assert!(parse_part(&[]).is_err());
    }

    #[test]
    fn bad_scale_bits_refuses() {
        let mut blob = G1_BLOB.to_vec();
        blob[0] = 7;
        assert!(parse_part(&blob).err().unwrap().contains("scale_bits"));
    }
}
