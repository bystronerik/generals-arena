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

/// Widen one IEEE f16 bit pattern to the f32 bit pattern, matching the
/// hardware conversion numpy uses (subnormals normalized, NaN payload
/// shifted, quiet bit preserved). Exhaustively tested against numpy over
/// all 65536 patterns.
pub fn f16_to_f32_bits(h: u16) -> u32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = h as u32 & 0x3ff;
    match exp {
        0 => {
            if man == 0 {
                sign // signed zero
            } else {
                // Subnormal f16 (value = man * 2^-24): normalize. With the
                // msb of `man` at bit position p, shift = 10 - p, the value
                // is 1.xxx * 2^(p - 24), so the f32 exponent field is
                // 127 + p - 24 = 113 - shift and the mantissa is the bits
                // below the msb, left-aligned into 23.
                let shift = man.leading_zeros() - 21;
                let exp = 113 - shift;
                sign | (exp << 23) | ((man << (13 + shift)) & 0x7f_ffff)
            }
        }
        0x1f if man == 0 => sign | 0x7f80_0000, // inf
        // NaN: payload kept, quiet bit set — the hardware conversion numpy
        // uses quiets signaling NaNs. Unreachable from a real artifact (the
        // pack tool's f16 round-trip check refuses non-finite weights), but
        // the exhaustive numpy pin covers every pattern.
        0x1f => sign | 0x7fc0_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    }
}

// ---- The joe-net-v2 packed container -------------------------------------
//
// model.packed layout (docs/bots/joe-rs/rans-plan.md section 2, container
// framing added in P3; every integer little-endian):
//
//     magic: b"JNP2"
//     the original safetensors header block, verbatim:
//         header_len: u64, header_json: [u8; header_len]
//     then, per tensor in ascending data_offsets order, two parts
//     (low byte plane, then high byte plane of the f16 values):
//         codec: u8      (0 = raw plane bytes, 1 = rans0)
//         codec 0: len: u32, bytes: [u8; len]
//         codec 1: the rans0 part (parse_part's wire layout)
//
// The unpacker rebuilds model.safetensors byte-for-byte — header verbatim,
// data section from the decoded planes widened f16 -> f32 — and refuses to
// keep the result unless its sha256 equals the manifest's
// `safetensors_sha256`. Schema knowledge stays in net.rs: this code reads
// shapes only to size the tensors it reconstructs.

use crate::io::json;
use crate::sha256;
use std::path::Path;

const MAGIC: &[u8; 4] = b"JNP2";

fn decode_plane<'a>(packed: &'a [u8], at: &mut usize, n: usize) -> Result<Vec<u8>, String> {
    let codec = *packed
        .get(*at)
        .ok_or_else(|| format!("packed truncated at part header, offset {at}"))?;
    *at += 1;
    match codec {
        0 => {
            let len_bytes: [u8; 4] = packed
                .get(*at..*at + 4)
                .ok_or("packed truncated in raw part length")?
                .try_into()
                .unwrap();
            let len = u32::from_le_bytes(len_bytes) as usize;
            *at += 4;
            if len != n {
                return Err(format!("raw part is {len} bytes, tensor plane needs {n}"));
            }
            let bytes = packed
                .get(*at..*at + len)
                .ok_or("packed truncated in raw part payload")?;
            *at += len;
            Ok(bytes.to_vec())
        }
        1 => {
            let (part, used) = parse_part(&packed[*at..])?;
            *at += used;
            decode(&part, n)
        }
        other => Err(format!("unknown part codec {other}")),
    }
}

/// Reconstruct `model.safetensors` from `model.packed` in `dir`, verifying
/// the manifest's digests on both sides. Idempotent: an existing output
/// with the right digest is left alone.
pub fn unpack_artifact(dir: &Path) -> Result<(), String> {
    let manifest_text = std::fs::read_to_string(dir.join("manifest.json"))
        .map_err(|e| format!("read manifest.json: {e}"))?;
    let manifest = json::parse(&manifest_text).map_err(|e| format!("manifest.json: {e}"))?;
    let format = manifest.str_field("pack_format")?;
    if format != "joe-net-v2" {
        return Err(format!("pack_format {format:?}, this binary unpacks \"joe-net-v2\""));
    }
    let packed_name = manifest.str_field("packed")?;
    let packed_sha = manifest.str_field("packed_sha256")?;
    let out_name = manifest.str_field("safetensors")?;
    let out_sha = manifest.str_field("safetensors_sha256")?;

    let out_path = dir.join(out_name);
    if let Ok(existing) = std::fs::read(&out_path) {
        if sha256::hex_digest(&existing) == out_sha {
            eprintln!("[unclejoe] unpack-artifact: {out_name} already matches, nothing to do");
            return Ok(());
        }
    }

    let packed = std::fs::read(dir.join(packed_name))
        .map_err(|e| format!("read {packed_name}: {e}"))?;
    let got = sha256::hex_digest(&packed);
    if got != packed_sha {
        return Err(format!("{packed_name} sha256 {got} != manifest packed_sha256 {packed_sha}"));
    }
    if packed.len() < 12 || &packed[..4] != MAGIC {
        return Err("packed file does not start with the JNP2 magic".into());
    }
    let header_len =
        u64::from_le_bytes(packed[4..12].try_into().unwrap()) as usize;
    let header_end = 12 + header_len;
    let header_block = packed
        .get(4..header_end)
        .ok_or("packed truncated inside the safetensors header")?;
    let header_json = std::str::from_utf8(&packed[12..header_end])
        .map_err(|e| format!("header JSON is not UTF-8: {e}"))?;
    let header = json::parse(header_json).map_err(|e| format!("header JSON: {e}"))?;
    let json::Json::Object(entries) = &header else {
        return Err("header JSON is not an object".into());
    };

    // Tensor byte ranges, ascending and contiguous from zero.
    let mut ranges: Vec<(usize, usize, String)> = Vec::new();
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        if entry.str_field("dtype").map_err(|e| format!("{name}: {e}"))? != "F32" {
            return Err(format!("{name}: dtype is not F32"));
        }
        let offs = entry
            .field("data_offsets")
            .and_then(|v| v.as_array().ok_or_else(|| format!("{name}: data_offsets")))?;
        let begin = offs[0].as_i64().ok_or("data_offsets[0]")? as usize;
        let end = offs[1].as_i64().ok_or("data_offsets[1]")? as usize;
        ranges.push((begin, end, name.clone()));
    }
    ranges.sort();
    let mut expect = 0usize;
    for (begin, end, name) in &ranges {
        if *begin != expect || end < begin || (end - begin) % 4 != 0 {
            return Err(format!("{name}: data_offsets [{begin}, {end}) not contiguous at {expect}"));
        }
        expect = *end;
    }

    let mut data = vec![0u8; expect];
    let mut at = header_end;
    for (begin, end, name) in &ranges {
        let n = (end - begin) / 4;
        let lo = decode_plane(&packed, &mut at, n).map_err(|e| format!("{name}/lo: {e}"))?;
        let hi = decode_plane(&packed, &mut at, n).map_err(|e| format!("{name}/hi: {e}"))?;
        let out = &mut data[*begin..*end];
        for i in 0..n {
            let bits = f16_to_f32_bits(u16::from_le_bytes([lo[i], hi[i]]));
            out[4 * i..4 * i + 4].copy_from_slice(&bits.to_le_bytes());
        }
    }
    if at != packed.len() {
        return Err(format!("{} trailing bytes after the last part", packed.len() - at));
    }

    let mut file = Vec::with_capacity(header_block.len() + data.len());
    file.extend_from_slice(header_block);
    file.extend_from_slice(&data);
    let got = sha256::hex_digest(&file);
    if got != out_sha {
        return Err(format!(
            "reconstructed {out_name} sha256 {got} != manifest safetensors_sha256 {out_sha}"
        ));
    }
    let tmp = dir.join(format!("{out_name}.tmp"));
    std::fs::write(&tmp, &file).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &out_path).map_err(|e| format!("rename to {out_name}: {e}"))?;
    eprintln!(
        "[unclejoe] unpack-artifact: wrote {out_name} ({} bytes, sha256 {got})",
        file.len()
    );
    Ok(())
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
    fn f16_widening_edge_cases_match_numpy() {
        for (h, f) in [
            (0x0000u16, 0x0000_0000u32), // +0
            (0x8000, 0x8000_0000),       // -0
            (0x3C00, 0x3F80_0000),       // 1.0
            (0x0001, 0x3380_0000),       // smallest subnormal
            (0x03FF, 0x387F_C000),       // largest subnormal
            (0x7BFF, 0x477F_E000),       // largest normal
            (0x7C00, 0x7F80_0000),       // +inf
            (0xFC00, 0xFF80_0000),       // -inf
            (0x7E00, 0x7FC0_0000),       // quiet NaN
        ] {
            assert_eq!(f16_to_f32_bits(h), f, "pattern {h:#06x}");
        }
    }

    #[test]
    fn f16_widening_matches_numpy_exhaustively() {
        // FNV-1a over the widened f32 bit patterns of all 65536 f16
        // patterns, in order, little-endian bytes. The constant comes from
        // numpy: `np.arange(65536, np.uint16).view(np.float16)
        // .astype(np.float32).view(np.uint32)` hashed the same way.
        let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
        for h in 0..=0xFFFFu32 {
            for b in f16_to_f32_bits(h as u16).to_le_bytes() {
                acc = (acc ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(acc, 0x5d79_f1b0_86f3_0345);
    }

    #[test]
    fn bad_scale_bits_refuses() {
        let mut blob = G1_BLOB.to_vec();
        blob[0] = 7;
        assert!(parse_part(&blob).err().unwrap().contains("scale_bits"));
    }
}
