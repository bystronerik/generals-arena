"""rANS entropy coder for the joe-net-v2 packed artifact (rans0 codec).

The rans_byte variant of the public-domain ryg_rans reference
(github.com/rygorous/ryg_rans): u32 state, normalization interval lower
bound L = 1 << 23, byte-at-a-time renormalization. Encoding is LIFO — the
encoder walks the symbol stream in reverse and writes its buffer
backwards; the decoder reads forward. The flushed 4-byte little-endian
final state heads the payload.

This module is the single Python implementation: the pack tool encodes
with it and immediately self-decodes with it before writing anything
(rans-plan.md section 3). Frequency tables travel in the container, so
normalization exists only here — the Rust decoder validates and consumes
tables, it never builds them.

Stream format per part (rans-plan.md section 2):

    scale_bits: u8        (14 in v1)
    freqs: [u16; 256]     (little-endian, sum == 1 << scale_bits)
    payload_len: u32      (little-endian)
    payload: [u8]         (4-byte LE initial state, then the byte stream)

Dev-time tool, outside the content hash (fingerprint._SKIP_DIRS excludes
tools/).
"""
from __future__ import annotations

import struct

RANS_L = 1 << 23
SCALE_BITS = 14


def normalize_freqs(counts, scale_bits: int = SCALE_BITS) -> list[int]:
    """Largest-remainder normalization to sum exactly 1 << scale_bits.

    Every present symbol (count > 0) keeps frequency >= 1; absent symbols
    get 0 and are undecodable by construction.
    """
    m = 1 << scale_bits
    total = sum(counts)
    if total == 0:
        raise ValueError("empty symbol stream")
    present = [s for s in range(256) if counts[s] > 0]
    if len(present) > m:
        raise ValueError("alphabet larger than 1 << scale_bits")
    ideal = [counts[s] * m / total for s in present]
    freqs = [0] * 256
    for s, x in zip(present, ideal):
        freqs[s] = max(1, int(x))
    diff = m - sum(freqs)
    if diff != 0:
        # Distribute the remainder over the symbols with the largest
        # fractional loss (diff > 0) or the largest slack above 1 (diff < 0).
        order = sorted(present, key=lambda s: counts[s] * m / total - freqs[s],
                       reverse=diff > 0)
        i = 0
        while diff != 0:
            s = order[i % len(order)]
            step = 1 if diff > 0 else -1
            if step < 0 and freqs[s] <= 1:
                i += 1
                continue
            freqs[s] += step
            diff -= step
            i += 1
    assert sum(freqs) == m
    return freqs


def encode(data: bytes, freqs: list[int], scale_bits: int = SCALE_BITS) -> bytes:
    """rans_byte encode; returns the payload (4-byte LE state + stream)."""
    mask = (1 << scale_bits) - 1
    cum = [0] * 257
    for s in range(256):
        cum[s + 1] = cum[s] + freqs[s]
    assert cum[256] == mask + 1
    x = RANS_L
    out = bytearray()
    freqs_l, cum_l = list(freqs), cum  # local binds for the hot loop
    for s in reversed(data):
        f = freqs_l[s]
        if f == 0:
            raise ValueError(f"symbol {s} has zero frequency")
        x_max = ((RANS_L >> scale_bits) << 8) * f
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << scale_bits) + (x % f) + cum_l[s]
    out += struct.pack("<I", x)[::-1]  # bytes reversed with the stream below
    return bytes(out[::-1])


def decode(payload: bytes, freqs: list[int], n: int,
           scale_bits: int = SCALE_BITS) -> bytes:
    """rans_byte decode of n symbols; refuses malformed input loudly."""
    m = 1 << scale_bits
    if sum(freqs) != m:
        raise ValueError(f"frequency sum {sum(freqs)} != {m}")
    if len(payload) < 4:
        raise ValueError("payload shorter than the initial state")
    cum = [0] * 257
    for s in range(256):
        cum[s + 1] = cum[s] + freqs[s]
    slot2sym = bytearray(m)
    for s in range(256):
        if freqs[s]:
            slot2sym[cum[s]:cum[s + 1]] = bytes([s]) * freqs[s]
    x = struct.unpack("<I", payload[:4])[0]
    pos = 4
    mask = m - 1
    out = bytearray(n)
    stream = payload
    ln = len(stream)
    for i in range(n):
        slot = x & mask
        s = slot2sym[slot]
        f = freqs[s]
        x = f * (x >> scale_bits) + slot - cum[s]
        while x < RANS_L:
            if pos >= ln:
                raise ValueError(f"payload truncated at symbol {i}/{n}")
            x = (x << 8) | stream[pos]
            pos += 1
        out[i] = s
    if pos != ln:
        raise ValueError(f"{ln - pos} unconsumed payload bytes")
    if x != RANS_L:
        raise ValueError(f"final state {x:#x} != L")
    return bytes(out)


def pack_part(data: bytes, scale_bits: int = SCALE_BITS) -> bytes:
    """Encode one part: header + tables + payload; self-decode before return."""
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    freqs = normalize_freqs(counts, scale_bits)
    payload = encode(data, freqs, scale_bits)
    if decode(payload, freqs, len(data), scale_bits) != data:
        raise AssertionError("self-decode mismatch")  # refuse to emit
    return (bytes([scale_bits])
            + b"".join(struct.pack("<H", f) for f in freqs)
            + struct.pack("<I", len(payload))
            + payload)
