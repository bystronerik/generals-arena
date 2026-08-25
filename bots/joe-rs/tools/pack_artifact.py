#!/usr/bin/env python3
"""Pack model.safetensors into the joe-net-v2 container (model.packed).

The pack side of docs/bots/joe-rs/rans-plan.md P3: split every tensor's
f16 values into two byte planes, encode each part with whichever codec is
smaller — `rans0` (tools/rans.py) or raw plane bytes — and frame the
parts behind the original safetensors header, copied verbatim. The Rust
side (`joe-rs unpack-artifact`, src/nn/pack.rs) reconstructs the
safetensors byte-for-byte at intake and refuses on any digest mismatch.

Refusals, in order, nothing written on any failure:

1. The manifest does not carry ``quantized: "f16"`` — an f16 container
   must never be the thing that rounds.
2. ``safetensors_sha256`` does not match the file on disk.
3. Any tensor is not F32, or any value does not survive the f32 -> f16
   -> f32 round trip bit-exactly.
4. The packed container, decoded back in Python, is not byte-identical
   to the original file.

On success: writes ``model.packed`` and rewrites ``manifest.json`` with
``pack_format``, ``packed``, and ``packed_sha256``. ``model.safetensors``
stays on disk — dev-side tools keep reading it; only the submission zip
omits it.

Usage: .venv/bin/python bots/joe-rs/tools/pack_artifact.py [artifact_dir]

Dev-time tool, outside the content hash (fingerprint._SKIP_DIRS excludes
``tools/``).
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rans  # noqa: E402

MAGIC = b"JNP2"
CODEC_RAW = 0
CODEC_RANS0 = 1


def parse_safetensors(raw: bytes):
    header_len = int.from_bytes(raw[:8], "little")
    header_block = raw[: 8 + header_len]
    header = json.loads(raw[8 : 8 + header_len])
    data = raw[8 + header_len :]
    tensors = []
    for name, ent in header.items():
        if name == "__metadata__":
            continue
        if ent["dtype"] != "F32":
            raise SystemExit(f"{name}: dtype {ent['dtype']} is not F32")
        b, e = ent["data_offsets"]
        tensors.append((b, e, name))
    tensors.sort()
    expect = 0
    for b, e, name in tensors:
        if b != expect:
            raise SystemExit(f"{name}: data_offsets not contiguous")
        expect = e
    if expect != len(data):
        raise SystemExit("data section length disagrees with the header")
    return header_block, [(name, data[b:e]) for b, e, name in tensors]


def pack_plane(plane: bytes) -> bytes:
    blob = rans.pack_part(plane)  # self-decodes before returning
    raw_cost = 1 + 4 + len(plane)
    if raw_cost < 1 + len(blob):
        return bytes([CODEC_RAW]) + struct.pack("<I", len(plane)) + plane
    return bytes([CODEC_RANS0]) + blob


def unpack_plane(packed: bytes, at: int, n: int) -> tuple[bytes, int]:
    """Python mirror of the Rust decode_plane, for the self-verify pass."""
    codec = packed[at]
    at += 1
    if codec == CODEC_RAW:
        (length,) = struct.unpack_from("<I", packed, at)
        assert length == n
        return packed[at + 4 : at + 4 + n], at + 4 + n
    assert codec == CODEC_RANS0
    scale_bits = packed[at]
    freqs = list(struct.unpack_from("<256H", packed, at + 1))
    (payload_len,) = struct.unpack_from("<I", packed, at + 513)
    payload = packed[at + 517 : at + 517 + payload_len]
    return rans.decode(payload, freqs, n, scale_bits), at + 517 + payload_len


def main() -> None:
    art = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "artifact"
    manifest_path = art / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("quantized") != "f16":
        raise SystemExit("manifest has no quantized: \"f16\" — run quantize_artifact.py first")
    st_name = manifest["safetensors"]
    raw = (art / st_name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["safetensors_sha256"]:
        raise SystemExit(f"{st_name} sha256 {digest} != manifest safetensors_sha256")

    header_block, tensors = parse_safetensors(raw)
    out = bytearray(MAGIC)
    out += header_block
    n_rans = n_raw = 0
    for name, tbytes in tensors:
        f32 = np.frombuffer(tbytes, np.float32)
        f16 = f32.astype(np.float16)
        if f16.astype(np.float32).tobytes() != tbytes:
            raise SystemExit(f"{name}: values do not survive the f16 round trip")
        planes = np.frombuffer(f16.tobytes(), np.uint8).reshape(-1, 2)
        for c in (0, 1):
            part = pack_plane(planes[:, c].tobytes())
            if part[0] == CODEC_RANS0:
                n_rans += 1
            else:
                n_raw += 1
            out += part

    # Self-verify: reconstruct the exact original file from the container.
    packed = bytes(out)
    at = len(MAGIC) + len(header_block)
    rebuilt = bytearray()
    for name, tbytes in tensors:
        n = len(tbytes) // 4
        lo, at = unpack_plane(packed, at, n)
        hi, at = unpack_plane(packed, at, n)
        f16 = np.empty(n, np.uint16)
        f16[:] = np.frombuffer(lo, np.uint8).astype(np.uint16)
        f16 |= np.frombuffer(hi, np.uint8).astype(np.uint16) << 8
        rebuilt += f16.view(np.float16).astype(np.float32).tobytes()
    assert at == len(packed)
    if bytes(header_block) + bytes(rebuilt) != raw:
        raise SystemExit("self-verify failed: the container does not reproduce the file")

    (art / "model.packed").write_bytes(packed)
    manifest["pack_format"] = "joe-net-v2"
    manifest["packed"] = "model.packed"
    manifest["packed_sha256"] = hashlib.sha256(packed).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"model.packed: {len(packed):,} B from {len(raw):,} B "
          f"({n_rans} rans0 parts, {n_raw} raw), sha256 {manifest['packed_sha256']}")


if __name__ == "__main__":
    main()
