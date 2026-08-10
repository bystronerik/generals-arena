#!/usr/bin/env python3
"""Convert the frozen Python-morpheus TorchScript artifact to safetensors.

    python bots/morpheus-rs/tools/convert_artifact.py
    python bots/morpheus-rs/tools/convert_artifact.py --check

Dev-time only. It uses torch, it never runs during a match, and it lives under
`tools/` so it stays outside this bot's content hash — editing the converter
must not re-identify the bot, only re-running it and committing a different
`artifact/` may (packaging.md).

**Direction of the artifact contract (rewrite-plan §3).** Training and its
TorchScript exports are untouched. This reads the *committed* artifact of the
frozen oracle, verifies every digest the manifest claims, and writes a
weights-only safetensors file the Rust bot can mmap. Nothing that consumes the
`.pt` files changes, so a new checkpoint means re-running this, not editing an
exporter.

**Why weights-only is enough.** safetensors carries tensors and no graph. The
graph lives in `crates/core/src/nn/network.rs`, written against `network.py` and
proved against TorchScript outputs by the `net` parity surface. That is the
same split the plan chose over ONNX: no translation layer between two graph
formats, and the numerics are ours to control.

Key names drop TorchScript's `model.` prefix so they read exactly as the
`network.py` module paths (`stem.weight`, `blocks.7.dw.weight`, `wdl.bias`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
ORACLE_ARTIFACT = REPO / "bots" / "morpheus" / "artifact"
DEST_ARTIFACT = BOT_DIR / "artifact"

SAFETENSORS_NAME = "model.safetensors"
MANIFEST_NAME = "manifest.json"

# The Rust loader refuses to start unless these match, the same guardrails
# `inference.validate_manifest` applies on the Python side.
ARCHITECTURE_VERSION = "morpheus-net-v1"
TENSOR_SCHEMA_VERSION = "morpheus-tensor-v1"
ACTION_SCHEMA_VERSION = "morpheus-action-v1"

TORCHSCRIPT_PREFIX = "model."

# What the graph in `network.rs` expects to find, checked here rather than
# discovered as a panic at warmup. Shapes are derived from the manifest's own
# architecture block so a re-trained checkpoint with different widths fails
# with a diff instead of loading and producing nonsense.
GROUP_NORM_GROUPS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_keys(arch: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Every tensor `network.rs` reads, with the shape the manifest implies."""
    ch = int(arch["trunk_channels"])
    mid = int(arch["expansion"])
    cin = int(arch["in_channels"])
    n_blocks = int(arch["n_blocks"])
    n_bins = int(arch["n_army_bins"])
    n_policy = int(arch["policy_channels"])
    feat = ch * 2

    keys: dict[str, tuple[int, ...]] = {
        "stem.weight": (ch, cin, 3, 3),
        "gn_stem.weight": (ch,),
        "gn_stem.bias": (ch,),
    }
    for i in range(n_blocks):
        keys.update(
            {
                f"blocks.{i}.pw_expand.weight": (mid, ch, 1, 1),
                f"blocks.{i}.gn_expand.weight": (mid,),
                f"blocks.{i}.gn_expand.bias": (mid,),
                f"blocks.{i}.dw.weight": (mid, 1, 3, 3),
                f"blocks.{i}.gn_dw.weight": (mid,),
                f"blocks.{i}.gn_dw.bias": (mid,),
                f"blocks.{i}.pw_project.weight": (ch, mid, 1, 1),
                f"blocks.{i}.gn_project.weight": (ch,),
                f"blocks.{i}.gn_project.bias": (ch,),
            }
        )
    for name, out_ch in (
        ("policy", n_policy),
        ("hidden_owner", 1),
        ("enemy_army_bins", n_bins),
        ("enemy_general", 1),
        ("hidden_castle", 1),
    ):
        keys[f"{name}.weight"] = (out_ch, ch, 1, 1)
        keys[f"{name}.bias"] = (out_ch,)
    for name, out in (
        ("pass_fc", 1),
        ("wdl", 3),
        ("land_margin", 1),
        ("army_margin", 1),
        ("castle_margin", 1),
        ("turns_to_termination", 1),
    ):
        keys[f"{name}.weight"] = (out, feat)
        keys[f"{name}.bias"] = (out,)
    return keys


def verify_source_manifest(manifest: dict[str, Any], src: Path) -> None:
    """Every digest the source manifest claims, re-derived from the files."""
    for key in ("architecture_version", "tensor_schema", "action_schema"):
        if key not in manifest:
            raise SystemExit(f"source manifest missing {key!r}")
    checks = (
        ("architecture_version", ARCHITECTURE_VERSION),
        ("tensor_schema", TENSOR_SCHEMA_VERSION),
        ("action_schema", ACTION_SCHEMA_VERSION),
    )
    for key, want in checks:
        if manifest[key] != want:
            raise SystemExit(f"{key}: {manifest[key]!r} != {want!r}")

    artifact = src / str(manifest["artifact_file"])
    digest = sha256_file(artifact)
    if digest != manifest["weights_sha256"]:
        raise SystemExit(
            f"weights_sha256 mismatch for {artifact.name}: "
            f"manifest {manifest['weights_sha256']} != file {digest}"
        )
    # The online entry points are not read by the Rust bot — one graph serves
    # all three — but a mismatch here means the committed artifact directory is
    # inconsistent, and converting from an inconsistent directory is worse than
    # failing.
    for label, digest_want in (manifest.get("online_weights_sha256") or {}).items():
        name = (manifest.get("online_entry_points") or {}).get(label)
        if not name:
            raise SystemExit(f"online_weights_sha256 has {label!r} with no entry point")
        got = sha256_file(src / str(name))
        if got != digest_want:
            raise SystemExit(
                f"online weights mismatch for {name}: {digest_want} != {got}"
            )


def extract_state_dict(artifact: Path) -> dict[str, Any]:
    import torch  # local: this file is the only place in the bot that needs it

    module = torch.jit.load(str(artifact), map_location="cpu")
    module.eval()
    out: dict[str, Any] = {}
    for key, value in module.state_dict().items():
        name = key[len(TORCHSCRIPT_PREFIX):] if key.startswith(TORCHSCRIPT_PREFIX) else key
        if value.dtype != torch.float32:
            raise SystemExit(f"{name}: expected float32, got {value.dtype}")
        # `.contiguous()` is not cosmetic. safetensors writes the raw buffer,
        # and the Rust loader reads row-major with no stride metadata, so a
        # non-contiguous tensor would arrive silently permuted.
        out[name] = value.detach().contiguous().clone()
    return out


def check_against_manifest(
    tensors: dict[str, Any], manifest: dict[str, Any]
) -> None:
    want = expected_keys(manifest["architecture"])
    got = {k: tuple(v.shape) for k, v in tensors.items()}
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if missing:
        raise SystemExit(f"artifact missing tensors: {missing}")
    if extra:
        raise SystemExit(f"artifact has tensors the graph does not read: {extra}")
    for key, shape in sorted(want.items()):
        if got[key] != shape:
            raise SystemExit(f"{key}: shape {got[key]} != manifest-implied {shape}")

    total = sum(int(v.numel()) for v in tensors.values())
    claimed = int(manifest["architecture"]["parameter_count"])
    if total != claimed:
        raise SystemExit(f"parameter count {total} != manifest {claimed}")

    arch = manifest["architecture"]
    for name, width in (("trunk_channels", arch["trunk_channels"]), ("expansion", arch["expansion"])):
        if int(width) % GROUP_NORM_GROUPS:
            raise SystemExit(f"{name}={width} not divisible by {GROUP_NORM_GROUPS}")


def serialize(tensors: dict[str, Any], *, metadata: dict[str, str]) -> bytes:
    """A byte-reproducible safetensors blob.

    `safetensors.torch.save_file` is not usable here: it serializes the
    `__metadata__` block from a Rust `HashMap`, whose iteration order varies
    between processes, so two conversions of the same weights produce two
    different files. That would be a nuisance anywhere and a real problem here
    — `artifact/` is inside the bot's content hash, so a re-conversion that
    changed no weight would still mint a new bot identity, and `--check` could
    never distinguish "the artifact drifted" from "the library reshuffled a
    dict".

    The format is small enough to own: `<u64 little-endian header length>`,
    that many bytes of JSON, then tensor buffers at the byte offsets the header
    names, relative to the end of the header. Writing it here fixes the key
    order (sorted, in both the header and the data region) and the JSON
    separators. `verify_reference_reader` checks the result against the real
    library, so this owns the layout without owning the specification.
    """
    header: dict[str, Any] = {"__metadata__": dict(sorted(metadata.items()))}
    buffers: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        value = tensors[name]
        raw = value.numpy().tobytes()  # contiguous by construction, row-major
        header[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        buffers.append(raw)
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return len(text).to_bytes(8, "little") + text + b"".join(buffers)


def verify_reference_reader(path: Path, tensors: dict[str, Any]) -> None:
    """Round-trip the hand-written blob through the official reader.

    The Rust loader makes the same layout assumptions this writer does, so if
    both were wrong in the same way nothing downstream would notice. The
    reference implementation is the third opinion.
    """
    from safetensors.torch import load_file  # local, dev-time only

    loaded = load_file(str(path))
    if set(loaded) != set(tensors):
        raise SystemExit("safetensors round trip lost or invented tensors")
    for name, want in tensors.items():
        got = loaded[name]
        if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype:
            raise SystemExit(f"{name}: round trip changed shape or dtype")
        if not got.equal(want):
            raise SystemExit(f"{name}: round trip changed values")


def build_dest_manifest(
    source: dict[str, Any], *, safetensors_digest: str, source_digest: str
) -> dict[str, Any]:
    manifest = json.loads(json.dumps(source))  # deep copy, no aliasing
    manifest["artifact_file"] = SAFETENSORS_NAME
    manifest["weights_sha256"] = safetensors_digest
    manifest["runtime"] = "morpheus-rs-native+float32"
    manifest["quantization"] = {
        "engine": "none",
        "format": "float32",
        "runtime": "morpheus-rs-native+float32",
    }
    # Provenance, so a divergence can be traced to the exact `.pt` this came
    # from rather than to "some export".
    manifest["source_artifact"] = {
        "bot": "morpheus",
        "file": str(source["artifact_file"]),
        "sha256": source_digest,
        "runtime": str(source["runtime"]),
    }
    # The Rust bot has one graph, not three entry-point modules; leaving these
    # in would name files that do not exist beside the manifest.
    manifest.pop("online_entry_points", None)
    manifest.pop("online_weights_sha256", None)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ORACLE_ARTIFACT,
        help="artifact directory of the frozen Python oracle",
    )
    parser.add_argument("--dest", type=Path, default=DEST_ARTIFACT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed conversion reproduces byte-for-byte; write nothing",
    )
    args = parser.parse_args()

    src_manifest_path = args.source / MANIFEST_NAME
    if not src_manifest_path.is_file():
        raise SystemExit(f"missing source manifest: {src_manifest_path}")
    source = json.loads(src_manifest_path.read_text())
    verify_source_manifest(source, args.source)

    artifact = args.source / str(source["artifact_file"])
    source_digest = sha256_file(artifact)
    tensors = extract_state_dict(artifact)
    check_against_manifest(tensors, source)

    args.dest.mkdir(parents=True, exist_ok=True)
    out_path = args.dest / SAFETENSORS_NAME
    staged = out_path.with_suffix(".safetensors.tmp")
    blob = serialize(tensors, metadata={
        "architecture_version": ARCHITECTURE_VERSION,
        "source_sha256": source_digest,
    })
    staged.write_bytes(blob)
    verify_reference_reader(staged, tensors)
    digest = sha256_file(staged)
    dest_manifest = build_dest_manifest(
        source, safetensors_digest=digest, source_digest=source_digest
    )
    manifest_text = json.dumps(dest_manifest, indent=2, sort_keys=True) + "\n"

    if args.check:
        staged.unlink()
        problems = []
        if not out_path.is_file():
            problems.append(f"missing {out_path}")
        elif sha256_file(out_path) != digest:
            problems.append(f"{out_path.name} differs from a fresh conversion")
        dest_manifest_path = args.dest / MANIFEST_NAME
        if not dest_manifest_path.is_file():
            problems.append(f"missing {dest_manifest_path}")
        elif dest_manifest_path.read_text() != manifest_text:
            problems.append(f"{MANIFEST_NAME} differs from a fresh conversion")
        if problems:
            for line in problems:
                print(f"convert_artifact --check: {line}", file=sys.stderr)
            return 1
        print(f"convert_artifact --check: committed artifact matches ({digest[:12]})")
        return 0

    shutil.move(str(staged), str(out_path))
    (args.dest / MANIFEST_NAME).write_text(manifest_text)
    print(
        f"wrote {out_path.relative_to(REPO)} "
        f"({out_path.stat().st_size} bytes, {len(tensors)} tensors, sha256 {digest[:12]})"
    )
    print(f"source {artifact.relative_to(REPO)} sha256 {source_digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
