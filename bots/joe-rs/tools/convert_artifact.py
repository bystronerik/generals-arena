"""Convert the committed joe artifact (ema.eqx) to safetensors for joe-rs.

Dev-time tool (outside the content hash — fingerprint._SKIP_DIRS excludes
``tools/``). It never parses the ``.eqx`` framing itself: the weights go
through ``eqx.tree_deserialise_leaves`` into the exact deployment template
that ``bots/joe/agent.py`` builds, so a leaf-order bug here is impossible
without also breaking the deployed Python bot (port-plan §3, R2).

Checks, in order, refusing to write on any failure:

1. ``manifest.json``'s ``weights_sha256`` matches the ``.eqx`` file.
2. The flattened tree has exactly 132 array leaves, 11,514,586 parameters.
3. Every derived dotted name and shape matches the joe-net-v1 schema.
4. Round-trip: the written safetensors reloads bit-exact against the
   deserialised leaves.

Output: ``bots/joe-rs/artifact/model.safetensors`` and a manifest that
copies joe's (provenance intact) plus ``safetensors_sha256``, the source
``weights_sha256``, and ``tensor_schema: joe-net-v1``.

Usage: .venv/bin/python bots/joe-rs/tools/convert_artifact.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

JOE_RS_DIR = Path(__file__).resolve().parent.parent
JOE_DIR = JOE_RS_DIR.parent / "joe"
TENSOR_SCHEMA = "joe-net-v1"

# Depth of the shipped tier: M7 (depth 7). Pinned here, not read from the
# manifest, for the same reason as the two counts below — a mis-shaped
# artifact must refuse rather than convert. Changing tier means editing
# these three plus `DEPTH` in `src/nn/net.rs` of both Rust crates.
DEPTH = 7
EXPECTED_LEAVES = 132
EXPECTED_PARAMS = 11_514_586

sys.path.insert(0, str(JOE_DIR))


def _dotted_name(path) -> str:
    import jax.tree_util as jtu

    parts = []
    for key in path:
        if isinstance(key, jtu.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, jtu.SequenceKey):
            parts.append(str(key.idx))
        elif isinstance(key, jtu.DictKey):
            parts.append(str(key.key))
        else:
            raise ValueError(f"unhandled path key {key!r}")
    return ".".join(parts)


def expected_schema() -> dict:
    """joe-net-v1: name -> shape, from port-plan §3."""
    schema = {
        "embedder.weight": (384, 351),
        "embedder.bias": (384,),
        "value_token": (1, 384),
        "pos_encoding": (52, 384),
        "norm_out.weight": (384,),
        "norm_out.bias": (384,),
        "policy_head.weight": (90, 384),
        "policy_head.bias": (90,),
        "value_head.weight": (128, 384),
        "value_head.bias": (128,),
        "temporal_type_embed": (2, 384),
        "bin_centers": (128,),
    }
    for side in ("army", "land"):
        schema[f"temporal_encoder.{side}_l1.weight"] = (512, 512)
        schema[f"temporal_encoder.{side}_l1.bias"] = (512,)
        schema[f"temporal_encoder.{side}_l2.weight"] = (384, 512)
        schema[f"temporal_encoder.{side}_l2.bias"] = (384,)
    for i in range(DEPTH):
        pre = f"transformer_layers.{i}"
        schema[f"{pre}.norm1.weight"] = (384,)
        schema[f"{pre}.norm1.bias"] = (384,)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            schema[f"{pre}.attn.{proj}.weight"] = (384, 384)
            schema[f"{pre}.attn.{proj}.bias"] = (384,)
        schema[f"{pre}.norm2.weight"] = (384,)
        schema[f"{pre}.norm2.bias"] = (384,)
        schema[f"{pre}.ff_linear1.weight"] = (1152, 384)
        schema[f"{pre}.ff_linear1.bias"] = (1152,)
        schema[f"{pre}.ff_linear2.weight"] = (384, 1152)
        schema[f"{pre}.ff_linear2.bias"] = (384,)
    return schema


def main() -> None:
    import equinox as eqx
    import jax.random as jrandom
    import jax.tree_util as jtu
    from safetensors.numpy import load_file, save_file

    from joe_net import HistoryTransformer

    with open(JOE_DIR / "artifact" / "manifest.json") as f:
        manifest = json.load(f)

    eqx_path = JOE_DIR / "artifact" / manifest["weights"]
    eqx_bytes = eqx_path.read_bytes()
    sha = hashlib.sha256(eqx_bytes).hexdigest()
    if sha != manifest["weights_sha256"]:
        raise SystemExit(
            f"sha256 mismatch on {eqx_path}: {sha} != manifest "
            f"{manifest['weights_sha256']}")
    print(f"source sha256 verified: {sha}")

    # The exact template agent.py builds (agent.py __init__).
    arch = manifest["network"]
    template = HistoryTransformer(
        grid_size=int(arch["pad_to"]),
        pad_to=int(arch["pad_to"]),
        history_size=int(arch["history_size"]),
        patch_size=int(arch["patch_size"]),
        depth=int(arch["depth"]),
        embed_dim=int(arch["embed_dim"]),
        n_head=int(arch["n_head"]),
        ff_factor=int(arch["ff_factor"]),
        use_bf16=False,
        value_loss=arch["value_loss"],
        num_bins=int(arch["num_bins"]),
        v_min=float(arch["v_min"]),
        v_max=float(arch["v_max"]),
        key=jrandom.PRNGKey(0),
    )
    net = eqx.tree_deserialise_leaves(str(eqx_path), template)

    arrays = eqx.filter(net, eqx.is_array)
    leaves, _ = jtu.tree_flatten_with_path(arrays)
    tensors = {}
    for path, leaf in leaves:
        name = _dotted_name(path)
        arr = np.asarray(leaf)
        if arr.dtype != np.float32:
            raise SystemExit(f"{name}: dtype {arr.dtype}, expected float32")
        tensors[name] = arr

    schema = expected_schema()
    if set(tensors) != set(schema):
        missing = sorted(set(schema) - set(tensors))
        extra = sorted(set(tensors) - set(schema))
        raise SystemExit(f"schema mismatch: missing={missing} extra={extra}")
    for name, shape in schema.items():
        if tensors[name].shape != shape:
            raise SystemExit(
                f"{name}: shape {tensors[name].shape}, expected {shape}")

    n_leaves = len(tensors)
    n_params = sum(int(t.size) for t in tensors.values())
    if n_leaves != EXPECTED_LEAVES or n_params != EXPECTED_PARAMS:
        raise SystemExit(
            f"leaf/param mismatch: {n_leaves} leaves, {n_params} params; "
            f"expected {EXPECTED_LEAVES}/{EXPECTED_PARAMS}")
    if n_params != manifest["n_params"]:
        raise SystemExit(
            f"param sum {n_params} != manifest n_params {manifest['n_params']}")
    print(f"schema verified: {n_leaves} tensors, {n_params} params")

    out_dir = JOE_RS_DIR / "artifact"
    out_path = out_dir / "model.safetensors"
    save_file(tensors, str(out_path), metadata={"tensor_schema": TENSOR_SCHEMA})

    # Round-trip: reload and compare every leaf bit-exact.
    reloaded = load_file(str(out_path))
    for name, arr in tensors.items():
        back = reloaded[name]
        if back.dtype != arr.dtype or back.shape != arr.shape or \
                not np.array_equal(back.view(np.uint32), arr.view(np.uint32)):
            raise SystemExit(f"round-trip mismatch on {name}")
    print("round-trip bit-exact: ok")

    st_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    out_manifest = dict(manifest)
    out_manifest["tensor_schema"] = TENSOR_SCHEMA
    out_manifest["safetensors"] = "model.safetensors"
    out_manifest["safetensors_sha256"] = st_sha
    out_manifest["safetensors_size"] = out_path.stat().st_size
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(out_manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, sha256 {st_sha})")


if __name__ == "__main__":
    main()
