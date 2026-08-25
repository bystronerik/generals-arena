"""Convert unclejoe's own artifact (ema.eqx) to safetensors for the crate.

Dev-time tool (outside the content hash — fingerprint._SKIP_DIRS excludes
``tools/``). Forked from ``bots/joe-rs/tools/convert_artifact.py`` with one
structural difference: unclejoe has its **own weights lineage** — the X16
run ``joe-X16-gcp-20260825``, exported by ``scripts/joe_export_bot.py
--out bots/unclejoe/artifact`` — so the source ``.eqx`` and the manifest
live in ``bots/unclejoe/artifact/``, not in ``bots/joe/``. The network
*code* is still joe's: the weights go through
``eqx.tree_deserialise_leaves`` into the exact ``HistoryTransformer``
template ``bots/joe/agent.py`` builds (depth read from the manifest), so a
leaf-order bug here is impossible without also breaking the Python joe
lineage (port-plan §3, R2).

Checks, in order, refusing to write on any failure:

1. ``manifest.json``'s ``weights_sha256`` matches the ``.eqx`` file.
2. The flattened tree has exactly 276 array leaves, 29,551,834 parameters.
3. Every derived dotted name and shape matches the joe-net-v1 schema at
   depth 16.
4. Round-trip: the written safetensors reloads bit-exact against the
   deserialised leaves.

Output: ``bots/unclejoe/artifact/model.safetensors`` and the manifest
updated in place (provenance intact) with ``safetensors_sha256``, the
source ``weights_sha256``, and ``tensor_schema: joe-net-v1``.

Usage: .venv/bin/python bots/unclejoe/tools/convert_artifact.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BOT_DIR = Path(__file__).resolve().parent.parent
JOE_DIR = BOT_DIR.parent / "joe"
ARTIFACT = BOT_DIR / "artifact"
TENSOR_SCHEMA = "joe-net-v1"

# Depth of the shipped tier: X16 (depth 16, ff x4). Pinned here, not read
# from the manifest, for the same reason as the two counts below — a
# mis-shaped artifact must refuse rather than convert. Changing tier means
# editing these three plus `DEPTH` in `src/nn/net.rs` and the two counts in
# `tools/quantize_artifact.py`. joe-rs's copy of this tool stays at
# 7 / 132 / 13,581,658 (tier M7F4).
DEPTH = 16
FF_DIM = 1536  # embed 384 x ff_factor 4
EXPECTED_LEAVES = 276
EXPECTED_PARAMS = 29_551_834

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
    """joe-net-v1: name -> shape, from port-plan §3, at depth 16."""
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
        schema[f"{pre}.ff_linear1.weight"] = (FF_DIM, 384)
        schema[f"{pre}.ff_linear1.bias"] = (FF_DIM,)
        schema[f"{pre}.ff_linear2.weight"] = (384, FF_DIM)
        schema[f"{pre}.ff_linear2.bias"] = (384,)
    return schema


def main() -> None:
    import equinox as eqx
    import jax.random as jrandom
    import jax.tree_util as jtu
    from safetensors.numpy import load_file, save_file

    from joe_net import HistoryTransformer

    with open(ARTIFACT / "manifest.json") as f:
        manifest = json.load(f)

    eqx_path = ARTIFACT / manifest["weights"]
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

    out_path = ARTIFACT / "model.safetensors"
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
    manifest["tensor_schema"] = TENSOR_SCHEMA
    manifest["safetensors"] = "model.safetensors"
    manifest["safetensors_sha256"] = st_sha
    manifest["safetensors_size"] = out_path.stat().st_size
    with open(ARTIFACT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, sha256 {st_sha})")


if __name__ == "__main__":
    main()
