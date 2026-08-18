"""Round the joe artifact (ema.eqx) through IEEE f16, in place.

Dev-time tool (outside the content hash — fingerprint._SKIP_DIRS excludes
``tools/``). It is the single quantization point of the joe lineage: joe
loads the rounded ``.eqx`` directly, ``bots/joe-rs/tools/convert_artifact.py``
converts the same rounded leaves to safetensors, and the fan-out copies
those bytes downstream. No loader anywhere rounds at run time, so there is
exactly one rounding implementation to keep honest.

Why f16: the measured contrast (docs/research/measurements/
joe-rs-f16-quantization.md) excludes a regression over 3,544 rated games,
frame agreement is 99.93% over 7,206 recorded frames, and the submission
zip halves because the 13 zeroed mantissa bits deflate away. The values
stay float32 on disk and in memory; only their precision changes.

Checks, in order, refusing to write on any failure:

1. ``manifest.json``'s ``weights_sha256`` matches the ``.eqx`` file.
2. The manifest does not already carry ``quantized`` (no double rounding —
   idempotent in effect, but a second run means a confused pipeline).
3. The tree deserialises into the exact deployment template that
   ``bots/joe/agent.py`` builds: 132 float32 leaves, 11,514,586 parameters.
4. The original ``.eqx`` is backed up (default ``data/joe/``) before the
   in-place replace, and the replace goes through a temporary file.

The manifest keeps the chain of custody: ``weights_sha256`` moves to the
rounded bytes, the original digest is kept as
``pre_quantization_weights_sha256``, and ``quantized: "f16"`` marks the
artifact. The checkpoint block (the R2 key of the f32 EMA source) is
untouched.

Usage: .venv/bin/python bots/joe/tools/quantize_artifact.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JOE_DIR = Path(__file__).resolve().parent.parent
REPO = JOE_DIR.parents[1]
BACKUP_DIR = REPO / "data" / "joe"

EXPECTED_LEAVES = 132
EXPECTED_PARAMS = 11_514_586

sys.path.insert(0, str(JOE_DIR))


def main() -> None:
    import equinox as eqx
    import jax.random as jrandom
    import jax.tree_util as jtu

    from joe_net import HistoryTransformer

    manifest_path = JOE_DIR / "artifact" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    if "quantized" in manifest:
        raise SystemExit(
            f"manifest already says quantized={manifest['quantized']!r}; "
            f"re-quantizing a rounded artifact means the pipeline ran twice. "
            f"Restore the f32 .eqx first (pre_quantization_weights_sha256 "
            f"names it)."
        )

    eqx_path = JOE_DIR / "artifact" / manifest["weights"]
    eqx_bytes = eqx_path.read_bytes()
    sha = hashlib.sha256(eqx_bytes).hexdigest()
    if sha != manifest["weights_sha256"]:
        raise SystemExit(
            f"sha256 mismatch on {eqx_path}: {sha} != manifest "
            f"{manifest['weights_sha256']}")
    print(f"source sha256 verified: {sha}")

    # The exact template agent.py builds (agent.py __init__), the same code
    # path convert_artifact.py uses — a leaf-order bug here is impossible
    # without also breaking the deployed Python bot.
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

    counted = {"leaves": 0, "params": 0, "changed": 0}

    def round_leaf(leaf):
        if not eqx.is_array(leaf):
            return leaf
        arr = np.asarray(leaf)
        if arr.dtype != np.float32:
            raise SystemExit(f"unexpected dtype {arr.dtype} in tree")
        counted["leaves"] += 1
        counted["params"] += int(arr.size)
        rounded = arr.astype(np.float16).astype(np.float32)
        counted["changed"] += int(np.count_nonzero(rounded != arr))
        return rounded

    quantized = jtu.tree_map(round_leaf, net)
    if counted["leaves"] != EXPECTED_LEAVES or counted["params"] != EXPECTED_PARAMS:
        raise SystemExit(
            f"leaf/param mismatch: {counted['leaves']} leaves, "
            f"{counted['params']} params; expected "
            f"{EXPECTED_LEAVES}/{EXPECTED_PARAMS}")
    print(
        f"rounded {counted['params']} params in {counted['leaves']} leaves; "
        f"{counted['changed']} values changed")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"ema.f32.{sha[:12]}.eqx"
    if not backup.exists():
        shutil.copy2(eqx_path, backup)
    print(f"f32 original backed up at {backup}")

    tmp = eqx_path.with_suffix(".eqx.tmp")
    eqx.tree_serialise_leaves(str(tmp), quantized)

    # Prove the file reloads to exactly the rounded leaves before replacing.
    reloaded = eqx.tree_deserialise_leaves(str(tmp), template)
    for (_, a), (_, b) in zip(
        jtu.tree_flatten_with_path(eqx.filter(quantized, eqx.is_array))[0][0:],
        jtu.tree_flatten_with_path(eqx.filter(reloaded, eqx.is_array))[0][0:],
        strict=True,
    ):
        if not np.array_equal(
            np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32)
        ):
            tmp.unlink()
            raise SystemExit("round-trip mismatch; original left untouched")
    tmp.replace(eqx_path)
    print("round-trip bit-exact: ok")

    new_bytes = eqx_path.read_bytes()
    manifest["pre_quantization_weights_sha256"] = sha
    manifest["weights_sha256"] = hashlib.sha256(new_bytes).hexdigest()
    manifest["weights_size"] = len(new_bytes)
    manifest["quantized"] = "f16"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(
        f"wrote {eqx_path} ({len(new_bytes)} bytes, "
        f"sha256 {manifest['weights_sha256']})")


if __name__ == "__main__":
    main()
