#!/usr/bin/env python3
"""Model-size ceiling for joe-rs — the optimized Rust stack (Modal x86 CPU).

Rust sibling of scripts/joe_modal_size_ceiling.py: the same AverageJoe-ratio
size ladder (M, M7F4, L, L16, XL20, XL24), measured through joe-rs's own
`bench --stages` harness — AVX2+FMA intrinsics gemm, PackedB strip-major
weights, the residual-sweep kernels; everything the deployed bot runs.

joe-rs pins its shapes as consts on purpose (net.rs cross-checks the
manifest), so each tier is its own build: at image-build time the crate is
copied per tier, `EMBED` / `DEPTH` / `FF_DIM` and the manifest `ff_factor`
check are patched with sed, and `cargo build --release` produces one binary
per tier (target-cpu=x86-64-v3 from .cargo/config.toml, toolchain 1.97.1).
A wrong sed cannot pass silently: the per-tier manifest is generated from
the same numbers, and a const/manifest mismatch refuses to load.

Weights are random (latency does not depend on the values): a generated
F32 safetensors + manifest per tier, matching tools/convert_artifact.py's
expected_schema() generalized to the tier's dims.

Every tier runs interleaved (pass-level round robin) inside ONE container —
Modal per-core speed spans ~2x across hosts, so only same-host contrasts
count; M7F4 anchors to the known joe-rs numbers. Input is the recorded
synthetic-long wire log, the same stream the parity gate replays. Strict
(cpu limit 1.0, 2 GB) mirrors the competition shape; burst-4 is the
CFS-throttle-free control.

Usage (never pipe through tail/head — redirect to a file, AGENTS.md):

    .venv/bin/modal run scripts/joe_rs_modal_size_ceiling.py > /tmp/joe_rs_size.log 2>&1 &
    .venv/bin/modal app list      # then check startup a few minutes in
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
IN_LOG = REPO / "data" / "joe" / "joe-rs-parity" / "games" / "synthetic-long.in.log"

# Same ladder as scripts/joe_modal_size_ceiling.py. Cost model
# depth * embed^2 * (4 + 2 * ff), normalized to M in the comment.
TIERS = {
    "M":    dict(depth=5,  embed=384, ff=3),  # 1.0x
    "M7F4": dict(depth=7,  embed=384, ff=4),  # 1.68x — the shipped net, anchor
    "L":    dict(depth=11, embed=448, ff=3),  # 3.0x
    "L16":  dict(depth=16, embed=480, ff=3),  # 5.0x
    "XL20": dict(depth=20, embed=512, ff=3),  # 7.1x
    "XL24": dict(depth=24, embed=528, ff=3),  # 9.1x
}

N_PASSES = 3

CARGO = "PATH=/root/.cargo/bin:$PATH"


def _build_commands() -> list[str]:
    """One patched copy + release build per tier; binaries land in /root/bins."""
    cmds = ["mkdir -p /root/bins"]
    for name, t in TIERS.items():
        ff_dim = t["embed"] * t["ff"]
        cmds.append(
            f"cp -r /root/joe-rs /root/b && cd /root/b"
            f" && sed -i 's/pub const EMBED: usize = 384;/pub const EMBED: usize = {t['embed']};/' src/nn/net.rs"
            f" && sed -i 's/pub const DEPTH: usize = 7;/pub const DEPTH: usize = {t['depth']};/' src/nn/net.rs"
            f" && sed -i 's/pub const FF_DIM: usize = 1536;/pub const FF_DIM: usize = {ff_dim};/' src/nn/net.rs"
            f" && sed -i 's/(\"ff_factor\", 4)/(\"ff_factor\", {t['ff']})/' src/nn/net.rs"
            f" && grep -q 'EMBED: usize = {t['embed']};' src/nn/net.rs"
            f" && grep -q 'DEPTH: usize = {t['depth']};' src/nn/net.rs"
            f" && {CARGO} cargo build --release"
            f" && cp target/release/joe-rs /root/bins/joe-rs-{name}"
            f" && cd / && rm -rf /root/b"
        )
    return cmds


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential")
    .pip_install("numpy==2.4.6")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain 1.97.1",
        f"{CARGO} rustup component add rustfmt clippy",
        f"{CARGO} rustup target add x86_64-unknown-linux-musl",
    )
    .add_local_dir(str(REPO / "bots" / "joe-rs"), "/root/joe-rs", copy=True,
                   ignore=["target", "__pycache__", "tests", "artifact"])
    .run_commands(*_build_commands())
    .add_local_file(str(IN_LOG), "/root/synthetic-long.in.log", copy=True)
)

app = modal.App("joe-rs-size-ceiling")


def _write_artifact(dir_path, depth: int, embed: int, ff: int) -> None:
    """Random-weight joe-net-v1 artifact at the tier's dims (F32 safetensors)."""
    import numpy as np

    ff_dim = embed * ff
    rng = np.random.default_rng(11)

    def w(shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    tensors: dict = {}

    def linear(name, out_dim, in_dim):
        tensors[f"{name}.weight"] = w((out_dim, in_dim))
        tensors[f"{name}.bias"] = w((out_dim,))

    def norm(name, dim):
        tensors[f"{name}.weight"] = np.ones(dim, dtype=np.float32)
        tensors[f"{name}.bias"] = np.zeros(dim, dtype=np.float32)

    linear("embedder", embed, 351)                 # N_CHANNELS 39 * 3 * 3
    tensors["value_token"] = w((1, embed))
    tensors["pos_encoding"] = w((52, embed))       # 49 patches + 3
    for i in range(depth):
        p = f"transformer_layers.{i}"
        norm(f"{p}.norm1", embed)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(f"{p}.attn.{proj}", embed, embed)
        norm(f"{p}.norm2", embed)
        linear(f"{p}.ff_linear1", ff_dim, embed)
        linear(f"{p}.ff_linear2", embed, ff_dim)
    norm("norm_out", embed)
    linear("policy_head", 90, embed)
    linear("value_head", 128, embed)
    for side in ("army", "land"):
        linear(f"temporal_encoder.{side}_l1", 512, 512)
        linear(f"temporal_encoder.{side}_l2", embed, 512)
    tensors["temporal_type_embed"] = w((2, embed))
    tensors["bin_centers"] = np.linspace(-1.0, 1.0, 128, dtype=np.float32)

    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        data = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    head = json.dumps(header).encode()
    with open(dir_path / "model.safetensors", "wb") as f:
        f.write(len(head).to_bytes(8, "little"))
        f.write(head)
        for b in blobs:
            f.write(b)

    n_params = sum(int(a.size) for a in tensors.values())
    manifest = {
        "tensor_schema": "joe-net-v1",
        "n_params": n_params,
        "network": {
            "network": "history_transformer",
            "pad_to": 21, "history_size": 7,
            "depth": depth, "embed_dim": embed, "n_head": 8,
            "ff_factor": ff, "patch_size": 3, "num_bins": 128,
        },
        "note": "random weights, size-ceiling latency bench only",
    }
    (dir_path / "manifest.json").write_text(json.dumps(manifest, indent=1))


def _bench_impl(label: str) -> dict:
    import os
    import platform
    import subprocess
    from pathlib import Path as P

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "RAYON_NUM_THREADS"):
        os.environ[var] = "1"

    def log(msg):
        print(f"[rs-size:{label}] {msg}", flush=True)

    cpu_model = "unknown"
    for line in P("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    log(f"machine={platform.machine()} cpu={cpu_model}")
    assert platform.machine() == "x86_64", platform.machine()

    arts = P("/root/artifacts")
    for name, t in TIERS.items():
        d = arts / name
        d.mkdir(parents=True, exist_ok=True)
        _write_artifact(d, t["depth"], t["embed"], t["ff"])
        n_params = json.loads((d / "manifest.json").read_text())["n_params"]
        log(f"artifact {name}: {n_params / 1e6:.2f}M params")

    results: dict = {name: [] for name in TIERS}
    for p in range(N_PASSES):
        for name in TIERS:
            env = dict(os.environ, JOE_RS_ARTIFACT=str(arts / name))
            with open("/root/synthetic-long.in.log", "rb") as f:
                run = subprocess.run(
                    [f"/root/bins/joe-rs-{name}", "bench", "--stages"],
                    stdin=f, capture_output=True, text=True, env=env)
            if run.returncode != 0:
                log(f"FAIL {name} pass {p + 1}: {run.stderr[-2000:]}")
                results[name].append({"error": run.stderr[-2000:]})
                continue
            stats = json.loads(run.stdout.strip().splitlines()[-1])
            results[name].append(stats)
            fwd, tot = stats["stages"]["forward"], stats["stages"]["total"]
            log(f"pass {p + 1}/{N_PASSES} {name}: forward p50 {fwd['p50']:.1f} "
                f"p99 {fwd['p99']:.1f} | total p50 {tot['p50']:.1f} "
                f"p99 {tot['p99']:.1f} max {tot['max']:.1f}")

    return {"label": label, "cpu_model": cpu_model, "tiers": TIERS,
            "passes": results}


@app.function(image=IMAGE, cpu=(1.0, 1.0), memory=(2048, 2048), timeout=3600)
def bench_strict() -> dict:
    """The competition shape: hard 1-core quota, 2 GB."""
    return _bench_impl("strict")


@app.function(image=IMAGE, cpu=(1.0, 4.0), memory=(4096, 4096), timeout=3600)
def bench_burst() -> dict:
    """Control: burst to 4 cores allowed — true compute, no CFS throttle tail."""
    return _bench_impl("burst")


@app.local_entrypoint()
def main():
    calls = [("strict", bench_strict.spawn()), ("burst", bench_burst.spawn())]
    out = {}
    for name, call in calls:
        out[name] = call.get()
        print(f"===== {name} done =====", flush=True)
    print("RESULTS_JSON_BEGIN", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    print("RESULTS_JSON_END", flush=True)
