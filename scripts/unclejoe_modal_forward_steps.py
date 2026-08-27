#!/usr/bin/env python3
"""unclejoe: where the forward pass spends its time, on one x86 core.

    modal run scripts/unclejoe_modal_forward_steps.py > /tmp/uj_steps.log 2>&1 &
    modal app list          # then check startup a few minutes in
    modal app logs <app-id>

Writes docs/research/measurements/unclejoe-forward-steps-modal.json.

`bench --stages` already splits the move into eight cells, and the answer is
always the same: `forward` is the move. This script splits that one cell into
its 25 steps — patchify, embed, the temporal encoder, then per block norm1,
q/k/v, the per-head pack, scores, scale, softmax, context and scatter,
out-proj, both residuals, norm2, ff1, SiLU, ff2, and finally norm_out and the
two heads.

Four containers, one CPU each. Modal does not let a CPU-only function ask for
a CPU generation, so the fleet decides, and identical code has spanned 29-77
ms p50 across runs. `max_inputs=1` retires each container after its input, so
four inputs mean four hosts rather than one host four times, and each result
carries its own `model name` from /proc/cpuinfo and the GEMM kernel the
binary's runtime dispatch picked. Read the per-step *shares* across
containers and the absolute milliseconds only within one.

The binary is compiled once, in the image, so all four containers run the
same bytes and the host is the only variable. Every container runs plain
`bench` first: the total stays comparable to the recorded figures, and a
fleet-generation difference shows up as a shift in a number we already know.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "research" / "measurements"
BOT = REPO / "bots" / "unclejoe"
IN_LOG = REPO / "data" / "joe" / "joe-rs-parity" / "games" / "synthetic-long.in.log"

REMOTE = "/root/unclejoe"
ARTIFACT = f"{REMOTE}/artifact"

# Only what `cargo build --release` and the artifact loader need. The 118 MB
# `model.safetensors` and the 118 MB `ema.eqx` stay home: the container
# rebuilds the safetensors from the 50 MB `model.packed` with the binary's own
# `unpack-artifact`, which verifies both sha256 digests from the manifest.
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential")
    # Network is allowed at image-build time; the bench itself resolves
    # nothing. Toolchain tracks the sandbox's 1.97 stable (port-plan §1).
    # The components and the musl target are named here because
    # `rust-toolchain.toml` asks for them and an install-on-demand inside
    # `cargo build` would need the network again.
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain 1.97.1 "
        "--component rustfmt --component clippy "
        "--target x86_64-unknown-linux-musl"
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir(str(BOT / "src"), f"{REMOTE}/src", copy=True)
    .add_local_file(str(BOT / "Cargo.toml"), f"{REMOTE}/Cargo.toml", copy=True)
    .add_local_file(str(BOT / "Cargo.lock"), f"{REMOTE}/Cargo.lock", copy=True)
    .add_local_file(str(BOT / "rust-toolchain.toml"), f"{REMOTE}/rust-toolchain.toml", copy=True)
    # Carries `-C target-cpu=x86-64-v3` for the Linux targets. Any AVX-512
    # path is still chosen by runtime detection, not by this flag.
    .add_local_file(str(BOT / ".cargo" / "config.toml"), f"{REMOTE}/.cargo/config.toml", copy=True)
    # `src/nn/pack.rs` includes this one under `cfg(test)`. The release build
    # never parses it; it costs 7 KB to keep the path honest.
    .add_local_file(
        str(BOT / "tests" / "fixtures" / "rans_vectors.rs"),
        f"{REMOTE}/tests/fixtures/rans_vectors.rs", copy=True)
    .add_local_file(str(BOT / "artifact" / "manifest.json"), f"{ARTIFACT}/manifest.json", copy=True)
    .add_local_file(str(BOT / "artifact" / "model.packed"), f"{ARTIFACT}/model.packed", copy=True)
    .add_local_file(str(IN_LOG), "/root/synthetic-long.in.log", copy=True)
    # Build and unpack once, in the image: four containers then run the same
    # bytes against the same weights, and neither cost lands in a measurement.
    .run_commands(
        f"cd {REMOTE} && cargo build --release",
        f"cd {REMOTE} && JOE_RS_ARTIFACT={ARTIFACT} ./target/release/unclejoe unpack-artifact",
    )
)

app = modal.App("unclejoe-forward-steps")


# `single_use_containers` retires the container after one input, so four inputs
# land on four hosts instead of one host four times.
@app.function(
    image=IMAGE, cpu=1, memory=4096, timeout=60 * 60, single_use_containers=True)
def bench(replica: int) -> dict:
    import os
    import platform
    import subprocess
    import time

    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
    ):
        os.environ[var] = "1"

    out: dict = {
        "replica": replica,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "online_cpus": os.cpu_count(),
        "cpu_model": "",
        "cpu_mhz": "",
        "avx512f": False,
        "avx2": False,
    }
    # Modal masks `model name` — it reads "unknown". The host class comes from
    # vendor / family / model / stepping instead, the same identification the
    # [AVX-512 gate](docs/research/measurements/joe-rs-avx512-gate.md) used.
    ident: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in ("vendor_id", "cpu family", "model", "stepping") and key not in ident:
            ident[key] = value
        elif key == "model name" and not out["cpu_model"]:
            out["cpu_model"] = value
        elif key == "cpu MHz" and not out["cpu_mhz"]:
            out["cpu_mhz"] = value
        elif key == "flags" and not out["avx2"]:
            flags = set(value.split())
            out["avx2"] = "avx2" in flags and "fma" in flags
            out["avx512f"] = "avx512f" in flags
    out["cpu_ident"] = " ".join(
        f"{k}={ident[k]}" for k in ("vendor_id", "cpu family", "model", "stepping")
        if k in ident)

    env = dict(os.environ)
    env["JOE_RS_ARTIFACT"] = ARTIFACT

    def run(args: list[str]) -> subprocess.CompletedProcess:
        with open("/root/synthetic-long.in.log", "rb") as f:
            return subprocess.run(
                [f"{REMOTE}/target/release/unclejoe", *args],
                stdin=f, capture_output=True, text=True, env=env)

    check = subprocess.run(
        [f"{REMOTE}/target/release/unclejoe", "selfcheck"],
        capture_output=True, text=True, env=env)
    out["selfcheck_rc"] = check.returncode
    out["selfcheck"] = check.stdout.strip()
    if check.returncode != 0:
        out["selfcheck_stderr"] = check.stderr[-3000:]
        return out

    # The uninstrumented whole-move figure first, so the split has something
    # to be read against.
    t0 = time.time()
    plain = run(["bench"])
    out["plain_s"] = round(time.time() - t0, 1)
    out["bench_rc"] = plain.returncode
    out["bench_stdout"] = plain.stdout.strip()
    if plain.returncode != 0:
        out["bench_stderr"] = plain.stderr[-3000:]
        return out

    staged = run(["bench", "--forward-stages"])
    out["steps_rc"] = staged.returncode
    if staged.returncode != 0:
        out["steps_stderr"] = staged.stderr[-3000:]
        return out
    lines = [ln for ln in staged.stdout.strip().splitlines() if ln.strip()]
    out["staged_total_line"] = lines[0]
    for line in lines[1:]:
        blob = json.loads(line)
        out["move_stages" if "stages" in blob else "forward_steps"] = blob
    # The per-turn value telemetry floods stderr; keep only the tables.
    out["tables"] = "\n".join(
        ln for ln in staged.stderr.splitlines() if " value " not in ln)
    return out


@app.local_entrypoint()
def main(containers: int = 4, batches: int = 1) -> None:
    """`--batches N` repeats the fan-out N times, minutes apart.

    One batch of four is the measurement. A second batch exists only because
    the fleet is mixed: a batch can land entirely on AVX2 hosts and say
    nothing about the AVX-512 tile the same binary dispatches to elsewhere.
    Batches are never averaged together — see the fleet-variance caveat in
    the module docstring.
    """
    batch_results = []
    for batch in range(batches):
        results = list(bench.map(range(containers)))
        results.sort(key=lambda r: r["replica"])
        for r in results:
            print(f"\n===== batch {batch} replica {r['replica']}: "
                  f"{r.get('cpu_ident', '?')} @ {r['cpu_mhz']} MHz "
                  f"(avx512f={r['avx512f']}) =====")
            print(r.get("tables", r.get("selfcheck_stderr", "no tables")))
        batch_results.append({"batch": batch, "results": results})
    record = {"containers": containers, "batches": batch_results}
    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    path = MEASUREMENTS / "unclejoe-forward-steps-modal.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {path}")
